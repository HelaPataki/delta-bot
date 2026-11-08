"""
Симуляция депозита: «толпа в шортах + OI + дельта», вход на подтверждении (1h-свечи).

Сигнал: л/ш аккаунтов ≤ 0.8 (и для сравнения ≤ 1), OI за 3д ≥ +10%, накопленная дельта 7д ≥ 0 —
первая свеча, где условие стало выполняться; пауза 72 ч между сигналами монеты.
Вход: в течение 72 ч первая 1h-свеча, закрывшаяся на +6% выше цены сигнала с объёмом >= x3; вход по её закрытию.
Выходы: стоп 12% и тейк 8 / 10 / 15 / 20%, не дольше 48 ч (потом по закрытию). Стоп и тейк в одной свече — стоп.
Комиссия 0.1%. Вариант «проскальзывание»: стоп исполняется на 1% хуже (−13%).

Депозит: старт 100, риск 2% текущего депозита на сделку (позиция = 2% / 12% ≈ 17% депозита по номиналу,
при x2 маржа ≈ 8%). Слотов 2 / 3 / 5: если все заняты — сигнал пропускается. Одна позиция на монету.
«Без ночи»: пропускать входы, свеча которых закрылась с 23:00 до 05:00 по Киеву включительно.
"""

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("INTERVAL", "1h")

import numpy as np
import pandas as pd

import history as Hst
import pre_pump as PP
import pre_pump_trades as TR

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "results/pre_pump_sim"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
WAIT, HOLD = PP.C(72), PP.C(48)
STOP = 12.0
TAKES = [8, 10, 15, 20]
FEE = 0.1
SLIP = 1.0
RISK = 2.0
SLOTS = [2, 3, 5]
H = 3600 * 1000


def coin(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return None
    h, l, c = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    t, vx, elig = f.open_time.to_numpy(), f.vol_x.to_numpy(), f.elig.to_numpy()
    n = len(f)
    rows = []
    for cname, lim in (("л/ш ≤ 0.8", 0.8), ("л/ш ≤ 1", 1.0)):
        cond = (f.ls_acc <= lim) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0)
        for s in TR.signals(cond, elig):
            e = next((j for j in range(s + 1, min(s + 1 + WAIT, n)) if c[j] >= c[s] * 1.06 and vx[j] >= 3), None)
            if e is None or e + HOLD >= n:
                continue
            px, hw, lw = c[e], h[e + 1:e + 1 + HOLD], l[e + 1:e + 1 + HOLD]
            s_hit = lw <= px * (1 - STOP / 100)
            si = int(np.argmax(s_hit)) if s_hit.any() else 10 ** 6
            row = {"symbol": sym, "cond": cname,
                   "entry_time": t[e] + IH_MS, "year": pd.Timestamp(t[s], unit="ms").year}
            for tk in TAKES:
                t_hit = hw >= px * (1 + tk / 100)
                ti = int(np.argmax(t_hit)) if t_hit.any() else 10 ** 6
                if si < HOLD and si <= ti:
                    pct, k, kind = -STOP, si, "стоп"
                elif ti < HOLD:
                    pct, k, kind = tk, ti, "тейк"
                else:
                    pct, k, kind = (c[e + HOLD] / px - 1) * 100, HOLD - 1, "время"
                row[f"pct{tk}"] = round(pct - FEE, 3)
                row[f"kind{tk}"] = kind
                row[f"exit{tk}"] = t[e + 1 + k] + IH_MS          # закрытие свечи выхода
            rows.append(row)
    return pd.DataFrame(rows)


IH_MS = PP.IH * H


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps[Hst.base_asset(s)] for s in syms if Hst.base_asset(s) in caps}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]
    if LIMIT:
        mine = mine[:LIMIT]
    b = Hst.load_klines("BTCUSDT", PP.INTERVAL)
    btc = pd.DataFrame({"open_time": b.open_time, "btc7d": (b.close / b.close.shift(PP.C(168)) - 1) * 100})
    print(f"часть {SHARD}: монет {len(mine)}", flush=True)

    def work(sym):
        try:
            return coin(sym, uni[sym], btc)
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(3) as ex:
        for i, p in enumerate(ex.map(work, mine), 1):
            if p is not None and len(p):
                parts.append(p)
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.concat(parts).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)


def simulate(tr, tk, slots, slip):
    """tr — сделки по времени входа. Возвращает итоги, помесячно и журнал."""
    eq, peak, mdd = 100.0, 100.0, 0.0
    open_pos = []                      # (exit_time, symbol, risk_amount, R, kind)
    log, taken, skipped = [], 0, 0
    streak = best_streak = 0

    def close_until(ts):
        nonlocal eq, peak, mdd, streak, best_streak
        open_pos.sort()
        while open_pos and open_pos[0][0] <= ts:
            ext, sym, risk_amt, R, kind = open_pos.pop(0)
            pnl = risk_amt * R
            eq += pnl
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
            streak = streak + 1 if pnl < 0 else 0
            best_streak = max(best_streak, streak)
            log.append({"exit_time": ext, "symbol": sym, "result": kind, "pnl": pnl, "equity": eq})

    for r in tr.itertuples():
        close_until(r.entry_time)
        if len(open_pos) >= slots or any(p[1] == r.symbol for p in open_pos):
            skipped += 1
            continue
        pct = getattr(r, f"pct{tk}")
        kind = getattr(r, f"kind{tk}")
        if slip and kind == "стоп":
            pct -= SLIP
        open_pos.append((getattr(r, f"exit{tk}"), r.symbol, eq * RISK / 100, pct / STOP, kind))
        taken += 1
    close_until(float("inf"))
    lg = pd.DataFrame(log)
    if len(lg) == 0:
        return None, None
    lg["month"] = pd.to_datetime(lg.exit_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m")
    monthly = lg.groupby("month").agg(сделок=("pnl", "size"), депозит_в_конце=("equity", "last"))
    prev = monthly.депозит_в_конце.shift(1).fillna(100)
    monthly["за месяц %"] = ((monthly.депозит_в_конце / prev - 1) * 100).round(1)
    wins = (lg.pnl > 0).mean() * 100
    summary = {"сделок": taken, "пропущено (слоты заняты)": skipped, "винрейт": f"{wins:.0f}%",
               "итог депозита": f"{eq - 100:+.0f}%", "макс. просадка": f"{mdd * 100:.0f}%",
               "серия убытков": best_streak,
               "месяцев в плюсе": f"{(monthly['за месяц %'] > 0).sum()} из {len(monthly)}",
               "худший месяц": f"{monthly['за месяц %'].min():.1f}%"}
    return summary, monthly


def combine():
    files = sorted(f for f in os.listdir(f"{OUT}/parts") if f.endswith(".csv.gz"))
    a = pd.concat([pd.read_csv(f"{OUT}/parts/{f}") for f in files], ignore_index=True).sort_values("entry_time")
    hr = pd.to_datetime(a.entry_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.hour
    a["night"] = (hr >= 23) | (hr <= 5)
    w = open(f"{OUT}/pre_pump_sim_summary.md", "w").write
    w("# Симуляция депозита: «толпа в шортах + OI + дельта», вход на подтверждении (1h)\n\n"
      f"Старт 100, риск {RISK:g}% текущего депозита на сделку, стоп {STOP:g}%, держать до 48 ч. "
      "Позиция ≈ 17% депозита по номиналу (при x2 маржа ≈ 8%). Слоты заняты — сигнал пропускается. "
      "Одна позиция на монету. Комиссия 0.1%. «Без ночи» — пропуск входов с закрытием свечи 23:00–05:00 по Киеву. "
      f"«Проскальзывание» — стоп исполняется на {SLIP:g}% хуже.\n\n")
    rows, months = [], {}
    for cond in ("л/ш ≤ 0.8", "л/ш ≤ 1"):
        for night in (False, True):
            for slip in (False, True):
                for tk in TAKES:
                    for sl in SLOTS:
                        tr = a[(a.cond == cond) & (~a.night if night else True)]
                        s, m = simulate(tr, tk, sl, slip)
                        if s is None:
                            continue
                        rows.append({"условие": cond, "ночь": "без ночи" if night else "все", "проскальз.": "да" if slip else "нет",
                                     "тейк %": tk, "слотов": sl, **s})
                        if cond == "л/ш ≤ 0.8" and sl == 3 and night and slip:
                            months[tk] = m
    t = pd.DataFrame(rows)
    w("## Главное: л/ш ≤ 0.8, без ночи, с проскальзыванием\n\n"
      + t[(t.условие == "л/ш ≤ 0.8") & (t.ночь == "без ночи") & (t["проскальз."] == "да")]
      .drop(columns=["условие", "ночь", "проскальз."]).to_markdown(index=False) + "\n\n")
    w("## Все варианты\n\n" + t.to_markdown(index=False) + "\n\n")
    for tk, m in months.items():
        w(f"## По месяцам: тейк {tk}%, 3 слота, без ночи, с проскальзыванием\n\n" + m.round(1).to_markdown() + "\n\n")
    out = a.copy()
    out["entry_time"] = pd.to_datetime(out.entry_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    for tk in TAKES:
        out[f"exit{tk}"] = pd.to_datetime(out[f"exit{tk}"], unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    out.to_csv(f"{OUT}/pre_pump_sim_trades.csv", index=False)
    print(open(f"{OUT}/pre_pump_sim_summary.md").read()[:4000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
