"""
Эксперимент (не бот). Две части, 1h-свечи, с 2025-01-01.

ЧАСТЬ 1. «Не входить на вершине» для рабочего условия
  (л/ш аккаунтов ≤ 0.8, OI 3д ≥ +10%, накопл. дельта 7д ≥ 0; вход на подтверждении +6% с объёмом ≥ x3; стоп 12%,
  тейк 10% или 20%, до 48 ч). Фильтры на момент входа: рост цены за 7 дней ≤ +20 / +30 / +50%;
  цена входа не выше 90% от максимума 30 дней. Симуляция депозита как в pre_pump_sim: риск 2%, 3 слота,
  без ночи (вход 23:00–05:00 по Киеву пропускается), стоп с проскальзыванием 1%.

ЧАСТЬ 2. Рисунок «AKE»: тишина → всплеск → возврат в диапазон → затухание.
  Тишина: ширина диапазона (макс. high / мин. low − 1) за Q дней до всплеска ≤ R.
  Всплеск: 1h-свеча с объёмом ≥ x5 к норме и high выше верха диапазона на S и больше.
  Сигнал (свеча i): последний такой всплеск был не раньше чем D дней назад; закрытие снова внутри диапазона
  (не выше верха +5% и не ниже низа −5%); медианный объём последних 12 ч ≤ 1.5 × медианный объём периода тишины.
  Первая свеча, где всё это стало выполняться; пауза 72 ч между сигналами монеты.
  Варианты: Q = 7 / 14 / 21 дней, R = 20% / 35%, S = +10% / +25%, D = 3 / 7 / 10 дней (72 варианта).
  Что считаем: как часто после сигнала памп (high ≥ +30%) и дамп (low ≤ −20%) — за 72 ч и за 7 дней,
  и во сколько раз чаще, чем после любой свечи («подъём»). Перекос вверх = подъём пампов / подъём дампов.
  Сделки (лонг), для каждого варианта:
    A — сразу по закрытию сигнальной свечи;
    E — пробой: стоп-ордер на 0.5% выше верха диапазона, действует 7 дней (вход внутри свечи: если low
        этой же свечи ниже стопа — считается стоп).
    Выходы: стоп 12% / тейк 20% (72 ч); стоп 8% / тейк 16% (72 ч);
            «по структуре»: стоп на 2% ниже низа диапазона (не больше 25%), тейк = 2 × стоп, до 7 дней.
  Комиссия 0.1%.
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
import pre_pump_sim as PS
import pre_pump_trades as TR

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "results/pre_pump_ake"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
H = 3600 * 1000
IH_MS = PP.IH * H
C = PP.C
FEE = 0.1
QS, RS, SS, DS = [7, 14, 21], [0.20, 0.35], [0.10, 0.25], [3, 7, 10]
HORS = {"72ч": C(72), "7д": C(168)}
EXITS = [("стоп 12 / тейк 20, 72 ч", 12.0, 20.0, C(72)), ("стоп 8 / тейк 16, 72 ч", 8.0, 16.0, C(72))]


def path_result(h, l, c, e, px, stop, take, hold, inbar_low=None):
    """(% результата, индекс свечи выхода, вид). None — не хватает данных."""
    if e + hold >= len(c):
        return None
    if inbar_low is not None and inbar_low <= px * (1 - stop / 100):
        return -stop, e, "стоп"
    for k in range(e + 1, e + 1 + hold):
        if l[k] <= px * (1 - stop / 100):
            return -stop, k, "стоп"
        if h[k] >= px * (1 + take / 100):
            return take, k, "тейк"
    return (c[e + hold] / px - 1) * 100, e + hold, "время"


# ---------------- часть 1 ----------------
def part1(sym, f):
    h, l, c = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    t, vx, elig = f.open_time.to_numpy(), f.vol_x.to_numpy(), f.elig.to_numpy()
    n = len(f)
    hi30 = pd.Series(h).rolling(C(720), min_periods=C(168)).max().to_numpy()
    px7 = f.px7d.to_numpy()
    cond = (f.ls_acc <= 0.8) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0)
    rows = []
    for s in TR.signals(cond, elig):
        e = next((j for j in range(s + 1, min(s + 1 + C(72), n)) if c[j] >= c[s] * 1.06 and vx[j] >= 3), None)
        if e is None:
            continue
        row = {"symbol": sym, "entry_time": t[e] + IH_MS, "year": pd.Timestamp(t[s], unit="ms").year,
               "px7d_entry": px7[e], "px7d_signal": px7[s], "to_hi30": c[e] / hi30[e]}
        ok = True
        for tk in (10, 20):
            r = path_result(h, l, c, e, c[e], 12.0, tk, C(48))
            if r is None:
                ok = False
                break
            pct, k, kind = r
            row[f"pct{tk}"], row[f"kind{tk}"], row[f"exit{tk}"] = pct - FEE, kind, t[k] + IH_MS
        if ok:
            rows.append(row)
    return rows


# ---------------- часть 2 ----------------
def part2(sym, f):
    h, l, c, o = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy(), f.open.to_numpy()
    qv, vx, t, elig = f.turnover.to_numpy(), f.vol_x.to_numpy(), f.open_time.to_numpy(), f.elig.to_numpy()
    n = len(f)
    idx = np.arange(n)
    S = pd.Series
    med12 = S(qv).rolling(12, min_periods=6).median().to_numpy()
    # будущее для «подъёма»: базовые доли по всем подходящим свечам
    base = {}
    fut = {}
    for hn, w in HORS.items():
        pad_h = np.concatenate([h, np.full(w, np.nan)])
        pad_l = np.concatenate([l, np.full(w, np.nan)])
        up = np.lib.stride_tricks.sliding_window_view(pad_h, w)[1:n + 1].max(axis=1) / c - 1
        dn = np.lib.stride_tricks.sliding_window_view(pad_l, w)[1:n + 1].min(axis=1) / c - 1
        okw = elig & (idx + w < n)
        fut[hn] = (up >= 0.30, dn <= -0.20, okw)
        base[hn] = (int(okw.sum()), int((okw & (up >= 0.30)).sum()), int((okw & (dn <= -0.20)).sum()))
    sigs = []
    for q in QS:
        wq = C(q * 24)
        rhi = S(h).shift(1).rolling(wq, min_periods=wq).max().to_numpy()
        rlo = S(l).shift(1).rolling(wq, min_periods=wq).min().to_numpy()
        qmed = S(qv).shift(1).rolling(wq, min_periods=wq).median().to_numpy()
        width = rhi / rlo - 1
        for r in RS:
            for sp in SS:
                spike = (vx >= 5) & (h >= rhi * (1 + sp)) & (width <= r)
                last = np.maximum.accumulate(np.where(spike, idx, -1))
                for d in DS:
                    j = last
                    has = (j >= 0) & (idx - j <= C(d * 24)) & (idx > j)
                    jj = np.where(has, j, 0)
                    hi, lo, qm = rhi[jj], rlo[jj], qmed[jj]
                    cond = has & (c <= hi * 1.05) & (c >= lo * 0.95) & (med12 <= 1.5 * qm)
                    var = f"Q{q} R{int(r * 100)} S{int(sp * 100)} D{d}"
                    for s in TR.signals(pd.Series(cond), elig):
                        row = {"symbol": sym, "var": var, "Q": q, "R": int(r * 100), "S": int(sp * 100), "D": d,
                               "signal_time": t[s], "year": pd.Timestamp(t[s], unit="ms").year,
                               "range_hi": hi[s], "range_lo": lo[s], "close": c[s],
                               "spike_time": t[jj[s]], "spike_up": (h[jj[s]] / hi[s] - 1) * 100}
                        for hn in HORS:
                            pu, du, okw = fut[hn]
                            row[f"pump_{hn}"] = bool(pu[s]) if okw[s] else np.nan
                            row[f"dump_{hn}"] = bool(du[s]) if okw[s] else np.nan
                        # сделки
                        lvl_struct = lo[s] * 0.98
                        for entry in ("A сразу", "E пробой"):
                            if entry == "A сразу":
                                e, px, inbar = s, c[s], None
                            else:
                                lvl = hi[s] * 1.005
                                e = next((k for k in range(s + 1, min(s + 1 + C(168), n)) if h[k] >= lvl), None)
                                if e is None:
                                    continue
                                px, inbar = max(lvl, o[e]), l[e]
                            for ename, st, tk, hold in EXITS:
                                res = path_result(h, l, c, e, px, st, tk, hold, inbar)
                                if res is not None:
                                    row[f"{entry}|{ename}"] = res[0] - FEE
                            st = min(max((1 - lvl_struct / px) * 100, 1.0), 25.0)
                            res = path_result(h, l, c, e, px, st, 2 * st, C(168), inbar)
                            if res is not None:
                                row[f"{entry}|по структуре"] = res[0] - FEE
                                row[f"{entry}|стоп по структуре %"] = st
                        sigs.append(row)
    return sigs, base


def coin(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return None
    sigs, base = part2(sym, f)
    return part1(sym, f), sigs, base


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
    btc = pd.DataFrame({"open_time": b.open_time, "btc7d": (b.close / b.close.shift(C(168)) - 1) * 100})
    print(f"часть {SHARD}: монет {len(mine)}", flush=True)

    def work(sym):
        try:
            return coin(sym, uni[sym], btc)
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    p1, p2, bases = [], [], []
    with ThreadPoolExecutor(3) as ex:
        for i, res in enumerate(ex.map(work, mine), 1):
            if res:
                p1 += res[0]
                p2 += res[1]
                bases.append({f"{hn}_{k}": v for hn, (a, b_, c_) in res[2].items() for k, v in zip(("n", "pump", "dump"), (a, b_, c_))})
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.DataFrame(p1).to_csv(f"{OUT}/parts/p1_{SHARD}.csv.gz", index=False)
    pd.DataFrame(p2).to_csv(f"{OUT}/parts/p2_{SHARD}.csv.gz", index=False)
    pd.DataFrame(bases).to_csv(f"{OUT}/parts/base_{SHARD}.csv.gz", index=False)


def read(prefix):
    fs = sorted(f for f in os.listdir(f"{OUT}/parts") if f.startswith(prefix))
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{f}"))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def combine():
    w = open(f"{OUT}/pre_pump_ake_summary.md", "w").write
    w("# Эксперимент: «не входить на вершине» и рисунок «AKE» (1h, с 2025-01-01)\n\n"
      "Не бот — только история. Комиссия 0.1%. Результат сделок — % движения цены (при x2 вдвое больше).\n\n")

    # ---- часть 1
    a = read("p1_").sort_values("entry_time")
    hr = pd.to_datetime(a.entry_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.hour
    a = a[~((hr >= 23) | (hr <= 5))]
    filters = {"без фильтра": pd.Series(True, index=a.index),
               "рост за 7д до входа ≤ +50%": a.px7d_entry <= 50,
               "рост за 7д до входа ≤ +30%": a.px7d_entry <= 30,
               "рост за 7д до входа ≤ +20%": a.px7d_entry <= 20,
               "вход ≤ 90% от максимума 30д": a.to_hi30 <= 0.90,
               "рост 7д ≤ +30% и вход ≤ 90% от макс. 30д": (a.px7d_entry <= 30) & (a.to_hi30 <= 0.90)}
    rows = []
    for name, m in filters.items():
        for tk in (10, 20):
            s, _ = PS.simulate(a[m.fillna(False)], tk, 3, True)
            if s:
                g = a[m.fillna(False)]
                rows.append({"фильтр": name, "тейк %": tk, "сигналов": len(g),
                             "ср. % 2025": round(g[g.year == 2025][f"pct{tk}"].mean(), 2),
                             "ср. % 2026": round(g[g.year == 2026][f"pct{tk}"].mean(), 2), **s})
    w("## Часть 1. Рабочее условие + фильтр «не на вершине»\n\n"
      "л/ш ≤ 0.8, OI 3д ≥ +10%, дельта 7д ≥ 0; вход на подтверждении; стоп 12%, до 48 ч. "
      "Депозит: риск 2%, 3 слота, без ночи, стоп с проскальзыванием 1%.\n\n"
      + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    w("Как распределён рост за 7 дней до входа: " + ", ".join(
        f"до +{x}%: {(a.px7d_entry <= x).mean() * 100:.0f}%" for x in (10, 20, 30, 50, 100)) + "\n\n")

    # ---- часть 2
    s = read("p2_")
    b = read("base_").sum()
    w("## Часть 2. Рисунок «AKE»: тишина → всплеск → возврат в диапазон → затухание\n\n"
      "Q — дней тишины до всплеска, R — макс. ширина диапазона тишины, S — насколько всплеск выше верха диапазона, "
      "D — не дольше скольких дней назад был всплеск.\n\n")
    for hn in HORS:
        bp, bd = b[f"{hn}_pump"] / b[f"{hn}_n"], b[f"{hn}_dump"] / b[f"{hn}_n"]
        w(f"Базовая доля за {hn}: памп +30% после {bp * 100:.2f}% свечей, дамп −20% после {bd * 100:.2f}%.\n\n")
    rows = []
    for var, g in s.groupby("var"):
        r = {"вариант": var, "сигналов": len(g), "монет": g.symbol.nunique()}
        for hn in HORS:
            bp, bd = b[f"{hn}_pump"] / b[f"{hn}_n"], b[f"{hn}_dump"] / b[f"{hn}_n"]
            pu, du = g[f"pump_{hn}"].dropna().astype(float).mean(), g[f"dump_{hn}"].dropna().astype(float).mean()
            r[f"памп {hn}"] = f"{pu * 100:.1f}%"
            r[f"дамп {hn}"] = f"{du * 100:.1f}%"
            r[f"подъём пампов {hn}"] = round(pu / bp, 2)
            r[f"перекос {hn}"] = round((pu / bp) / (du / bd), 2) if du > 0 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("перекос 7д", ascending=False)
    w("### Как часто после сигнала памп / дамп (по перекосу за 7 дней)\n\n" + t.to_markdown(index=False) + "\n\n")

    tcols = [c for c in s.columns if "|" in c and "стоп по структуре %" not in c]
    rows = []
    for var, g in s.groupby("var"):
        for col in tcols:
            x = g[col].dropna()
            x25, x26 = g[g.year == 2025][col].dropna(), g[g.year == 2026][col].dropna()
            if len(x) < 30:
                continue
            entry, ex = col.split("|")
            rows.append({"вариант": var, "вход": entry, "выход": ex, "сделок": len(x),
                         "винрейт": f"{(x > 0).mean() * 100:.0f}%", "ср. %": round(x.mean(), 2),
                         "ср. % 2025": round(x25.mean(), 2) if len(x25) else "-",
                         "ср. % 2026": round(x26.mean(), 2) if len(x26) else "-",
                         "_min": min(x25.mean() if len(x25) >= 10 else -99, x26.mean() if len(x26) >= 10 else -99)})
    tt = pd.DataFrame(rows)
    if len(tt):
        w("### Сделки: лучшие 30 по худшему из двух лет (минимум 30 сделок, по 10 в каждом году)\n\n"
          + tt.sort_values("_min", ascending=False).head(30).drop(columns="_min").to_markdown(index=False) + "\n\n")
        w("### Сделки: лучший вариант для каждого входа и выхода\n\n"
          + tt.loc[tt.groupby(["вход", "выход"])["_min"].idxmax()].drop(columns="_min").to_markdown(index=False) + "\n\n")
        if "стоп по структуре %" in "".join(s.columns):
            for entry in ("A сразу", "E пробой"):
                col = f"{entry}|стоп по структуре %"
                if col in s:
                    w(f"Стоп «по структуре» ({entry}): медиана {s[col].median():.1f}%, "
                      f"четверть сделок со стопом больше {s[col].quantile(0.75):.1f}%.\n\n")
    out = s.copy()
    for col in ("signal_time", "spike_time"):
        out[col] = pd.to_datetime(out[col], unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    out.round(4).to_csv(f"{OUT}/pre_pump_ake_signals.csv.gz", index=False)
    print(open(f"{OUT}/pre_pump_ake_summary.md").read()[:5000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
