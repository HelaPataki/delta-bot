"""
Эксперимент (не бот): «коробка → пробой». 1h-свечи, с 2025-01-01, капа $10M–400M, оборот свечи от $50K.

Коробка на момент свечи k: максимум high (верх) и минимум low (низ) за Q дней ДО этой свечи;
ширина = верх / низ − 1 не больше W.
Вход (лонг):
  «стоп-ордер» — свеча k, где high дошёл до верх × 1.005; вход по этому уровню (или по открытию, если открылась
                 выше). Вход внутри свечи: если low этой же свечи ниже стопа — считается стоп.
  «закрытие»   — свеча k закрылась выше верха коробки с объёмом ≥ x3 к норме; вход по закрытию.
Фильтр «толпа в шортах»: лонг/шорт аккаунтов ≤ 1 на последнем известном закрытии перед входом.
Повторный пробой той же монеты — не раньше чем через 72 ч.
Выходы:
  «стоп 12 / тейк 20, 72 ч»;
  «стоп под коробкой / тейк 20, 7 д» — стоп на 1% ниже низа коробки (не дальше 30% от входа);
  «стоп под коробкой / тейк 2× стоп, 7 д».
Варианты: Q = 7 / 10 / 14 дней, W = 20 / 30 / 40%, 2 входа, с фильтром и без (36 вариантов × 3 выхода).
Комиссия 0.1%. Просадка — при риске 2% депозита на сделку, сделки по порядку входа (пересечения не ограничены).
"""

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("INTERVAL", "1h")
os.environ.setdefault("START", "2024-07-01")
os.environ.setdefault("SIG_FROM", "2025-01-01")

import numpy as np
import pandas as pd

import history as Hst
import pre_pump as PP
import pre_pump_ake as AK
import pre_pump_exits as PE
import pre_pump_trades as TR

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "results/pre_pump_box"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
C, H = PP.C, 3600 * 1000
QS, WS = [7, 10, 14], [0.20, 0.30, 0.40]
FEE, RISK = 0.1, 2.0
MIN_N = int(os.environ.get("MIN_N", "20"))   # минимум сделок в каждом году
EX_FROM = pd.Timestamp("2026-09-01", tz=Hst.KYIV).timestamp() * 1000


def coin(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return []
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    vx, t, elig, ls = f.vol_x.to_numpy(), f.open_time.to_numpy(), f.elig.to_numpy(), f.ls_acc.to_numpy()
    n = len(f)
    S = pd.Series
    rows = []
    for q in QS:
        wq = C(q * 24)
        top = S(h).shift(1).rolling(wq, min_periods=wq).max().to_numpy()
        bot = S(l).shift(1).rolling(wq, min_periods=wq).min().to_numpy()
        width = top / bot - 1
        for wmax in WS:
            box = width <= wmax
            for entry, ev in (("стоп-ордер", box & (h >= top * 1.005)),
                              ("закрытие", box & (c > top) & (vx >= 3))):
                for k in TR.signals(S(ev), elig):
                    if entry == "стоп-ордер":
                        lvl = top[k] * 1.005
                        px, inbar, ls_known = max(lvl, o[k]), l[k], ls[k - 1]
                    else:
                        px, inbar, ls_known = c[k], None, ls[k]
                    row = {"symbol": sym, "Q": q, "W": int(wmax * 100), "entry": entry,
                           "crowd_short": bool(ls_known <= 1) if not np.isnan(ls_known) else False,
                           "t": t[k], "year": pd.Timestamp(t[k], unit="ms").year,
                           "box_top": top[k], "box_bot": bot[k], "px": px, "over_top": (px / top[k] - 1) * 100}
                    r = AK.path_result(h, l, c, k, px, 12.0, 20.0, C(72), inbar)
                    if r is None:
                        continue
                    row["стоп 12 / тейк 20, 72 ч"] = r[0] - FEE
                    row["stop12_R"] = (r[0] - FEE) / 12
                    st = min((1 - bot[k] * 0.99 / px) * 100, 30.0)
                    row["стоп под коробкой, %"] = st
                    for name, tk in (("стоп под коробкой / тейк 20, 7 д", 20.0), ("стоп под коробкой / тейк 2× стоп, 7 д", 2 * st)):
                        r = AK.path_result(h, l, c, k, px, st, tk, C(168), inbar)
                        if r is None:
                            break
                        row[name] = r[0] - FEE
                        row[name + " R"] = (r[0] - FEE) / st
                    else:
                        w7 = h[k + 1:k + 1 + C(168)]
                        row["пик за 7д, %"] = (w7.max() / px - 1) * 100
                        rows.append(row)
    return rows


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
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(work, mine), 1):
            rows += r
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)


EXITS = [("стоп 12 / тейк 20, 72 ч", "stop12_R"),
         ("стоп под коробкой / тейк 20, 7 д", "стоп под коробкой / тейк 20, 7 д R"),
         ("стоп под коробкой / тейк 2× стоп, 7 д", "стоп под коробкой / тейк 2× стоп, 7 д R")]


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True).sort_values("t")
    months = {y: max((g.t.max() - g.t.min()) / 86_400_000 / 30.4, 1) for y, g in a.groupby("year")}
    rows = []
    for flt in (False, True):
        b = a[a.crowd_short] if flt else a
        for (q, wv, entry), g in b.groupby(["Q", "W", "entry"]):
            for ex, rcol in EXITS:
                x25, x26 = g[g.year == 2025][ex], g[g.year == 2026][ex]
                if len(x25) < MIN_N or len(x26) < MIN_N:
                    continue
                r = g[rcol].to_numpy()
                rows.append({"коробка": f"{q} д, ≤ {wv}%", "вход": entry, "толпа в шортах": "да" if flt else "нет",
                             "выход": ex, "сделок/мес": round(len(g) / sum(months.values()), 1),
                             "винрейт 25": f"{(x25 > 0).mean() * 100:.0f}%", "винрейт 26": f"{(x26 > 0).mean() * 100:.0f}%",
                             "ср. % 25": round(x25.mean(), 2), "ср. % 26": round(x26.mean(), 2),
                             "ср. R": round(r.mean(), 2), "серия убытков": PE.streak(g[ex].to_numpy()),
                             "просадка при риске 2%": f"{PE.max_dd(r):.0f}%",
                             "пик за 7д (медиана)": round(g["пик за 7д, %"].median(), 1),
                             "_min": np.nanmin([x25.mean(), x26.mean()])})
    t = pd.DataFrame(rows)
    if t.empty:
        t = pd.DataFrame(columns=["_min"])
    w = open(f"{OUT}/pre_pump_box_summary.md", "w").write
    w("# Эксперимент «коробка → пробой» (1h, с 2025-01-01; не бот)\n\n"
      "Коробка — максимум и минимум за N дней до свечи, ширина не больше W. Лонг на пробое верха. "
      "Комиссия 0.1%. % — движение цены (при x2 вдвое больше). R — результат в долях стопа. "
      "Просадка — при риске 2% депозита на сделку, сделки по порядку (пересечения не ограничены). "
      "Пик за 7д — максимум цены за 7 дней после входа, % от входа (сколько вообще давал пробой).\n\n")
    w("## Лучшие 25 по худшему из двух лет\n\n" + t.sort_values("_min", ascending=False).head(25)
      .drop(columns="_min").to_markdown(index=False) + "\n\n")
    w("## Все варианты\n\n" + t.drop(columns="_min").to_markdown(index=False) + "\n\n")
    w(f"Стоп под коробкой: медиана {a['стоп под коробкой, %'].median():.1f}%, "
      f"у четверти сделок больше {a['стоп под коробкой, %'].quantile(0.75):.1f}%.\n")
    # сделки с сентября — для проверки на графиках
    ex = a[(a.t >= EX_FROM) & (a.Q == 10) & (a.W == 30) & (a.entry == "стоп-ордер")].copy()
    ex["свеча входа (TradingView, 1Ч)"] = pd.to_datetime(ex.t, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    ex = ex.rename(columns={"symbol": "монета", "box_bot": "низ коробки", "box_top": "верх коробки", "px": "вход",
                            "crowd_short": "толпа в шортах"})
    cols = ["монета", "свеча входа (TradingView, 1Ч)", "низ коробки", "верх коробки", "вход", "толпа в шортах",
            "стоп 12 / тейк 20, 72 ч", "стоп под коробкой, %", "стоп под коробкой / тейк 20, 7 д", "пик за 7д, %"]
    open(f"{OUT}/box_trades_since_sep.md", "w").write(
        "# Пробои коробки с сентября 2026 (коробка 10 дней, ширина ≤ 30%, стоп-ордер над верхом)\n\n"
        "Время — как на TradingView (время открытия часовой свечи, по Киеву). Результаты — % цены после комиссии.\n\n"
        + ex[cols].round(6).to_markdown(index=False) + "\n")
    out = a.copy()
    out["t"] = pd.to_datetime(out.t, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    out.round(5).to_csv(f"{OUT}/pre_pump_box_trades.csv.gz", index=False)
    print(open(f"{OUT}/pre_pump_box_summary.md").read()[:4000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
