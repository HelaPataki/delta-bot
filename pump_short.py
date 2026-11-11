"""
Шорт после пампа: сравнение входов и выходов на тех же вершинах волн, что в pump_life.py.
Живого бота не трогает.

Вершина волны и момент M (откат −20%, закрытие 1h-свечи) — как в pump_life.py.
Фильтры (только данные до входа):
  все | спот (на Binance есть спот и в волне был спотовый объём) | нет спота
  толпа в лонге: все аккаунты L/S ≥ 1.46 и капа на вершине ≥ $130M
  толпа в лонге сильно: L/S ≥ 2 и капа ≥ $130M
  спот + OI за откат ≤ −11%
Входы:
  A  «сразу»: шорт по закрытию свечи M, стоп = вершина +0,5%.
  B  «узкий стоп»: шорт по закрытию M, стоп = вход +10%.
  C  «отскок к середине»: после M ждём отскок к середине между вершиной и минимумом после M
     (лимитка, до 7 дней), стоп = вершина +0,5%.
  D  «сутки без хая»: шорт по закрытию через 24 ч после M, если хай не обновлён; стоп = вершина +0,5%.
Выходы (для каждого входа): тейк −10 / −15 / −20 / −30% от входа; не дошло за 10 дней — закрытие по рынку.
Если стоп и тейк в одной часовой свече — стоп. Учтены: комиссия 0,1% на круг и фандинг (шорт получает
положительный, платит отрицательный). Одна сделка на монету одновременно (пока открыта — новые вершины
этой монеты пропускаются отдельно для каждого варианта).
Результат в % от цены и в R (R = размер стопа). Депозит-эффект: «риск 2% на сделку».
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-12-01")

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/pump_short")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = PL.H
HOR = 240
TAKES = [10, 15, 20, 30]
FEE = 0.1
ENTRIES = ["A сразу", "B стоп +10%", "C отскок к середине", "D сутки без хая"]


def coin(sym, cap):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < PL.W + 200:
        return []
    eps = PL.find_episodes(k)
    if not eps:
        return []
    h, l, c = (k[x].to_numpy() for x in ("high", "low", "close"))
    t, qv = k.open_time.to_numpy(), k.quote_volume.to_numpy()
    n, c_now = len(k), c[-1]
    tops = []
    for ep in eps:
        for ws, hi, m in PL.find_waves(k, ep["s"], ep["e"]):
            tops.append((ws, hi, m))
    if not tops:
        return []
    day = lambda ms: pd.Timestamp(ms, unit="ms", tz="UTC").date()
    need = set()
    for ws, hi, m in tops:
        for q in (ws, hi, m):
            need.add(day(t[q]))
            need.add(day(t[q] - 3 * H))
            need.add(day(t[q] + 2 * H))
    met = PL.metrics(sym, need)
    fund = PL.funding(sym, min(need).replace(day=1))
    spot = PL.klines_any("spot", Hst.base_asset(sym) + "USDT", min(need), max(need) + dt.timedelta(days=1))
    sp_qv = pd.Series(spot.quote_volume.to_numpy(), index=spot.open_time.to_numpy()) if spot is not None else None
    OI, LS = PL.AsOf(met, "oi"), PL.AsOf(met, "glob")
    ft = fund.t.to_numpy() if fund is not None else np.array([])
    fr = fund.rate8.to_numpy() * fund.t.diff().fillna(8 * H).to_numpy() / (8 * H) if fund is not None else np.array([])

    def fund_paid(t0, t1):                       # % цены, который шорт получил (+) / заплатил (−)
        if not len(ft):
            return 0.0
        a, b = np.searchsorted(ft, t0, side="right"), np.searchsorted(ft, t1, side="right")
        return float(fr[a:b].sum())

    def trade(i_entry, entry, stop):
        """Шорт с закрытия свечи i_entry. Возвращает список (тейк, pnl%, R, часы)."""
        hh, ll = h[i_entry + 1:i_entry + 1 + HOR], l[i_entry + 1:i_entry + 1 + HOR]
        if len(hh) < HOR:
            return None
        risk = (stop / entry - 1) * 100
        si = np.nonzero(hh >= stop)[0]
        si = si[0] if len(si) else 10 ** 6
        out = {}
        for X in TAKES:
            tk = np.nonzero(ll <= entry * (1 - X / 100))[0]
            tk = tk[0] if len(tk) else 10 ** 6
            if si == tk == 10 ** 6:
                j, pnl = HOR - 1, (1 - c[i_entry + HOR] / entry) * 100
            elif si <= tk:
                j, pnl = si, -risk
            else:
                j, pnl = tk, float(X)
            pnl += fund_paid(t[i_entry] + H, t[i_entry + 1 + j] + H) - FEE
            out[X] = (pnl, pnl / risk, i_entry + 1 + j)
        return out, risk

    rows = []
    busy = {(e, X): -1 for e in ENTRIES for X in TAKES}
    for ws, hi, m in tops:
        top = h[hi]
        t_top, t_m = t[hi] + H, t[m] + H
        s_v = float(sp_qv.reindex(t[ws:hi + 1]).fillna(0).sum()) if sp_qv is not None else 0.0
        oi_pb = PL.ratio(OI.at(t_m), OI.at(t_top)) * 100
        ls = LS.at(t_m)
        capt = cap * top / c_now / 1e6 if cap else np.nan
        base = {"монета": sym, "вершина": PL.kyiv(t[hi]), "момент −20%": PL.kyiv(t[m]),
                "спот": s_v > 0, "OI за откат, %": oi_pb, "все L/S": ls, "капа $M": capt}
        stop_top = top * 1.005
        plans = {"A сразу": (m, c[m], stop_top), "B стоп +10%": (m, c[m], c[m] * 1.10)}
        # C: отскок к середине
        lo, fill = l[m], None
        for q in range(m + 1, min(n, m + 1 + 168)):
            mid = lo + 0.5 * (top - lo)
            if h[q] >= mid:
                fill = (q, mid)
                break
            lo = min(lo, l[q])
        if fill:
            plans["C отскок к середине"] = (fill[0], fill[1], stop_top)
        # D: сутки без хая
        if m + 24 < n and h[m + 1:m + 25].max() <= top:
            plans["D сутки без хая"] = (m + 24, c[m + 24], stop_top)
        for name, (ie, entry, stop) in plans.items():
            if stop <= entry or ie + HOR >= n:
                continue
            res = trade(ie, entry, stop)
            if res is None:
                continue
            out, risk = res
            for X, (pnl, R, iexit) in out.items():
                if ie <= busy[(name, X)]:
                    continue
                busy[(name, X)] = iexit
                rows.append({**base, "вход": name, "тейк %": X, "время входа": PL.kyiv(t[ie]),
                             "стоп %": risk, "pnl %": pnl, "R": R, "часов в сделке": iexit - ie})
    return rows


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]

    def work(sym):
        try:
            return coin(sym, uni[sym])
        except Exception as e:
            print(f"{sym}: {e!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/trades{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, сделок {len(rows)}")


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    try:
        caps = Hst.coingecko_mcaps()
    except Exception as e:
        print("CoinGecko:", e)
        caps = {}
    json.dump({s: caps.get(Hst.base_asset(s)) for s in syms}, open(f"{OUT}/universe.json", "w"))


def filters(a):
    spot = a["спот"].astype(bool)
    crowd = (a["все L/S"] >= 1.46) & (a["капа $M"] >= 130)
    return {"все": a.index == a.index, "спот": spot, "нет спота": ~spot,
            "толпа в лонге (L/S≥1,46, капа≥130M)": crowd,
            "толпа в лонге сильно (L/S≥2, капа≥130M)": (a["все L/S"] >= 2) & (a["капа $M"] >= 130),
            "спот + толпа в лонге": spot & crowd,
            "спот + OI за откат ≤ −11%": spot & (a["OI за откат, %"] <= -11)}


def sim_deposit(g, risk_pct=2.0):
    """Последовательно по времени входа, риск 2% депозита на сделку (без пересечений не проверяем)."""
    g = g.sort_values("время входа")
    eq, peak, mdd, streak, worst = 100.0, 100.0, 0.0, 0, 0
    for R in g.R:
        eq *= 1 + risk_pct / 100 * R
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
        streak = streak + 1 if R < 0 else 0
        worst = max(worst, streak)
    return eq, mdd * 100, worst


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True)
    a["год"] = a["время входа"].str[:4]
    a.to_csv(f"{OUT}/trades.csv.gz", index=False)
    months = (pd.to_datetime(a["время входа"]).max() - pd.to_datetime(a["время входа"]).min()).days / 30.4
    F = filters(a)
    rows = []
    for fname, fm in F.items():
        for e in ENTRIES:
            for X in TAKES:
                g = a[fm & (a["вход"] == e) & (a["тейк %"] == X)]
                if len(g) < 30:
                    continue
                r = {"фильтр": fname, "вход": e, "тейк %": X, "сделок": len(g), "в месяц": round(len(g) / months, 1),
                     "винрейт": f"{(g.R > 0).mean() * 100:.0f}%", "стоп ср %": round(g["стоп %"].mean(), 1),
                     "ср pnl %": round(g["pnl %"].mean(), 2), "ср R": round(g.R.mean(), 2)}
                for y in ("2025", "2026"):
                    gy = g[g["год"] == y]
                    r[f"R {y}"] = round(gy.R.mean(), 2) if len(gy) >= 15 else None
                    r[f"n {y}"] = len(gy)
                eq, mdd, worst = sim_deposit(g)
                r["депозит 100→ (риск 2%)"] = round(eq)
                r["макс просадка %"] = round(mdd)
                r["стопов подряд"] = worst
                rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/pump_short_table.csv", index=False)
    with open(f"{OUT}/pump_short_summary.md", "w") as f:
        f.write("# Шорт после пампа: входы × выходы × фильтры\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
        f.write("## Лучшие варианты (R положительный в обоих годах, сортировка по худшему году)\n\n")
        ok = d.dropna(subset=["R 2025", "R 2026"]).copy()
        ok["хуже год"] = ok[["R 2025", "R 2026"]].min(axis=1)
        f.write(ok[ok["хуже год"] > 0].sort_values("хуже год", ascending=False).head(40).to_markdown(index=False) + "\n\n")
        f.write("## Вся таблица\n\n" + d.to_markdown(index=False) + "\n")
    print(open(f"{OUT}/pump_short_summary.md").read()[:8000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
