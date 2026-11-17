"""
Лонг-сетап блогера: то же самое, что blogger_long.py, но RSI на разных таймфреймах. Только история.

Всё как в blogger_long (объём спота, VWAP, толпа, OI 4ч, баллы 0–3, скор до 15, вход/выходы, кулдаун 24ч),
меняется только RSI(14) и его таймфрейм — пороги баллов те же: ≥60 → 1, ≥70 → 2, ≥80 → 3:
  • 15м — контроль (должно совпасть с первым прогоном);
  • 1ч и 4ч — «живой» RSI на момент закрытия 15m-свечи: закрытые часовые/4ч свечи + текущая
    недосформированная (как его видно на TradingView в реальном времени); 4ч-свечи — от 00:00 UTC (как на Binance).
Каждый вариант — свой поток сигналов со своим кулдауном.
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import blogger_long as B
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/blogger_rsi")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
M15, H, DAY = B.M15, B.H, B.DAY
TFS = {"RSI 15м": 1, "RSI 1ч": 4, "RSI 4ч": 16}


def live_rsi(t, c, k, n=14):
    """RSI таймфрейма k×15m на закрытии каждой 15m-свечи (с текущей недосформированной свечой)."""
    cf = pd.Series(c).ffill().to_numpy()
    if k == 1:
        return B.rsi(cf, n)
    hc = np.where((t + M15) % (k * M15) == 0)[0]
    cc = cf[hc]
    d = pd.Series(cc).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()
    j = np.searchsorted(hc, np.arange(len(c)), side="right") - 1
    ok = j >= 0
    jj = np.where(ok, j, 0)
    closed = hc[jj] == np.arange(len(c))
    dlt = cf - cc[jj]
    u = np.where(closed, up[jj], (up[jj] * (n - 1) + np.maximum(dlt, 0)) / n)
    w = np.where(closed, dn[jj], (dn[jj] * (n - 1) + np.maximum(-dlt, 0)) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(w > 0, 100 - 100 / (1 + u / w), 100.0)
    r[~ok | ~np.isfinite(u) | ~np.isfinite(w)] = np.nan
    return r


def coin(sym):
    d1 = dt.datetime.now(dt.timezone.utc).date()
    d_pre = B.D0 - dt.timedelta(days=8)
    sp = B.kl15("spot", sym, d_pre, d1)
    if (sp is None or len(sp) < 2000) and sym.startswith("1000"):
        sp = B.kl15("spot", sym[7:] if sym.startswith("1000000") else sym[4:], d_pre, d1)
    if sp is None or len(sp) < 2000:
        return []
    fu = B.kl15("um", sym, d_pre, d1)
    if fu is None or len(fu) < 2000:
        return []
    t0, t1 = int(fu.open_time.min()), int(fu.open_time.max())
    grid = pd.DataFrame({"open_time": np.arange(t0, t1 + M15, M15, dtype="int64")})
    k = grid.merge(fu, on="open_time", how="left")
    k = k.merge(sp[["open_time", "quote_volume"]].rename(columns={"quote_volume": "spot_vol"}), on="open_time", how="left")
    t = k.open_time.to_numpy()
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    v = k.volume.to_numpy()

    sv = pd.Series(k.spot_vol.to_numpy())
    vol24 = sv.rolling(DAY, min_periods=DAY).sum()
    prev3 = sv.shift(DAY).rolling(3 * DAY, min_periods=3 * DAY).sum() / 3
    vx = (vol24 / prev3.replace(0, np.nan)).to_numpy()
    day = t // (24 * H)
    tp_ = (h + l + c) / 3
    pv = pd.Series(np.nan_to_num(tp_ * v)).groupby(day).cumsum().to_numpy()
    vv = pd.Series(np.nan_to_num(v)).groupby(day).cumsum().to_numpy()
    vwap = np.where(vv > 0, pv / np.where(vv > 0, vv, 1), np.nan)
    vw = (c / vwap - 1) * 100
    low24 = pd.Series(l).rolling(DAY, min_periods=DAY // 2).min().to_numpy()
    rs = {name: live_rsi(t, c, kk) for name, kk in TFS.items()}

    g_vx, g_vw = B.grade(vx, [2, 3, 4]), B.grade(vw, [0, 2, 4])
    g_rs = {name: B.grade(r, [60, 70, 80]) for name, r in rs.items()}
    base = (t >= pd.Timestamp(B.D0, tz="UTC").timestamp() * 1000) & (g_vx > 0) & (g_vw > 0)
    anyr = np.zeros(len(t), bool)
    for g in g_rs.values():
        anyr |= g > 0
    cand = np.where(base & anyr)[0]
    if not len(cand):
        return []
    days = {pd.Timestamp(int(t[i]), unit="ms", tz="UTC").date() + dt.timedelta(days=d) for i in cand for d in (-1, 0)}
    met = PL.metrics(sym, days)
    oi, crowd = PL.AsOf(met, "oi_usd"), PL.AsOf(met, "glob")

    cache, rows = {}, []
    for name in TFS:
        last = -10 ** 15
        for i in cand:
            if g_rs[name][i] == 0:
                continue
            tc = int(t[i]) + M15
            if tc - last < B.COOL:
                continue
            if i not in cache:
                o_now, o_4h = oi.at(tc, 20 * 60 * 1000), oi.at(tc - 4 * H, 20 * 60 * 1000)
                oich = (o_now / o_4h - 1) * 100 if o_now and o_4h and np.isfinite(o_now) and np.isfinite(o_4h) else np.nan
                cr = crowd.at(tc, 20 * 60 * 1000)
                g_oi, g_cr = B.grade([oich], [0, 1, 3])[0], B.grade([cr], [2.5, 2.0, 1.5], lower=True)[0]
                r = B.simulate(o, h, l, c, i + 1, low24[i]) if g_oi and g_cr else None
                cache[i] = (oich, cr, g_oi, g_cr, r)
            oich, cr, g_oi, g_cr, r = cache[i]
            if r is None:
                continue
            last = tc
            rows.append({"вариант": name, "symbol": sym, "t": tc,
                         "скор": int(g_vx[i] + g_rs[name][i] + g_vw[i] + g_oi + g_cr),
                         "спот x": vx[i], **{nm: rr[i] for nm, rr in rs.items()}, "VWAP, %": vw[i], "толпа": cr,
                         "OI 4ч, %": oich, "цена 24ч, %": (c[i] / c[i - DAY] - 1) * 100 if i >= DAY else np.nan, **r})
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    B.OUT = OUT
    B.prep()


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/sig{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} сигналов, монет {len(mine)}")


KEY = ["стоп 24ч / тейк 6", "стоп 24ч / тейк 15", "стоп 8 / тейк 10", "стоп 5 / тейк 10",
       "лесенка, стоп 24ч", "лесенка + безубыток после TP1"]


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["время (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/signals.csv.gz", index=False)
    weeks = (ev.t.max() - ev.t.min()) / (7 * 24 * H) or 1
    w = open(f"{OUT}/blogger_rsi_summary.md", "w").write
    w("# Сетап блогера: RSI на разных таймфреймах\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Период: {PL.kyiv(ev.t.min())[:10]} — {PL.kyiv(ev.t.max())[:10]} ({weeks:.0f} нед.). Результаты — % цены на "
      "сделку после комиссии 0,1% (2x — умножь на 2). PF > 1 — в плюсе.\n\n")

    groups = [("скор ≥ 13", lambda e: e[e.скор >= 13]), ("скор 11–12", lambda e: e[(e.скор >= 11) & (e.скор <= 12)]),
              ("все", lambda e: e)]
    w("## Сравнение вариантов — средняя сделка, %\n\n")
    tab = []
    for name in TFS:
        e = ev[ev.вариант == name]
        for gn, f in groups:
            g = f(e).sort_values("t")
            tab.append({"RSI": name, "группа": gn, "сигналов": len(g), "в неделю": len(g) / weeks,
                        **{x: g[x].mean() for x in KEY}})
    w(pd.DataFrame(tab).round(2).to_markdown(index=False) + "\n\n")

    w("## То же — PF (заработано / потеряно)\n\n")
    tab = []
    for name in TFS:
        e = ev[ev.вариант == name]
        for gn, f in groups:
            g = f(e)
            row = {"RSI": name, "группа": gn}
            for x in KEY:
                r = g[x].dropna()
                neg = -r[r < 0].sum()
                row[x] = r[r > 0].sum() / neg if neg > 0 else np.nan
            tab.append(row)
    w(pd.DataFrame(tab).round(2).to_markdown(index=False) + "\n\n")

    w("## До какого уровня доходит цена, пока не выбит стоп 24ч (скор ≥ 11)\n\n")
    lv = []
    for name in TFS:
        g = ev[(ev.вариант == name) & (ev.скор >= 11)]
        mx = g["макс. рост до стопа, %"]
        lv.append({"RSI": name, "сигналов": len(g), **{f"≥ +{x:g}%": (mx >= x).mean() * 100 for x in [6, 15, 27.6]},
                   "стоп 24ч медиана, %": g["стоп 24ч, %"].median()})
    w(pd.DataFrame(lv).round(1).to_markdown(index=False) + "\n\n")

    allx = [f"стоп {s} / тейк {tp}" for s in B.STOPS for tp in B.TAKES] + KEY[-2:]
    w("## Лучшие сочетания вариант × группа × выход (по средней сделке, не меньше 100 сделок)\n\n")
    best = []
    for name in TFS:
        e = ev[ev.вариант == name]
        for gn, f in groups:
            g = f(e).sort_values("t")
            if len(g) < 100:
                continue
            for x in allx:
                s = B.stats(g[x], weeks)
                if s:
                    best.append({"RSI": name, "группа": gn, "выход": x, **s})
    bd = pd.DataFrame(best)
    bd = bd.sort_values("средняя, %", ascending=False) if len(bd) else bd
    w(bd.head(15).round(2).to_markdown(index=False) + "\n\n")
    if len(bd):
        b = bd.iloc[0]
        g = dict(groups)[b.группа](ev[ev.вариант == b.RSI]).copy()
        g["месяц"] = pd.to_datetime(g.t, unit="ms").dt.strftime("%Y-%m")
        w(f"Лучшее ({b.RSI}, {b.группа}, {b.выход}) по месяцам:\n\n")
        per = [{"месяц": m_, **B.stats(x.sort_values("t")[b.выход], 1)} for m_, x in g.groupby("месяц")]
        w(pd.DataFrame(per).drop(columns=["в неделю"], errors="ignore").round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/blogger_rsi_summary.md").read()[:8000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
