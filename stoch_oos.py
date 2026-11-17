"""
Двойной стохастик 4ч, ЛОНГ с фильтрами — проверка на 2021–2024 (вне выборки) и симуляция депозита.
Только история. Живых ботов не трогает. Фильтры НЕ подстраиваются — те же, что найдены на 2025–2026.

СИГНАЛ: Stoch(21,9,9) и Stoch(9,3,3) на 4ч-свечах фьючерса: у обоих %K пересекает %D снизу вверх на ОДНОЙ свече,
  %D < 30, %K свечой раньше < 30. Вход — по открытию первой часовой свечи после закрытия сигнальной 4ч-свечи.
ФИЛЬТРЫ:
  • BTC — изменение BTC за 24ч до сигнала не хуже −0,75%;
  • время — сигнальная 4ч-свеча закрывается в 00, 04 или 08 UTC (Киев: 02/06/10 зимой, 03/07/11 летом);
    вариант «без ночи» — только 04 и 08 UTC;
  • толпа — long/short по аккаунтам > 1,3 (архив Binance metrics есть примерно с конца 2021 — до этого фильтр
    не применить, такие сигналы в вариантах с толпой выпадают).
ВЫХОДЫ: тейк/стоп 10/10, 10/5, 15/5, 5/5 по часовым свечам; не дошло за 10 дней — по рынку. Стоп и тейк в одной
  свече — стоп. Комиссия 0,1% на круг. Результат в % цены без плеча.
ДЕПОЗИТ: $100, плечо 2x, 2 или 3 слота (на сделку — текущий депозит / число слотов), сложный процент.
  Пришёл сигнал, а свободных слотов нет — пропускаем. Несколько сигналов одновременно — сначала монеты
  с большим оборотом. Фандинг не учтён.
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL
import stoch as S

OUT = os.environ.get("OUT_DIR", "results/stoch_oos")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, H4 = 3600 * 1000, 4 * 3600 * 1000
D0 = dt.date(2020, 11, 1)
FROM = pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000
HOR = 240
FEE = 0.1
TPSL = [(10, 10), (10, 5), (15, 5), (5, 5)]


def sim(e, hh, ll, cc, tp, sl):
    up, dn = hh >= e * (1 + tp / 100), ll <= e * (1 - sl / 100)
    jt = int(np.argmax(up)) if up.any() else len(up)
    js = int(np.argmax(dn)) if dn.any() else len(dn)
    if jt == len(up) and js == len(dn):
        return (cc[-1] / e - 1) * 100 - FEE, len(cc)
    return ((-sl if js <= jt else tp) - FEE), (min(js, jt) + 1)


def coin(sym, btc):
    k = PL.klines_any("um", sym, D0, dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 1000:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c, qv = (k[x].to_numpy() for x in ("open", "high", "low", "close", "quote_volume"))
    g = pd.DataFrame({"b": t // H4, "o": o, "h": h, "l": l, "c": c}).groupby("b").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("o", "size"))
    g = g[g.n == 4]
    T4 = g.index.to_numpy() * H4
    k1, d1 = S.stoch(g.h.to_numpy(), g.l.to_numpy(), g.c.to_numpy(), 21, 9, 9)
    k2, d2 = S.stoch(g.h.to_numpy(), g.l.to_numpy(), g.c.to_numpy(), 9, 3, 3)
    sig = S.crosses(k1, d1, 1) & S.crosses(k2, d2, 1)
    # чтобы пересечения считались только по подряд идущим свечам
    gap = np.r_[True, np.diff(T4) != H4]
    sig &= ~gap
    idx = {int(x): i for i, x in enumerate(t)}
    rows = []
    for j in np.where(sig)[0]:
        ts = int(T4[j]) + H4
        if ts < FROM:
            continue
        i0 = idx.get(ts)
        if i0 is None or i0 < 24 or i0 + HOR > len(t):
            continue
        bi = np.searchsorted(btc[0], ts, side="right") - 1
        bj = np.searchsorted(btc[0], ts - 24 * H, side="right") - 1
        b24 = (btc[1][bi] / btc[1][bj] - 1) * 100 if bi >= 0 and bj >= 0 and btc[0][bi] > ts - 2 * H else np.nan
        r = {"symbol": sym, "t": ts, "час UTC": (ts // H) % 24, "BTC 24ч, %": b24, "оборот 24ч $": qv[i0 - 24:i0].sum()}
        e = o[i0]
        for tp, sl in TPSL:
            res, hrs = sim(e, h[i0:i0 + HOR], l[i0:i0 + HOR], c[i0:i0 + HOR], tp, sl)
            r[f"{tp}/{sl}"], r[f"{tp}/{sl} часов"] = res, hrs
        rows.append(r)
    if not rows:
        return []
    # толпа — только для сигналов, прошедших BTC и время (экономим загрузки)
    need = [r for r in rows if r["час UTC"] in (0, 4, 8) and np.isfinite(r["BTC 24ч, %"]) and r["BTC 24ч, %"] > -0.75]
    if need:
        days = {pd.Timestamp(r["t"], unit="ms", tz="UTC").date() + dt.timedelta(days=d) for r in need for d in (-1, 0)}
        met = PL.metrics(sym, days)
        glob = PL.AsOf(met, "glob")
        for r in need:
            r["толпа L/S"] = glob.at(r["t"], 3 * H)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(s for s in Hst.list_symbols() if s.endswith("USDT"))
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    b = PL.klines_any("um", "BTCUSDT", D0, dt.datetime.now(dt.timezone.utc).date())
    btc = (b.open_time.to_numpy() + H, b.close.to_numpy())

    def work(s):
        try:
            return coin(s, btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/s{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} сигналов, монет {len(mine)}")


def st(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"сделок": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {"сделок": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(),
            "PF": pos / neg if neg > 0 else np.inf, "серия минусов": best}


def deposit(g, ex, slots, lev=2.0):
    g = g.sort_values(["t", "оборот 24ч $"], ascending=[True, False])
    eq, peak, mdd = 100.0, 100.0, 0.0
    opened, taken, streak, best, yearly = [], 0, 0, 0, {}

    def settle(upto):
        nonlocal eq, peak, mdd
        for b in sorted(x for x in opened if x[0] <= upto):
            opened.remove(b)
            eq += b[1]
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak * 100)

    for t, r, hrs in zip(g.t, g[ex], g[f"{ex} часов"]):
        settle(t)
        if len(opened) >= slots or not np.isfinite(r):
            continue
        pnl = eq / slots * lev * r / 100
        opened.append((t + hrs * H, pnl))
        taken += 1
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)
        yearly.setdefault(pd.Timestamp(t, unit="ms").year, []).append(pnl)
    settle(float("inf"))
    return eq, mdd, taken, best, yearly


FILTERS = {
    "без фильтров (одна свеча)": lambda e: e,
    "BTC": lambda e: e[e["BTC 24ч, %"] > -0.75],
    "BTC + время": lambda e: e[(e["BTC 24ч, %"] > -0.75) & e["час UTC"].isin([0, 4, 8])],
    "BTC + время без ночи": lambda e: e[(e["BTC 24ч, %"] > -0.75) & e["час UTC"].isin([4, 8])],
    "BTC + время + толпа": lambda e: e[(e["BTC 24ч, %"] > -0.75) & e["час UTC"].isin([0, 4, 8]) & (e["толпа L/S"] > 1.3)],
    "BTC + время без ночи + толпа": lambda e: e[(e["BTC 24ч, %"] > -0.75) & e["час UTC"].isin([4, 8]) & (e["толпа L/S"] > 1.3)],
}


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    if "толпа L/S" not in ev:
        ev["толпа L/S"] = np.nan
    ev["год"] = pd.to_datetime(ev.t, unit="ms").dt.year
    ev["время (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/signals.csv.gz", index=False)
    w = open(f"{OUT}/stoch_oos_summary.md", "w").write
    w("# Двойной стохастик 4ч, лонг с фильтрами — 2021–2026\n\n" + __doc__.split("\n", 2)[2] + "\n")
    years = sorted(ev.год.unique())

    w("## 1. По годам: средняя сделка, % цены (в скобках винрейт)\n\n"
      "2021–2024 — проверка вне выборки (фильтры найдены на 2025–2026).\n\n")
    for ex in ("10/10", "15/5", "10/5"):
        rows = []
        for name, f in FILTERS.items():
            g = f(ev)
            row = {"фильтр": name}
            for y in years:
                x = g[g.год == y][ex]
                row[str(y)] = f"{x.mean():+.2f} ({(x > 0).mean() * 100:.0f}%) n{len(x)}" if len(x) else "—"
            oos = g[g.год <= 2024][ex]
            row["2021–24 вместе"] = f"{oos.mean():+.2f} ({(oos > 0).mean() * 100:.0f}%) n{len(oos)}" if len(oos) else "—"
            rows.append(row)
        w(f"### Тейк/стоп {ex}\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")

    w("## 2. Итог вне выборки (2021–2024) и в выборке (2025–2026)\n\n")
    rows = []
    for name, f in FILTERS.items():
        g = f(ev).sort_values("t")
        for per, x in (("2021–24", g[g.год <= 2024]), ("2025–26", g[g.год >= 2025])):
            for ex in ("10/10", "15/5", "10/5", "5/5"):
                rows.append({"фильтр": name, "период": per, "выход": ex, **st(x[ex]),
                             "в неделю": len(x) / (52 * (4 if per == "2021–24" else 1.9))})
    w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")

    w("## 3. Депозит $100, плечо 2x\n\n"
      "Итог — во что превратились $100 за весь период (2021-01 — 2026-11); просадка — худшее падение от пика, %.\n\n")
    rows = []
    for name, f in FILTERS.items():
        g = f(ev)
        if name.endswith("толпа"):
            g = g[g.год >= 2022]
        for ex in ("10/10", "15/5", "10/5"):
            for slots in (2, 3):
                eq, mdd, n, ser, yearly = deposit(g, ex, slots)
                rows.append({"фильтр": name + (" (с 2022)" if name.endswith("толпа") else ""), "выход": ex,
                             "слотов": slots, "сделок": n, "итог $": eq, "просадка, %": mdd, "серия минусов": ser,
                             **{str(y): sum(v) for y, v in sorted(yearly.items())}})
    w("Колонки по годам — прибыль/убыток в $ за год (депозит сложным процентом, поэтому поздние годы крупнее).\n\n")
    w(pd.DataFrame(rows).round(1).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/stoch_oos_summary.md").read()[:9000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
