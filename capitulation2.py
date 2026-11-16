"""
Капитуляция, часть 2: «первый слив — ловушка, слив посреди серии — точка для лонга». Живых ботов не трогает.

Отличия от capitulation.py:
  • акции, фонды, металлы, сырьё (TradFi, список из tradfi.py) убраны совсем — и из «рынка», и из корзин;
  • у каждого события считаем, в какой фазе падения рынок:
      «дней со сливом за 7 дней до» — сколько из 7 предыдущих суток содержали хотя бы один час, когда ≥30%
          ликвидных монет упали за час на 3%+ (текущий час не считается);
      «рынок за 7 дней, %» — медианная ликвидная монета за последние 7 дней;
      «рынок за 30 дней, %» — то же за 30 дней; BTC за 24 ч и за 7 дней.
  • выходы: через 24 ч / 72 ч / 7 дней; 72 ч с аварийным стопом −20% / −30%; тейк/стоп 10/10.
Сливы-триггеры (после срабатывания 24 ч пауза): A30 (≥30% монет −3%+ за час), A50, B5 (BTC −5% за 24 ч, час красный),
  D50 (≥50% монет −10%+ за 24 ч).
Честная проверка: пороги смотрим на 2021–2023, проверяем на 2024–2026; плюс разбивка по каждому году
  и депозит ($100, одна сделка за раз, весь депозит в корзину, 1x и 2x).
"""

import json
import os

os.environ.setdefault("START", "2020-12-01")

import numpy as np
import pandas as pd

import capitulation as C
import history as Hst
from tradfi import TRADFI

OUT = os.environ.get("OUT_DIR", "results/capitulation2")
C.OUT = OUT
H = C.H
EXITS = ["10/10", "24ч", "72ч", "168ч", "72ч стоп20", "72ч стоп30"]
BASKETS = ["5 сильнее всех упавших", "топ-5 крупных", "весь рынок", "BTC", "ETH"]
FIT_END = 2023


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = [s for s in Hst.list_symbols() if not any(x in s for x in C.SKIP) and s not in TRADFI]
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def trade(M, j, i0):
    o, h, l, c = M["o"][:, j], M["h"][:, j], M["l"][:, j], M["c"][:, j]
    if i0 + 168 > len(o):
        return None
    e = o[i0]
    if not np.isfinite(e) or e <= 0:
        return None
    hh, ll, cc = h[i0:i0 + 72], l[i0:i0 + 72], c[i0:i0 + 168]
    if not (np.isfinite(hh).all() and np.isfinite(ll).all() and np.isfinite(cc[[23, 71, 167]]).all()):
        return None
    res = {}
    hs, ht = ll <= e * 0.9, hh >= e * 1.1
    js = int(np.argmax(hs)) if hs.any() else 72
    jt = int(np.argmax(ht)) if ht.any() else 72
    res["10/10"] = ((cc[71] / e - 1) * 100 if js == 72 and jt == 72 else (-10 if js <= jt else 10)) - C.FEE
    for k in (24, 72, 168):
        res[f"{k}ч"] = (cc[k - 1] / e - 1) * 100 - C.FEE
    for st in (20, 30):
        hit = ll <= e * (1 - st / 100)
        res[f"72ч стоп{st}"] = (-st if hit.any() else (cc[71] / e - 1) * 100) - C.FEE
    return res


def combine():
    syms, t, M = C.load_matrix()
    print("матрица:", M["c"].shape)
    sets = C.liquid_sets(t, M)
    o, c = M["o"], M["c"]
    r1 = c / o - 1
    n = len(t)
    share3 = np.full(n, np.nan)
    med7 = np.full(n, np.nan)
    med30 = np.full(n, np.nan)
    for i in range(n):
        S = sets[i]
        if len(S) < 50:
            continue
        x = r1[i, S]
        x = x[np.isfinite(x)]
        if len(x) >= 50:
            share3[i] = (x <= -0.03).mean()
        if i >= 720:
            y = c[i, S] / o[i - 167, S] - 1
            y = y[np.isfinite(y)]
            med7[i] = np.median(y) * 100 if len(y) >= 50 else np.nan
            y = c[i, S] / o[i - 719, S] - 1
            y = y[np.isfinite(y)]
            med30[i] = np.median(y) * 100 if len(y) >= 50 else np.nan
    flush = np.nan_to_num(share3) >= 0.30
    defs = C.find_events(t, M, syms, sets)
    bi, ei = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    rows = []
    for k, evs in defs.items():
        if k == "M3":
            continue
        for i, s3, med, s10 in evs:
            fh = np.where(flush[max(0, i - 168):i])[0]
            fdays = len(set((i - (fh + max(0, i - 168)) - 1) // 24)) if len(fh) else 0
            feat = {"слив": k, "t": int(t[i]), "время (Киев)": C.kyiv(t[i]), "год": pd.Timestamp(int(t[i]), unit="ms").year,
                    "доля упавших за час": s3, "медиана монеты за час, %": med * 100, "доля −10% за сутки": s10,
                    "дней со сливом за 7д до": fdays, "рынок за 7д, %": med7[i], "рынок за 30д, %": med30[i],
                    "BTC за 24ч, %": (c[i, bi] / o[i - 23, bi] - 1) * 100,
                    "BTC за 7д, %": (c[i, bi] / o[i - 167, bi] - 1) * 100}
            for bname, js in C.baskets(i, syms, M, sets[i]).items():
                if bname not in BASKETS:
                    continue
                res = [r for r in (trade(M, j, i + 1) for j in js) if r]
                if not res:
                    continue
                rows.append({**feat, "корзина": bname, "монеты": ",".join(syms[j].replace("USDT", "") for j in js[:5]),
                             **{x: float(np.mean([r[x] for r in res])) for x in EXITS}})
    ev = pd.DataFrame(rows)
    ev.to_csv(f"{OUT}/events.csv.gz", index=False)
    summary(ev, (t[-1] - C.T0) / (365.25 * 24 * H))


def deposit(g, x, lev):
    busy, r, yrs = -1, [], []
    for tt, v, y in zip(g.t, g[x], g.год):
        if tt < busy:
            continue
        busy = tt + 169 * H if x == "168ч" else tt + 73 * H
        r.append(v)
        yrs.append(y)
    if not r:
        return {}
    r = np.array(r) * lev / 100
    r = np.maximum(r, -1)
    dep = 100 * np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[100, dep])[1:]
    by = pd.Series(r, index=yrs).groupby(level=0).apply(lambda s: (np.prod(1 + s) - 1) * 100)
    return {"сделок": len(r), "итог $": dep[-1], "макс. просадка, %": ((peak - dep) / peak).max() * 100,
            **{f"{y}, %": v for y, v in by.items()}}


def summary(ev, years):
    w = open(f"{OUT}/capitulation2_summary.md", "w").write
    w("# Капитуляция, часть 2: фаза падения\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w("Результаты — в % цены без плеча (2x — умножить на 2), средний результат события.\n\n")
    for k in ("A30", "D50", "B5", "A50"):
        e = ev[ev.слив == k]
        if not len(e):
            continue
        w(f"## {k}\n\n### По «дням со сливом за 7 дней до» (выход 72ч)\n\n")
        rows = []
        for b in BASKETS:
            g = e[e.корзина == b]
            for lo in (0, 1, 2, 3, 4, 5):
                x = g[g["дней со сливом за 7д до"] >= lo]
                fa, fb = x[x.год <= FIT_END], x[x.год > FIT_END]
                yrs = x.groupby("год")["72ч"].mean()
                rows.append({"корзина": b, "дней со сливом ≥": lo, "событий": len(x),
                             "2021–23: средняя": fa["72ч"].mean(), "2021–23: в плюс, %": (fa["72ч"] > 0).mean() * 100,
                             "2024–26: средняя": fb["72ч"].mean(), "2024–26: в плюс, %": (fb["72ч"] > 0).mean() * 100,
                             "годов в плюсе": f"{(yrs > 0).sum()}/{len(yrs)}", "худшее": x["72ч"].min()})
        w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")
        w("### По «рынок за 7 дней» (выход 72ч)\n\n")
        rows = []
        for b in BASKETS[:3]:
            g = e[e.корзина == b]
            for hi in (0, -5, -10, -15, -20, -25):
                x = g[g["рынок за 7д, %"] <= hi]
                fa, fb = x[x.год <= FIT_END], x[x.год > FIT_END]
                yrs = x.groupby("год")["72ч"].mean()
                rows.append({"корзина": b, "рынок за 7д ≤": hi, "событий": len(x),
                             "2021–23: средняя": fa["72ч"].mean(), "2024–26: средняя": fb["72ч"].mean(),
                             "2024–26: в плюс, %": (fb["72ч"] > 0).mean() * 100,
                             "годов в плюсе": f"{(yrs > 0).sum()}/{len(yrs)}", "худшее": x["72ч"].min()})
        w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")
        w("### Депозит $100: одна сделка за раз, весь депозит в корзину\n\n")
        rows = []
        for b in BASKETS[:3]:
            for lo in (0, 2, 3, 4):
                g = e[(e.корзина == b) & (e["дней со сливом за 7д до"] >= lo)].sort_values("t")
                for x in ("72ч", "72ч стоп20", "168ч"):
                    for lev in (1, 2):
                        d = deposit(g, x, lev)
                        if d:
                            rows.append({"корзина": b, "дней со сливом ≥": lo, "выход": x, "плечо": lev, **d})
        w(pd.DataFrame(rows).round(0).to_markdown(index=False) + "\n\n")
    e = ev[(ev.слив == "A30") & (ev.корзина == "5 сильнее всех упавших") & (ev.год == 2026)]
    cols = ["время (Киев)", "дней со сливом за 7д до", "рынок за 7д, %", "монеты", "24ч", "72ч", "72ч стоп20"]
    w("## 2026, A30, корзина «5 сильнее всех упавших» — для глаз\n\n" + e[cols].round(2).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/capitulation2_summary.md").read()[:5000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        C.run_shard()
