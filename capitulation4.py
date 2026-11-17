"""
Капитуляция, часть 4: подбор выхода СО СТОПОМ. Живых ботов не трогает. TradFi и SNDK исключены.

Сигнал (как в части 3): час, когда ≥25% (или ≥30%) из 100 самых торгуемых монет упали за час на 3%+,
  и это не первый слив: за 7 дней до было ≥2 (или ≥3) суток со сливами. После сигнала 24 ч пауза.
Корзина: «5 сильнее всех упавших» (из 50 самых торгуемых) и для сравнения «весь рынок».
Вход — открытие следующего часа. У каждой монеты свой стоп/тейк, результат события = среднее по корзине.
Выходы (все — со стопом, кроме первой строки для сравнения):
  • без стопа, 72 ч (как раньше);
  • только стоп 10 / 15 / 20 / 25 / 30%, держать 48 / 72 / 120 ч;
  • стоп + тейк: стоп 10 / 15 / 20, тейк 15 / 20 / 30, не дошло за 72 ч — по рынку;
  • стоп + безубыток: стоп 15 / 20, после +10% стоп переносится на вход, 72 ч;
  • трейлинг: стоп 15 / 20; после +10% стоп подтягивается на 5 / 8 % ниже максимума; держать до 120 ч.
Стоп и тейк (или подтяжка) в одной часовой свече — считаем стоп. Комиссия 0,1% на круг. Без плеча
  (с плечом 2x ROI в бирже = результат × 2: стоп 15% цены = −30% ROI).
Подбор — 2021–2023, проверка — 2024–2026. Депозит $100 на 3 частях (как в части 3), тоже с плечом 2x.
"""

import os

os.environ.setdefault("START", "2020-12-01")

import numpy as np
import pandas as pd

import capitulation as C
import capitulation3 as C3

OUT = os.environ.get("OUT_DIR", "results/capitulation4")
C.OUT = OUT
C3.OUT = OUT
H = C.H
FIT_END = 2023
MAXH = 120


def exits_def():
    ex = [("без стопа 72ч", dict(stop=None, take=None, hold=72))]
    for s in (10, 15, 20, 25, 30):
        for h in (48, 72, 120):
            ex.append((f"стоп {s}, {h}ч", dict(stop=s, take=None, hold=h)))
    for s in (10, 15, 20):
        for tp in (15, 20, 30):
            ex.append((f"стоп {s} / тейк {tp}", dict(stop=s, take=tp, hold=72)))
    for s in (15, 20):
        ex.append((f"стоп {s} + БУ после +10", dict(stop=s, take=None, hold=72, be=10)))
    for s in (15, 20):
        for tr in (5, 8):
            ex.append((f"стоп {s} + трейлинг {tr} после +10", dict(stop=s, take=None, hold=120, act=10, trail=tr)))
    return ex


EXITS = exits_def()
NAMES = [n for n, _ in EXITS]


def run_exit(e, hh, ll, cc, p):
    hold = p["hold"]
    h, l = hh[:hold], ll[:hold]
    stop0 = e * (1 - p["stop"] / 100) if p["stop"] else -np.inf
    level = np.full(hold, stop0)
    peakprev = np.maximum.accumulate(np.r_[e, h[:-1]])          # максимум до этой свечи
    if p.get("be"):
        level = np.where(peakprev >= e * (1 + p["be"] / 100), np.maximum(level, e), level)
    if p.get("act"):
        level = np.where(peakprev >= e * (1 + p["act"] / 100),
                         np.maximum(level, peakprev * (1 - p["trail"] / 100)), level)
    hs = l <= level
    ht = h >= e * (1 + p["take"] / 100) if p["take"] else np.zeros(hold, bool)
    ks = int(np.argmax(hs)) if hs.any() else hold
    kt = int(np.argmax(ht)) if ht.any() else hold
    if ks == hold and kt == hold:
        return (cc[hold - 1] / e - 1) * 100
    if ks <= kt:
        return (min(level[ks], hh[ks]) / e - 1) * 100
    return p["take"]


def trade(M, j, i0):
    o, h, l, c = M["o"][:, j], M["h"][:, j], M["l"][:, j], M["c"][:, j]
    if i0 + MAXH > len(o):
        return None
    e = o[i0]
    if not np.isfinite(e) or e <= 0:
        return None
    hh, ll, cc = h[i0:i0 + MAXH], l[i0:i0 + MAXH], c[i0:i0 + MAXH]
    if not (np.isfinite(hh).all() and np.isfinite(ll).all() and np.isfinite(cc).all()):
        return None
    return {n: run_exit(e, hh, ll, cc, p) - C.FEE for n, p in EXITS}


def combine():
    syms, t, M = C.load_matrix()
    n = len(t)
    sets = C.liquid_sets(t, M)
    r1 = (M["c"] / M["o"] - 1).astype("float32")
    sh = np.full(n, np.nan)
    for i in range(n):
        S = sets[i]
        if len(S) < 50:
            continue
        x = r1[i, S]
        x = x[np.isfinite(x)]
        if len(x) >= 50:
            sh[i] = (x <= -0.03).mean()
    start = int(np.searchsorted(t, C.T0))
    year = pd.to_datetime(t, unit="ms").year.to_numpy()
    rows = []
    for thr in (0.25, 0.30):
        flag = np.nan_to_num(sh) >= thr
        last = -10 ** 9
        for i in np.where(flag)[0]:
            if i < start or i >= n - MAXH - 1 or i - last < 24:
                continue
            last = i
            ser = C3.series_days(flag, i, 7)
            if ser < 2:
                continue
            S = sets[i]
            top50 = [j for j in S[:50] if np.isfinite(r1[i, j])]
            fall = sorted(top50, key=lambda j: r1[i, j])[:5]
            for bname, js in (("5 сильнее всех упавших", fall), ("весь рынок", list(S))):
                res = [r for r in (trade(M, j, i + 1) for j in js) if r]
                if not res:
                    continue
                rows.append({"порог": f"{int(thr * 100)}%", "t": int(t[i]), "год": int(year[i]),
                             "время (Киев)": C.kyiv(t[i]), "серия 7д": ser, "корзина": bname,
                             "монеты": ",".join(syms[j].replace("USDT", "") for j in js[:5]),
                             **{x: float(np.mean([r[x] for r in res])) for x in NAMES},
                             **{f"худшая монета | {x}": float(np.min([r[x] for r in res])) for x in NAMES[:1]}})
    ev = pd.DataFrame(rows)
    if not len(ev):
        open(f"{OUT}/capitulation4_summary.md", "w").write("Сигналов нет\n")
        return
    ev.to_csv(f"{OUT}/events.csv.gz", index=False)
    summary(ev)


def sim(g, x, parts=3, lev=1):
    g = g.assign(**{x: g[x] * lev}).assign(**{x: lambda d: d[x].clip(lower=-100)})
    return C3.slots_sim(g, parts, x)


def summary(ev):
    w = open(f"{OUT}/capitulation4_summary.md", "w").write
    w("# Капитуляция, часть 4: выход со стопом\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w("Средняя — средний результат события в % цены без плеча. «Худшее» — худшее событие (средняя по корзине). "
      "Депозит — $100, 3 части, одна сделка = 1/3 капитала.\n\n")
    for thr in ("25%", "30%"):
        for k in (2, 3):
            for b in ("5 сильнее всех упавших", "весь рынок"):
                g = ev[(ev.порог == thr) & (ev.корзина == b) & (ev["серия 7д"] >= k)].sort_values("t")
                if len(g) < 10:
                    continue
                rows = []
                for x in NAMES:
                    fa, fb = g[g.год <= FIT_END][x], g[g.год > FIT_END][x]
                    yrs = g.groupby("год")[x].mean()
                    d1, d2 = sim(g, x, 3, 1), sim(g, x, 3, 2)
                    rows.append({"выход": x, "2021–23: средняя": fa.mean(), "2024–26: средняя": fb.mean(),
                                 "в плюс, %": (g[x] > 0).mean() * 100, "худшее": g[x].min(),
                                 "годов в плюсе": f"{(yrs > 0).sum()}/{len(yrs)}",
                                 "$100 → x1": d1["итог $"], "просадка x1, %": d1["макс. просадка, %"],
                                 "$100 → x2": d2["итог $"], "просадка x2, %": d2["макс. просадка, %"]})
                t_ = pd.DataFrame(rows)
                w(f"## Порог {thr}, серия ≥{k} дней, корзина «{b}» — {len(g)} событий\n\n")
                w(t_.round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/capitulation4_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        C3.prep()
    elif mode == "combine":
        combine()
    else:
        C.run_shard()
