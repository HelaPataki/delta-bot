"""
Своя карта ликвидаций (как у Coinglass, по OI) + проверка на истории. Живого бота не трогает.

МОДЕЛЬ (каждый час, по каждой монете):
  • OI вырос на ΔOI → открыто ΔOI новых контрактов по средней цене часа (у каждого контракта есть лонг и шорт).
    Плечи (заданы заранее): 5x 10%, 10x 25%, 20x 25%, 25x 15%, 50x 15%, 100x 10%; поддерживающая маржа 0,5%.
    Цена ликвидации лонга = p × (1 − 1/L + 0,005), шорта = p × (1 + 1/L − 0,005).
  • Цена дошла до уровня → эти позиции сгорели (убираем).
  • Модель не может держать больше позиций, чем реальный OI: если больше — всё пропорционально уменьшаем.
  • Старые позиции постепенно закрываются: полураспад 14 дней.
ЧТО ПРОВЕРЯЕМ (на закрытии каждой 4h-свечи):
  H1 «магнит»: перекос ликвидаций сверху/снизу в пределах 10% (и 3%) от цены → куда цена пойдёт дальше.
  H2 «снятие»: свеча сожгла много ликвидаций (доля от OI) → продолжение или разворот.
Исход: что раньше за 72 ч — +3% или −3% (и ±5%); доходность через 24 ч и 72 ч.
Монеты: 80 самых ликвидных USDT-перпетуалов Binance (по обороту прошлого месяца). Период: с 2023-01-01.
"""

import datetime as dt
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2022-12-01")

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_map")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
FROM = pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000
TOPN = int(os.environ.get("TOPN", "80"))
H = PL.H
LEV = {5: 0.10, 10: 0.25, 20: 0.25, 25: 0.15, 50: 0.15, 100: 0.10}
MMR = 0.005
STEP = 0.0025                     # шаг сетки цен ~0,25%
DECAY = 0.5 ** (1 / (14 * 24))    # полураспад 14 дней, за час
FWD = 72


def coin(sym):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < 1000:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    v, qv, t = k.volume.to_numpy(), k.quote_volume.to_numpy(), k.open_time.to_numpy()
    n = len(k)
    days = set()
    d, end = pd.Timestamp(FROM, unit="ms", tz="UTC").date() - dt.timedelta(days=1), dt.datetime.now(dt.timezone.utc).date()
    while d < end:
        days.add(d)
        d += dt.timedelta(days=1)
    met = PL.metrics(sym, days)
    if met is None or len(met) < 1000:
        return None
    OI = PL.AsOf(met, "oi")
    oi = np.array([OI.at(t[i] + H, 2 * H) for i in range(n)])

    lo_g = math.log(np.nanmin(l[l > 0]) * 0.3)
    G = int((math.log(np.nanmax(h) * 3) - lo_g) / STEP) + 2
    longs, shorts = np.zeros(G), np.zeros(G)       # контракты, по уровню цены ликвидации
    gi = lambda p: min(G - 1, max(0, int((math.log(p) - lo_g) / STEP)))
    rows = []
    start = int(np.searchsorted(t, FROM - 30 * 24 * H))   # месяц разгона модели
    prev_oi = np.nan
    burnL = burnS = 0.0
    for i in range(start, n - 1):
        if not (h[i] > 0 and l[i] > 0):
            continue
        longs *= DECAY
        shorts *= DECAY
        # сгорело: шорты с уровнем <= high, лонги с уровнем >= low
        ih, il = gi(h[i]), gi(l[i])
        bs, bl = shorts[:ih + 1].sum(), longs[il:].sum()
        shorts[:ih + 1] = 0
        longs[il:] = 0
        burnL += bl
        burnS += bs
        # новые позиции
        if np.isfinite(oi[i]) and np.isfinite(prev_oi) and oi[i] > prev_oi and v[i] > 0:
            p = qv[i] / v[i]
            add = oi[i] - prev_oi
            for L, wgt in LEV.items():
                longs[gi(p * (1 - 1 / L + MMR))] += add * wgt
                shorts[gi(p * (1 + 1 / L - MMR))] += add * wgt
        # не больше реального OI
        if np.isfinite(oi[i]) and oi[i] > 0:
            for arr in (longs, shorts):
                s = arr.sum()
                if s > oi[i]:
                    arr *= oi[i] / s
        if np.isfinite(oi[i]):
            prev_oi = oi[i]
        # точка наблюдения: закрытие 4h-свечи
        if ((t[i] + H) % (4 * H)) != 0 or t[i] < FROM or i + FWD >= n or not (np.isfinite(oi[i]) and oi[i] > 0):
            continue
        px = c[i]
        ip = gi(px)
        a10, b10 = shorts[ip + 1:gi(px * 1.10) + 1].sum(), longs[gi(px * 0.90):ip].sum()
        a3, b3 = shorts[ip + 1:gi(px * 1.03) + 1].sum(), longs[gi(px * 0.97):ip].sum()
        # крупнейший кластер с каждой стороны в 10%
        up_seg, dn_seg = shorts[ip + 1:gi(px * 1.10) + 1], longs[gi(px * 0.90):ip]
        up_d = (math.exp((ip + 1 + int(np.argmax(up_seg))) * STEP + lo_g) / px - 1) * 100 if up_seg.sum() > 0 else np.nan
        dn_d = (math.exp((gi(px * 0.90) + int(np.argmax(dn_seg))) * STEP + lo_g) / px - 1) * 100 if dn_seg.sum() > 0 else np.nan
        hh, ll = h[i + 1:i + 1 + FWD], l[i + 1:i + 1 + FWD]

        def first(up, dn):
            iu = np.nonzero(hh >= px * (1 + up))[0]
            idn = np.nonzero(ll <= px * (1 - dn))[0]
            iu = iu[0] if len(iu) else 10 ** 6
            idn = idn[0] if len(idn) else 10 ** 6
            return "вверх" if iu < idn else "вниз" if idn < iu else ("нет" if iu == 10 ** 6 else "оба")

        rows.append({"symbol": sym, "t": t[i] + H, "price": px,
                     "imb10": (a10 - b10) / (a10 + b10) if a10 + b10 > 0 else np.nan,
                     "imb3": (a3 - b3) / (a3 + b3) if a3 + b3 > 0 else np.nan,
                     "liq10_oi": (a10 + b10) / oi[i], "up_big_pct": up_d, "dn_big_pct": dn_d,
                     "burnS_oi": burnS / oi[i], "burnL_oi": burnL / oi[i],
                     "ret4": (c[i] / c[i - 4] - 1) * 100 if i >= 4 else np.nan,
                     "first3": first(0.03, 0.03), "first5": first(0.05, 0.05),
                     "ret24": (c[i + 24] / px - 1) * 100, "ret72": (c[i + 72] / px - 1) * 100,
                     "qv4": qv[i - 3:i + 1].sum()})
        burnL = burnS = 0.0
    return pd.DataFrame(rows)


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    today = dt.datetime.now(dt.timezone.utc).date()
    last = dt.date(today.year, today.month, 1) - dt.timedelta(days=1)
    m = dt.date(last.year, last.month, 1)

    def vol(s):
        f = PL.zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m:%Y-%m}.zip")
        if f is None:
            return 0.0
        f = f[pd.to_numeric(f.iloc[:, 0], errors="coerce").notna()]
        return float(f.iloc[:, 7].astype(float).sum())

    with ThreadPoolExecutor(8) as ex:
        vols = dict(zip(syms, ex.map(vol, syms)))
    top = [s for s, _ in sorted(vols.items(), key=lambda x: -x[1])[:TOPN]]
    json.dump(top, open(f"{OUT}/universe.json", "w"))
    print("монеты:", top)


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(syms)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(2) as ex:
        parts = [p for p in ex.map(work, mine) if p is not None and len(p)]
    (pd.concat(parts) if parts else pd.DataFrame()).to_csv(f"{OUT}/parts/obs{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, с данными {len(parts)}")


def table(a, col, q=5, signed=False):
    x = a.dropna(subset=[col])
    try:
        b = pd.qcut(x[col], q, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for iv, g in x.groupby(b, observed=True):
        r = {"диапазон": f"{iv.left:.3g} … {iv.right:.3g}", "наблюдений": len(g),
             "+3% раньше −3%": f"{(g.first3 == 'вверх').mean() * 100:.0f}%",
             "−3% раньше +3%": f"{(g.first3 == 'вниз').mean() * 100:.0f}%",
             "+5% раньше −5%": f"{(g.first5 == 'вверх').mean() * 100:.0f}%",
             "−5% раньше +5%": f"{(g.first5 == 'вниз').mean() * 100:.0f}%",
             "ср. 24ч, %": round(g.ret24.mean(), 2), "ср. 72ч, %": round(g.ret72.mean(), 2)}
        for y, gy in g.groupby(g["год"]):
            up, dn = (gy.first3 == "вверх").mean(), (gy.first3 == "вниз").mean()
            r[f"{y}: вверх−вниз"] = f"{(up - dn) * 100:+.0f}"
        rows.append(r)
    return pd.DataFrame(rows)


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True)
    a["год"] = pd.to_datetime(a.t, unit="ms").dt.year
    a.to_csv(f"{OUT}/observations.csv.gz", index=False)
    # снятие: какая сторона сгорела сильнее за свечу
    a["burn_net"] = a.burnS_oi - a.burnL_oi            # >0 — сожгли шорты (цена шла вверх)
    w = open(f"{OUT}/liq_map_summary.md", "w").write
    w("# Своя карта ликвидаций — проверка на истории\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
    w(f"Монет: {a.symbol.nunique()}, наблюдений (4h-свечей): {len(a)}.\n\n")
    base_up, base_dn = (a.first3 == "вверх").mean() * 100, (a.first3 == "вниз").mean() * 100
    w(f"База (все свечи): +3% раньше −3% в {base_up:.0f}%, −3% раньше +3% в {base_dn:.0f}%; "
      f"ср. 24ч {a.ret24.mean():+.2f}%, 72ч {a.ret72.mean():+.2f}%.\n\n"
      "Колонки «год: вверх−вниз» — разница в процентных пунктах (плюс = чаще шла вверх).\n\n")
    for col, title in (("imb10", "H1: перекос ликвидаций в 10% (+1 = всё сверху, шорты; −1 = всё снизу, лонги)"),
                       ("imb3", "H1: перекос ликвидаций в 3%"),
                       ("liq10_oi", "Сколько ликвидаций рядом (в 10%) от OI"),
                       ("up_big_pct", "Расстояние до крупнейшего кластера сверху, %"),
                       ("dn_big_pct", "Расстояние до крупнейшего кластера снизу, %"),
                       ("burn_net", "H2: снятие за свечу (+ сожгли шорты, − сожгли лонги), доля OI"),
                       ("burnS_oi", "H2: сожгли шортов за свечу, доля OI"),
                       ("burnL_oi", "H2: сожгли лонгов за свечу, доля OI")):
        tb = table(a, col, q=10 if col.startswith(("imb", "burn")) else 5)
        if tb is not None:
            w(f"## {title}\n\n" + tb.to_markdown(index=False) + "\n\n")
    # комбинация: сильный перекос + свеча не против
    w("## Комбинации\n\n")
    combos = {
        "перекос10 ≥ +0,5 (сверху больше)": a.imb10 >= 0.5,
        "перекос10 ≤ −0,5 (снизу больше)": a.imb10 <= -0.5,
        "перекос3 ≥ +0,5": a.imb3 >= 0.5, "перекос3 ≤ −0,5": a.imb3 <= -0.5,
        "сожгли шорты ≥ 2% OI": a.burnS_oi >= 0.02, "сожгли лонги ≥ 2% OI": a.burnL_oi >= 0.02,
        "сожгли лонги ≥ 2% OI и снизу пусто (перекос10 ≥ +0,5)": (a.burnL_oi >= 0.02) & (a.imb10 >= 0.5),
        "сожгли шорты ≥ 2% OI и сверху пусто (перекос10 ≤ −0,5)": (a.burnS_oi >= 0.02) & (a.imb10 <= -0.5),
    }
    rows = []
    for name, m in combos.items():
        g = a[m]
        r = {"условие": name, "наблюдений": len(g),
             "+3% раньше": f"{(g.first3 == 'вверх').mean() * 100:.0f}%", "−3% раньше": f"{(g.first3 == 'вниз').mean() * 100:.0f}%",
             "+5% раньше": f"{(g.first5 == 'вверх').mean() * 100:.0f}%", "−5% раньше": f"{(g.first5 == 'вниз').mean() * 100:.0f}%",
             "ср. 24ч": round(g.ret24.mean(), 2), "ср. 72ч": round(g.ret72.mean(), 2)}
        for y, gy in g.groupby("год"):
            r[str(y)] = f"{((gy.first3 == 'вверх').mean() - (gy.first3 == 'вниз').mean()) * 100:+.0f} ({len(gy)})"
        rows.append(r)
    w(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/liq_map_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
