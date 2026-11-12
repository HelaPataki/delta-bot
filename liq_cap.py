"""
Идея Анастасии: ликвидации «притягивают» цену, только если их много относительно капитализации.
(пример: капа $50M, шортовых ликвидаций сверху ~$5M = 10% капы — серьёзно; капа $1B и $7M — цена туда не идёт)
Живого бота и другие скрипты не трогает.

Карта — та же модель, что в liq_map.py (по OI архива Binance), суммы в долларах.
Модель завышает суммы примерно в 10–13 раз против реальных ликвидаций Binance (liq_check), поэтому
кроме «сырых» процентов от капы показываем пересчёт ÷13 — примерно в масштабе реальных ликвидаций.
Капа — оценка: текущая капа CoinGecko × цена тогда / цена сейчас.

Монеты: все USDT-перпетуалы Binance с текущей капой $10M–$2B. Период: с 2025-01-01, наблюдение на закрытии 4h-свечи.
Для каждой стороны:
  ВВЕРХ: шортовые ликвидации над ценой в пределах 20%; «кластер» — самая толстая полоса шириной 1% (2–20% от цены).
         Цель: дойдёт ли цена до середины кластера раньше, чем уйдёт на столько же вниз (за 7 дней).
  ВНИЗ:  то же для лонговых ликвидаций под ценой.
Если «притяжение» есть — при большой доле кластера от капы цена должна чаще доходить до кластера,
чем в среднем (когда кластер маленький).
"""

import datetime as dt
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-12-01")

import numpy as np
import pandas as pd

import history as Hst
import liq_map as LM
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_cap")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = PL.H
FROM = pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000
WARM = FROM - 30 * 24 * H
FWD = 168
SCALE = 13.0


def coin(sym, cap):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < 500:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    v, qv, t = k.volume.to_numpy(), k.quote_volume.to_numpy(), k.open_time.to_numpy()
    n, c_now = len(k), k.close.iloc[-1]
    days, d = set(), pd.Timestamp(max(WARM, t[0]), unit="ms", tz="UTC").date()
    while d < dt.datetime.now(dt.timezone.utc).date():
        days.add(d)
        d += dt.timedelta(days=1)
    met = PL.metrics(sym, days)
    if met is None or len(met) < 500:
        return None
    OI, OIU = PL.AsOf(met, "oi"), PL.AsOf(met, "oi_usd")
    lo_g = math.log(np.nanmin(l[l > 0]) * 0.3)
    G = int((math.log(np.nanmax(h) * 3) - lo_g) / LM.STEP) + 2
    longs, shorts = np.zeros(G), np.zeros(G)
    lvl = np.exp(lo_g + (np.arange(G) + 0.5) * LM.STEP)
    gi = lambda p: min(G - 1, max(0, int((math.log(p) - lo_g) / LM.STEP)))
    W1 = max(1, int(round(0.01 / LM.STEP)))           # полоса ~1%
    prev, rows = np.nan, []
    for i in range(int(np.searchsorted(t, WARM)), n):
        if not (h[i] > 0 and l[i] > 0):
            continue
        oi = OI.at(t[i] + H, 2 * H)
        longs *= LM.DECAY
        shorts *= LM.DECAY
        shorts[:gi(h[i]) + 1] = 0
        longs[gi(l[i]):] = 0
        if np.isfinite(oi) and np.isfinite(prev) and oi > prev and v[i] > 0:
            p = qv[i] / v[i]
            for L, wgt in LM.LEV.items():
                longs[gi(p * (1 - 1 / L + LM.MMR))] += (oi - prev) * wgt
                shorts[gi(p * (1 + 1 / L - LM.MMR))] += (oi - prev) * wgt
        if np.isfinite(oi) and oi > 0:
            for arr in (longs, shorts):
                s = arr.sum()
                if s > oi:
                    arr *= oi / s
            prev = oi
        if ((t[i] + H) % (4 * H)) != 0 or t[i] < FROM or i + FWD >= n:
            continue
        px = c[i]
        mcap = cap * px / c_now
        ip = gi(px)
        hh, ll = h[i + 1:i + 1 + FWD], l[i + 1:i + 1 + FWD]
        row = {"symbol": sym, "t": int((t[i] + H) // 1000), "капа $M": mcap / 1e6,
               "OI$/капа %": OIU.at(t[i] + H, 2 * H) / mcap * 100, "оборот 24ч / капа %": qv[max(0, i - 23):i + 1].sum() / mcap * 100}
        for side, arr, sign in (("вверх", shorts, 1), ("вниз", longs, -1)):
            usd = arr * lvl
            a0, a1 = (gi(px * 1.02), gi(px * 1.20)) if sign > 0 else (gi(px * 0.80), gi(px * 0.98))
            seg = usd[a0:a1 + 1]
            tot20 = usd[ip + 1:gi(px * 1.20) + 1].sum() if sign > 0 else usd[gi(px * 0.80):ip].sum()
            if len(seg) < W1 or seg.sum() <= 0:
                row.update({f"{side}: ликв20 / капа %": tot20 / mcap * 100, f"{side}: кластер / капа %": 0.0,
                            f"{side}: до кластера %": np.nan, f"{side}: исход": "нет кластера"})
                continue
            band = np.convolve(seg, np.ones(W1), "valid")
            j = int(np.argmax(band))
            mid = lvl[a0 + j + W1 // 2]
            dist = mid / px - 1
            # дойдёт до кластера раньше, чем уйдёт на столько же в другую сторону
            if sign > 0:
                it, io = np.nonzero(hh >= mid)[0], np.nonzero(ll <= px * (1 - dist))[0]
            else:
                it, io = np.nonzero(ll <= mid)[0], np.nonzero(hh >= px * (1 - dist))[0]
            it = it[0] if len(it) else 10 ** 6
            io = io[0] if len(io) else 10 ** 6
            res = "к кластеру" if it < io else "от кластера" if io < it else ("никуда" if it == 10 ** 6 else "обе")
            row.update({f"{side}: ликв20 / капа %": tot20 / mcap * 100, f"{side}: кластер / капа %": band[j] / mcap * 100,
                        f"{side}: до кластера %": abs(dist) * 100, f"{side}: исход": res,
                        f"{side}: часов до кластера": it + 1 if it < 10 ** 6 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps.get(Hst.base_asset(s)) for s in syms}
    uni = {s: v for s, v in uni.items() if v and 10e6 <= v <= 2e9}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, uni[s])
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(2) as ex:
        parts = [p for p in ex.map(work, mine) if p is not None and len(p)]
    (pd.concat(parts) if parts else pd.DataFrame()).to_csv(f"{OUT}/parts/obs{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(parts)} из {len(mine)}")


def side_table(a, side, by):
    g0 = a[a[f"{side}: исход"].isin(["к кластеру", "от кластера", "никуда", "обе"])]
    rows = []
    for lab, g in g0.groupby(by, observed=True):
        if len(g) < 100:
            continue
        r = {"группа": str(lab), "наблюдений": len(g),
             "до кластера, % (медиана)": round(g[f"{side}: до кластера %"].median(), 1),
             "дошла до кластера первой": f"{(g[f'{side}: исход'] == 'к кластеру').mean() * 100:.0f}%",
             "ушла в другую сторону": f"{(g[f'{side}: исход'] == 'от кластера').mean() * 100:.0f}%",
             "никуда за 7 дней": f"{(g[f'{side}: исход'] == 'никуда').mean() * 100:.0f}%"}
        k, o_ = (g[f"{side}: исход"] == "к кластеру").mean(), (g[f"{side}: исход"] == "от кластера").mean()
        r["перевес к кластеру, п.п."] = f"{(k - o_) * 100:+.0f}"
        for y, gy in g.groupby("год"):
            if len(gy) >= 50:
                r[str(y)] = f"{((gy[f'{side}: исход'] == 'к кластеру').mean() - (gy[f'{side}: исход'] == 'от кластера').mean()) * 100:+.0f}"
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
    a["год"] = pd.to_datetime(a.t, unit="s").dt.year
    a.to_csv(f"{OUT}/observations.csv.gz", index=False)
    w = open(f"{OUT}/liq_cap_summary.md", "w").write
    w("# Ликвидации относительно капитализации: притягивают ли цену\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
    w(f"Монет: {a.symbol.nunique()}, наблюдений: {len(a)}.\n\n"
      "«Перевес к кластеру» = (дошла до кластера первой) − (ушла на то же расстояние в другую сторону), в процентных пунктах. "
      "Если притяжения нет — около 0 (или как в группе с маленьким кластером).\n\n")
    capb = pd.cut(a["капа $M"], [0, 50, 100, 300, 1000, 1e9], labels=["<$50M", "$50–100M", "$100–300M", "$300M–1B", ">$1B"])
    edges = [0, 0.5, 1, 2, 5, 10, 20, 50, 1e9]
    labs = ["<0,5%", "0,5–1%", "1–2%", "2–5%", "5–10%", "10–20%", "20–50%", "50%+"]
    for side, title in (("вверх", "ВВЕРХ: шорты над ценой"), ("вниз", "ВНИЗ: лонги под ценой")):
        w(f"## {title}\n\n")
        for col, name in ((f"{side}: кластер / капа %", "самая толстая полоса 1%"), (f"{side}: ликв20 / капа %", "всё в пределах 20%")):
            b = pd.cut(a[col], edges, labels=labs, right=False)
            bs = pd.cut(a[col] / SCALE, edges, labels=labs, right=False)
            w(f"### {name}, доля от капы (сырая модель)\n\n" + side_table(a.assign(_b=b), side, "_b").to_markdown(index=False) + "\n\n")
            w(f"### {name}, доля от капы (÷{SCALE:g}, в масштабе реальных ликвидаций)\n\n"
              + side_table(a.assign(_b=bs), side, "_b").to_markdown(index=False) + "\n\n")
        w("### По размеру капы × толщина кластера (сырая модель, полоса 1%)\n\n")
        b = pd.cut(a[f"{side}: кластер / капа %"], [0, 1, 5, 20, 1e9], labels=["<1%", "1–5%", "5–20%", "20%+"], right=False)
        w(side_table(a.assign(_g=capb.astype(str) + " | " + b.astype(str)), side, "_g").to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/liq_cap_summary.md").read()[:5000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
