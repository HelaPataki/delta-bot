"""
Сколько стоит опоздание со входом в L3 (часовая проверка, тейк +20% / стоп −10%, 7 дней).
Берём те же сделки, что в тесте l3_hourly (2025–2026) и l3_hourly_2023 (2023–2024), и пересчитываем их по 5-минутным
свечам архива Binance: вход не по закрытию часовой свечи, а через 0 / 5 / 10 / 15 / 20 / 30 / 45 / 60 минут
(по открытию 5-минутной свечи в этот момент). Тейк/стоп считаются от цены фактического входа; срок — 7 дней от сигнала.
Если тейк и стоп в одной 5-минутной свече — считаем стоп. Комиссия 0,1%.
Живого бота не трогает.
"""

import datetime as dt
import heapq
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

OUT = os.environ.get("OUT_DIR", "results/l3_delay")
SRC = os.environ.get("SRC", "results/l3_hourly/trades.csv.gz,results/l3_hourly_2023/trades.csv.gz").split(",")
EXIT = os.environ.get("EXIT", "20/10")
DELAYS = [0, 5, 10, 15, 20, 30, 45, 60]
M5, H = 5 * 60 * 1000, 3600 * 1000
TP, SL, HOLD, FEE = 0.20, 0.10, 168 * H, 0.1


def day_k(sym, d):
    url = f"{Hst.ARCHIVE}/data/futures/um/daily/klines/{sym}/5m/{sym}-5m-{d:%Y-%m-%d}.zip"
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    k = Hst.read_zip(r.content)
    return k[["open_time", "open", "high", "low", "close"]].astype(float)


def one(tr):
    sym, t0 = tr["монета"], int(tr["вход"])
    d0 = dt.datetime.fromtimestamp(t0 / 1000, dt.timezone.utc).date()
    parts = [day_k(sym, d0 + dt.timedelta(days=i)) for i in range(9)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    k = pd.concat(parts).drop_duplicates("open_time").sort_values("open_time")
    t, o, h, l, c = (k[x].to_numpy() for x in ("open_time", "open", "high", "low", "close"))
    out = {"монета": sym, "вход": t0, "год": tr["год"], "pnl 1h-тест": tr["pnl %"]}
    end = t0 + HOLD
    for D in DELAYS:
        i = int(np.searchsorted(t, t0 + D * 60 * 1000))
        if i >= len(t) or t[i] != t0 + D * 60 * 1000:
            out[f"pnl {D}"], out[f"часов {D}"] = np.nan, np.nan
            continue
        px = o[i]
        if D:
            out[f"сдвиг цены {D}, %"] = (px / o[int(np.searchsorted(t, t0))] - 1) * 100
        res, ex_t = None, None
        j = i
        while j < len(t) and t[j] < end:
            if l[j] <= px * (1 - SL):
                res, ex_t = -SL * 100, t[j]
                break
            if h[j] >= px * (1 + TP):
                res, ex_t = TP * 100, t[j]
                break
            j += 1
        if res is None:
            jj = min(int(np.searchsorted(t, end)) - 1, len(t) - 1)
            res, ex_t = (c[jj] / px - 1) * 100, t[jj]
        out[f"pnl {D}"] = res - FEE
        out[f"часов {D}"] = (ex_t - t0) / H
    return out


def sim(df, D, slots=3):
    E, peak, dd, heap = 100.0, 100.0, 0.0, []
    for _, r in df.sort_values("вход").iterrows():
        if not np.isfinite(r[f"pnl {D}"]):
            continue
        while heap and heap[0][0] <= r["вход"]:
            _, p, s = heapq.heappop(heap)
            E += s * p / 100
            peak, dd = max(peak, E), min(dd, E / peak - 1)
        if len(heap) < slots:
            heapq.heappush(heap, (r["вход"] + r[f"часов {D}"] * H, r[f"pnl {D}"], E / slots))
    while heap:
        _, p, s = heapq.heappop(heap)
        E += s * p / 100
        peak, dd = max(peak, E), min(dd, E / peak - 1)
    return f"${E:.0f} ({dd * 100:.0f}%)"


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.concat([pd.read_csv(f) for f in SRC if os.path.exists(f)])
    tr = tr[tr["сигнал"].str.startswith("L3") & (tr["тейк/стоп"] == EXIT)].drop_duplicates(["монета", "вход"])
    print("сделок:", len(tr), flush=True)
    with ThreadPoolExecutor(12) as ex:
        rows = [r for r in ex.map(one, tr.to_dict("records")) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/l3_delay_trades.csv.gz", index=False)
    years = sorted(df["год"].unique())
    T = []
    for D in DELAYS:
        g = df[np.isfinite(df[f"pnl {D}"])]
        r = {"опоздание, мин": D, "сделок": len(g), "винрейт": f"{(g[f'pnl {D}'] > 0).mean() * 100:.0f}%",
             "ср. pnl %": round(g[f"pnl {D}"].mean(), 2)}
        if D:
            r["цена ушла от закрытия, ср. %"] = round(g[f"сдвиг цены {D}, %"].mean(), 2)
        for y in years:
            gy = g[g["год"] == y]
            r[f"{y}: ср. pnl / $100"] = f"{gy[f'pnl {D}'].mean():+.2f} / {sim(gy, D)}"
        T.append(r)
    md = [__doc__.strip(), "", f"Выход: {EXIT}. Депозит $100, 3 слота по ⅓, без плеча, отдельно по каждому году.", "",
          f"Сверка: тест на часовых свечах дал ср. pnl {df['pnl 1h-тест'].mean():+.2f}% по этим же сделкам.", "",
          pd.DataFrame(T).to_markdown(index=False)]
    # если цена за время опоздания уже сильно ушла — стоит ли входить
    B = []
    for D in (15, 30, 45):
        col = f"сдвиг цены {D}, %"
        for lo, hi in ((-99, -2), (-2, 0), (0, 2), (2, 5), (5, 99)):
            g = df[(df[col] >= lo) & (df[col] < hi) & np.isfinite(df[f"pnl {D}"])]
            if len(g) >= 10:
                B.append({"опоздание, мин": D, "цена ушла от закрытия": f"{lo}…{hi}%", "сделок": len(g),
                          "винрейт": f"{(g[f'pnl {D}'] > 0).mean() * 100:.0f}%", "ср. pnl %": round(g[f"pnl {D}"].mean(), 2)})
    md += ["", "## Если за время опоздания цена уже ушла", "", pd.DataFrame(B).to_markdown(index=False)]
    open(f"{OUT}/l3_delay_summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
