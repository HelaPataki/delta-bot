"""
Шорт: влияет ли объём на результат (идея Насти после графика BLUR). Старые скрипты не меняются.

Сигналы и сделки — из прогона «Short exits» (вход через 8 мин, тейки 1:1,8 и 1:1,5), 2025–2026 и 2021–2024.
По полным 4-часовым свечам из архива Binance (оборот в $) для каждого сигнала:
  объём свечи x  — оборот сигнальной свечи к медиане 12 прошлых свечей (как у лонга);
  объём 24ч x    — оборот за сутки до сигнала включительно к обороту за предыдущие сутки;
  рост заранее x — оборот за сутки ДО сигнальной свечи к предыдущим суткам («объём начал расти заранее»).
Итог — винрейт и % на сделку по группам, отдельно для каждого периода (признак устойчив, только если работает в обоих).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

OUT = "results/short_volume"
H4 = 4 * 3600 * 1000
Hst.START = "2020-12-01"


def feats(sym, opens):
    k = Hst.load_klines(sym)
    if k is None or not len(k):
        return []
    k = k.sort_values("open_time").reset_index(drop=True)
    idx = {int(t): i for i, t in enumerate(k.open_time)}
    qv = k.quote_volume.astype(float).to_numpy()
    out = []
    for t in opens:
        i = idx.get(int(t))
        if i is None or i < 12:
            continue
        med = np.median(qv[i - 12:i])
        out.append({"symbol": sym, "open_time": int(t),
                    "объём свечи x": qv[i] / med if med > 0 else np.nan,
                    "объём 24ч x": qv[i - 5:i + 1].sum() / qv[i - 11:i - 5].sum() if qv[i - 11:i - 5].sum() > 0 else np.nan,
                    "рост заранее x": qv[i - 6:i].sum() / qv[i - 12:i - 6].sum() if qv[i - 12:i - 6].sum() > 0 else np.nan})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv("results/short_exits/short_exits_trades.csv.gz")
    t = t[t.variant.isin(["тейк 1:1,8 (как сейчас)", "тейк 1:1,5"])].copy()
    t["net"] = t.pnl_pct + t.funding_pct - 0.1
    sig = t[["symbol", "open_time"]].drop_duplicates()
    groups = sig.groupby("symbol").open_time.apply(list)
    with ThreadPoolExecutor(8) as ex:
        f = pd.DataFrame([x for r in ex.map(lambda kv: feats(*kv), groups.items()) for x in r])
    t = t.merge(f, on=["symbol", "open_time"], how="left")
    t.to_csv(f"{OUT}/short_volume_trades.csv.gz", index=False)

    bins = {"объём свечи x": ([0, 2, 5, 10, 1e9], ["до 2x", "2–5x", "5–10x", "больше 10x"]),
            "объём 24ч x": ([0, 1, 2, 4, 1e9], ["меньше 1x", "1–2x", "2–4x", "больше 4x"]),
            "рост заранее x": ([0, 1, 2, 4, 1e9], ["меньше 1x", "1–2x", "2–4x", "больше 4x"])}
    out = ["# Шорт: влияет ли объём на результат\n",
           "Вход через 8 мин, стоп 4,3%, выход 24 ч, комиссия и фандинг учтены. "
           f"Объём найден для {t['объём свечи x'].notna().mean() * 100:.0f}% сделок.\n"]
    for col, (b, lab) in bins.items():
        out.append(f"## {col}\n")
        rows = []
        for v in ("тейк 1:1,8 (как сейчас)", "тейк 1:1,5"):
            for period in ("2025–26", "2021–24"):
                d = t[(t.variant == v) & (t.period == period)]
                g = d.groupby(pd.cut(d[col], b, labels=lab), observed=False)
                for name, x in g:
                    rows.append({"тейк": v.split(" (")[0], "период": period, "группа": name, "сделок": len(x),
                                 "винрейт": f"{(x.result == 'тейк').mean() * 100:.0f}%" if len(x) else "—",
                                 "% на сделку": f"{x.net.mean():+.2f}%" if len(x) else "—"})
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    open(f"{OUT}/short_volume_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
