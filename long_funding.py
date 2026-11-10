"""
Эксперимент (не бот): помогает ли фандинг отбирать сделки нашего лонга.

Лонг: x5, дельта ≤ −17%, POC внизу, условный ордер на POC + 5 тиков, 1:4, до 10 дней.
Сделки: results/long_sim (2025–26) + results/y2021 (2021–24, капа $10M–400M).
Для каждой сделки — средний фандинг (в % за 8 ч) до закрытия сигнальной свечи: за 10 дней, 3 дня, 1 день,
и последняя выплата перед сигналом. Стандартный фандинг Binance — 0.01%.
Сравнение по группам: винрейт, средний результат на сделку (цена + фандинг, комиссия 0.1%).
"""

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import oi_funding as OF

OUT = "results/long_funding"
H = 3600 * 1000
WIN = {"10 дней": 240, "3 дня": 72, "1 день": 24}


def month_rates(sym, months):
    frames = []
    for m in months:
        df = OF.get_zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if df is None:
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "last_funding_rate" not in df.columns:
            df = df.iloc[:, :3]
            df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        if "funding_interval_hours" not in df:
            df["funding_interval_hours"] = 8
        frames.append(pd.DataFrame({"t": df.calc_time.astype(float),
                                    "r8": df.last_funding_rate.astype(float) * 8 / df.funding_interval_hours.astype(float) * 100}))
    return pd.concat(frames).sort_values("t") if frames else None


def per_symbol(item):
    sym, g = item
    need = set()
    for t in g.open_time:
        d0 = pd.Timestamp(t - 11 * 24 * H, unit="ms").date().replace(day=1)
        d1 = pd.Timestamp(t, unit="ms").date().replace(day=1)
        need |= set(Hst.months(d0, d1))
    f = month_rates(sym, sorted(need))
    rows = []
    for r in g.itertuples():
        T = r.open_time + 4 * H
        row = {"idx": r.Index}
        if f is not None:
            for name, hrs in WIN.items():
                x = f[(f.t > T - hrs * H) & (f.t <= T)].r8
                row[name] = x.mean() if len(x) else np.nan
            x = f[f.t <= T].r8
            row["последняя выплата"] = x.iloc[-1] if len(x) else np.nan
        rows.append(row)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    a = pd.read_csv("results/long_sim/oi_sim_trades.csv.gz").query("rr == 4 and hold_h == 240")
    b = pd.read_csv("results/y2021/long_trades_all.csv.gz").query("mcap >= 10e6 and mcap <= 400e6")
    t = pd.concat([a, b], ignore_index=True)
    t["net"] = t.pnl_pct + t.funding_pct - 0.1
    t["год"] = pd.to_datetime(t.open_time, unit="ms").dt.year
    rows = []
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(per_symbol, t.groupby("symbol")):
            rows += r
    f = pd.DataFrame(rows).set_index("idx")
    t = t.join(f)
    fh = open(f"{OUT}/long_funding_summary.md", "w")
    w = fh.write
    w("# Лонг дельта-бота и фандинг перед сигналом\n\n"
      f"Сделок: {len(t)} (2021–26), винрейт {(t.net > 0).mean() * 100:.0f}%, в среднем {t.net.mean():+.2f}% на сделку "
      "(цена + фандинг, комиссия 0.1%). Фандинг — в % за 8 ч, стандарт Binance 0.01%.\n\n"
      "⚠ Сделок мало: разница меньше 2–3 пунктов на сделку может быть случайной.\n\n")
    bins = [-10, -0.005, 0.0095, 0.0105, 0.0116, 10]
    lab = ["отрицательный (< −0.005%)", "ниже стандарта (−0.005…0.0095%)", "стандарт (0.0095…0.0105%)",
           "чуть выше (0.0105…0.0116%)", "выше 0.0116%"]
    for name in list(WIN) + ["последняя выплата"]:
        g = t.dropna(subset=[name]).assign(группа=lambda d: pd.cut(d[name], bins, labels=lab))
        tb = g.groupby("группа", observed=True).agg(
            сделок=("net", "size"), винрейт=("net", lambda x: f"{(x > 0).mean() * 100:.0f}%"),
            в_среднем=("net", "mean"), тейков=("result", lambda x: (x == "тейк").sum()),
            стопов=("result", lambda x: (x == "стоп").sum()))
        tb["в_среднем"] = tb["в_среднем"].round(2)
        w(f"## Фандинг за {name}\n\n" + tb.to_markdown() + "\n\n")
    fh.close()
    out = t[["symbol", "open_time", "result", "net"] + list(WIN) + ["последняя выплата"]].copy()
    out["open_time"] = pd.to_datetime(out.open_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    out.round(5).to_csv(f"{OUT}/long_funding_trades.csv", index=False)
    print(open(f"{OUT}/long_funding_summary.md").read())


if __name__ == "__main__":
    main()
