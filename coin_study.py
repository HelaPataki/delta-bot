"""
Изучение одной монеты (по умолчанию MUSDT). Живых ботов не трогает.

Только скачивает данные из архива Binance и сохраняет их в results/coin_study/<SYMBOL>/,
сам разбор (взлёты/падения, что было до и после) делается потом по этим файлам.

  klines_1h.csv.gz   — 1h-свечи фьючерса: цена, объём, покупки/продажи по рынку (дельта)
  spot_1h.csv.gz     — 1h-свечи спота (если спот есть)
  metrics_5m.csv.gz  — каждые 5 мин: OI, L/S топов (по аккаунтам и по позициям), L/S всех, taker buy/sell
  funding.csv.gz     — фандинг (% за 8 часов)
  info.md            — что скачалось и за какие даты
"""

import datetime as dt
import os

import pandas as pd

os.environ.setdefault("START", "2025-01-01")

import history as Hst
import pump_life as PL

SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", "MUSDT").split(",") if s.strip()]
START = dt.date.fromisoformat(os.environ["START"])


def save(df, path):
    if df is not None and len(df):
        df.to_csv(path, index=False, compression="gzip")


def span(df, col):
    if df is None or not len(df):
        return "нет данных"
    a, b = df[col].min(), df[col].max()
    return f"{PL.kyiv(a)} — {PL.kyiv(b)} (Киев), строк: {len(df)}"


def one(sym):
    out = f"results/coin_study/{sym}"
    os.makedirs(out, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date()
    k = PL.klines_any("um", sym, START, today)
    if k is None or not len(k):
        with open(f"{out}/info.md", "w") as f:
            f.write(f"# {sym}\n\nФьючерс {sym} в архиве Binance не найден.\n")
        print(sym, "not found")
        return
    first = pd.Timestamp(k.open_time.min(), unit="ms").date()
    save(k, f"{out}/klines_1h.csv.gz")
    spot = PL.klines_any("spot", Hst.base_asset(sym) + "USDT", first, today)
    save(spot, f"{out}/spot_1h.csv.gz")
    days, d = [], first
    while d < today:
        days.append(d)
        d += dt.timedelta(days=1)
    m = PL.metrics(sym, days)
    save(m, f"{out}/metrics_5m.csv.gz")
    fu = PL.funding(sym, first.replace(day=1))
    save(fu, f"{out}/funding.csv.gz")
    with open(f"{out}/info.md", "w") as f:
        f.write(f"# {sym}\n\n")
        f.write(f"- 1h фьючерс: {span(k, 'open_time')}\n")
        f.write(f"- 1h спот: {span(spot, 'open_time')}\n")
        f.write(f"- метрики 5м: {span(m, 't')}\n")
        f.write(f"- фандинг: {span(fu, 't')}\n")
    print(open(f"{out}/info.md").read())


if __name__ == "__main__":
    for s in SYMBOLS:
        one(s)
