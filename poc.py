"""
POC сигнальной свечи по минутным данным Binance.

POC = цена минутной свечи с наибольшим объёмом внутри 4h-свечи
(так же, как линия в индикаторе LuxAlgo Volume Delta Candles).
Цена минутки берётся как (high + low) / 2.

Для каждого сигнала из results/history_signals.csv считаем:
  poc_pos     — где POC внутри 4h-свечи: 0 = на минимуме, 100 = на максимуме
  poc_zone    — низ (0–33), середина (33–67), верх (67–100)
  close_vs_poc_pct — закрытие свечи относительно POC, %
  close_below_poc  — закрылась ли свеча ниже POC
и сравниваем, что было дальше.
"""

import io
import os
import sys
import time
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

IN_FILE = os.environ.get("IN_FILE", "results/history_signals.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results")
ARCHIVE = "https://data.binance.vision"
KYIV = ZoneInfo("Europe/Kyiv")
UTC = dt.timezone.utc

session = requests.Session()
session.headers["User-Agent"] = "delta-bot-poc"


def get(url, tries=4):
    for i in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return None


def kyiv_to_utc(s):
    """Время открытия свечи по Киеву -> UTC. Свечи 4h открываются в 00/04/08/... UTC,
    это снимает неоднозначность в ночь перевода часов."""
    naive = dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
    for fold in (0, 1):
        u = naive.replace(tzinfo=KYIV, fold=fold).astimezone(UTC)
        if u.hour % 4 == 0 and u.minute == 0:
            return u
    return naive.replace(tzinfo=KYIV).astimezone(UTC)


_cache = {}


def minutes(sym, day):
    key = (sym, day)
    if key in _cache:
        return _cache[key]
    url = f"{ARCHIVE}/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day:%Y-%m-%d}.zip"
    r = get(url)
    df = None
    if r is not None and r.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None)
        if isinstance(df.iloc[0, 0], str):
            df = df.iloc[1:]
        df = df.iloc[:, :6].astype(float)
        df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    _cache[key] = df
    return df


def poc_for(row):
    start = kyiv_to_utc(row["candle_open_kyiv"])
    m = minutes(row["symbol"], start.date())
    empty = {"poc": np.nan, "poc_pos": np.nan, "close_vs_poc_pct": np.nan, "minutes_found": 0}
    if m is None:
        return empty
    t0 = start.timestamp() * 1000
    w = m[(m.open_time >= t0) & (m.open_time < t0 + 4 * 3600 * 1000)]
    if len(w) == 0:
        return empty
    i = w.volume.idxmax()
    poc = (w.at[i, "high"] + w.at[i, "low"]) / 2
    hi, lo = w.high.max(), w.low.min()
    pos = (poc - lo) / (hi - lo) * 100 if hi > lo else 50.0
    return {"poc": poc, "poc_pos": pos,
            "close_vs_poc_pct": (row["close"] / poc - 1) * 100, "minutes_found": len(w)}


def stats(g):
    g = g[g.complete]
    if len(g) == 0:
        return pd.Series(dtype=object)
    return pd.Series({
        "сигналов": len(g),
        "сначала рост": f"{(g.first_move == 'рост').mean() * 100:.0f}%",
        "сначала падение": f"{(g.first_move == 'падение').mean() * 100:.0f}%",
        "+10% за 10д": f"{g.up10_day.notna().mean() * 100:.0f}%",
        "−10% за 10д": f"{g.dn10_day.notna().mean() * 100:.0f}%",
        "+20% за 10д": f"{g.up20_day.notna().mean() * 100:.0f}%",
        "−20% за 10д": f"{g.dn20_day.notna().mean() * 100:.0f}%",
        "медиана макс. роста": f"{g.max_up_pct.median():.1f}%",
        "медиана макс. падения": f"{g.max_dn_pct.median():.1f}%",
    })


def main():
    sig = pd.read_csv(IN_FILE)
    print(f"сигналов: {len(sig)}")
    rows = sig.to_dict("records")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(poc_for, rows))
    p = pd.DataFrame(res)
    for c in p.columns:
        sig[c] = p[c].values
    sig["poc_zone"] = pd.cut(sig.poc_pos, [-0.01, 33.3, 66.7, 100.01], labels=["низ", "середина", "верх"])
    sig["close_below_poc"] = np.where(sig.poc.isna(), "", np.where(sig.close < sig.poc, "да", "нет"))
    sig["poc"] = sig.poc.round(8)
    sig["poc_pos"] = sig.poc_pos.round(1)
    sig["close_vs_poc_pct"] = sig.close_vs_poc_pct.round(2)
    sig.to_csv(f"{OUT_DIR}/poc_signals.csv", index=False)

    ok = sig[sig.poc.notna()]
    print(f"POC посчитан: {len(ok)} из {len(sig)}")
    with open(f"{OUT_DIR}/poc_summary.md", "w") as f:
        f.write("# POC сигнальной свечи (по минуткам)\n\n")
        f.write(f"POC посчитан для {len(ok)} из {len(sig)} сигналов.\n")
        f.write("Зона POC: низ / середина / верх — в какой трети 4h-свечи (от минимума до максимума) стоит POC.\n\n")
        for mult in (3, 5, 10):
            for sign in ("+", "−"):
                g = ok[(ok.vol_ratio >= mult) & (ok.delta_sign == sign)]
                if len(g) == 0:
                    continue
                f.write(f"## Объём ≥ x{mult}, дельта {sign}17% и сильнее\n\n")
                f.write(pd.DataFrame({"все": stats(g)}).to_markdown() + "\n\n")
                f.write("**По зоне POC:**\n\n")
                f.write(pd.DataFrame({z: stats(g[g.poc_zone == z]) for z in ["низ", "середина", "верх"]}).to_markdown() + "\n\n")
                f.write("**Закрытие ниже / выше POC:**\n\n")
                f.write(pd.DataFrame({"закрылась ниже POC": stats(g[g.close_below_poc == "да"]),
                                      "закрылась выше POC": stats(g[g.close_below_poc == "нет"])}).to_markdown() + "\n\n")
                f.write("**Все сочетания: зона POC + закрытие выше/ниже POC:**\n\n")
                combos = {}
                for z in ["верх", "середина", "низ"]:
                    for cb, txt in (("да", "закрытие ниже"), ("нет", "закрытие выше")):
                        part = g[(g.poc_zone == z) & (g.close_below_poc == cb)]
                        if len(part[part.complete]):
                            combos[f"POC {z} + {txt}"] = stats(part)
                f.write(pd.DataFrame(combos).to_markdown() + "\n\n")
    print(open(f"{OUT_DIR}/poc_summary.md").read())


if __name__ == "__main__":
    main()
