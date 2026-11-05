"""
Насколько цена поднимается перед падением на −10%? По всплескам объёма x3 / x5 / x10.

Группы:
  1. все всплески (любая дельта)
  2. всплески с дельтой ≥ +17%
  3. всплески с дельтой ≥ +17%, POC в верхних 30%, закрытие ниже POC (наша шортовая теория; POC из results/poc_signals.csv)

Для каждой группы: в скольких % случаев за 10 дней было −10%, и среди них — насколько цена сначала
поднималась от закрытия сигнальной свечи (распределение и накопительно: «рост был не больше X%»).
По 4h high/low; если рост и −10% в одной 4h-свече, рост засчитывается (осторожно).
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import short_search as S

OUT_DIR = os.environ.get("OUT_DIR", "results/rise_check")
BINS = [-0.001, 2, 4, 7, 10, 1000]
LABELS = ["до 2%", "2–4%", "4–7%", "7–10%", "больше 10%"]
CUM = [2, 3, 4.3, 5, 7, 8, 10]


def row(g):
    g = g[g.complete]
    if len(g) == 0:
        return None
    d = g[g.dn10]
    r = {"сигналов": len(g), "было −10% за 10д": f"{g.dn10.mean() * 100:.0f}%"}
    if len(d):
        dist = pd.cut(d.rise_before_dn10, BINS, labels=LABELS).value_counts(normalize=True) * 100
        for lab in LABELS:
            r[f"рост {lab}"] = f"{dist.get(lab, 0):.0f}%"
        r["медиана роста"] = f"{d.rise_before_dn10.median():.1f}%"
    return r


def cum_row(g):
    g = g[g.complete & g.dn10]
    if len(g) == 0:
        return None
    return {f"≤ {c:g}%": f"{(g.rise_before_dn10 <= c).mean() * 100:.0f}%" for c in CUM}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    with_cap = [s for s in syms if Hst.base_asset(s) in caps]

    def work(sym):
        try:
            df = Hst.load_klines(sym)
            if df is None or len(df) < 200:
                return None
            o = S.coin(sym, df, caps[Hst.base_asset(sym)])
            return o[o.vol_ratio >= 3]
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(8) as ex:
        a = pd.concat([p for p in ex.map(work, with_cap) if p is not None and len(p)], ignore_index=True)
    a["candle_open_kyiv"] = (pd.to_datetime(a.open_time, unit="ms", utc=True)
                             .dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M"))
    p = pd.read_csv("results/poc_signals.csv")[["symbol", "candle_open_kyiv", "poc_pos", "close_below_poc"]]
    a = a.merge(p, on=["symbol", "candle_open_kyiv"], how="left")
    theory = (a.delta_pct >= 17) & (a.poc_pos >= 70) & (a.close_below_poc == "да")

    groups = [("все всплески", a.delta_pct.notna()),
              ("дельта ≥ +17%", a.delta_pct >= 17),
              ("дельта ≥ +17%, POC вверху, закрытие ниже POC", theory)]
    rows, crows = [], []
    for name, m in groups:
        for x in (3, 5, 10):
            g = a[m & (a.vol_ratio >= x)]
            r = row(g)
            if r:
                rows.append({"группа": name, "объём": f"≥ x{x}", **r})
            c = cum_row(g)
            if c:
                crows.append({"группа": name, "объём": f"≥ x{x}", **c})
    with open(f"{OUT_DIR}/rise_check_summary.md", "w") as f:
        f.write("# Рост перед падением на −10% (по всплескам объёма)\n\n")
        f.write("От цены закрытия сигнальной свечи, 10 дней, по 4h high/low. "
                "Распределение — среди случаев, где −10% было.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
        f.write("## Накопительно: в скольких % падений на −10% рост перед ним был не больше X%\n\n")
        f.write("(то есть стоп на +X% от закрытия пережил бы столько падений)\n\n")
        f.write(pd.DataFrame(crows).to_markdown(index=False) + "\n")
    a[["symbol", "candle_open_kyiv", "delta_pct", "vol_ratio", "change_pct", "poc_pos", "close_below_poc",
       "complete", "dn10", "dn10_hours", "rise_before_dn10", "max_up", "max_dn"]].round(3).to_csv(
        f"{OUT_DIR}/rise_check_signals.csv", index=False)
    print(open(f"{OUT_DIR}/rise_check_summary.md").read())


if __name__ == "__main__":
    main()
