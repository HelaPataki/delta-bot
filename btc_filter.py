"""
Зависит ли результат от рынка? Раскладываем уже посчитанные сделки по движению BTC
на момент закрытия сигнальной свечи (без заглядывания в будущее).

  BTC 4h  — изменение BTC за сигнальную 4h-свечу
  BTC 24h — изменение BTC за 24 часа до закрытия сигнальной свечи

Шорт: results/short_tests/short_tests_trades.csv (сценарии S0, C1, C3).
Лонг: results/variants/variants_trades.csv (лонг, дельта ≤ −17%, ожидание 1 свеча).
Результат в R (тейк +1,8R, стоп −1R).
"""

import os

import numpy as np
import pandas as pd

import history as Hst
from poc import kyiv_to_utc

OUT_DIR = os.environ.get("OUT_DIR", "results/btc_filter")
B24 = [-100, -3, -1, 1, 3, 100]
L24 = ["≤ −3%", "−3…−1%", "−1…+1%", "+1…+3%", "≥ +3%"]
B4 = [-100, -1.5, -0.5, 0.5, 1.5, 100]
L4 = ["≤ −1,5%", "−1,5…−0,5%", "−0,5…+0,5%", "+0,5…+1,5%", "≥ +1,5%"]


def btc_table():
    k = Hst.load_klines("BTCUSDT")
    k["btc_4h"] = (k.close / k.open - 1) * 100
    k["btc_24h"] = (k.close / k.close.shift(6) - 1) * 100
    return k.set_index("open_time")[["btc_4h", "btc_24h"]]


def add_btc(df, btc):
    t = df.candle_open_kyiv.map(lambda s: kyiv_to_utc(s).timestamp() * 1000)
    df = df.copy()
    df["btc_4h"] = btc.btc_4h.reindex(t.values).values
    df["btc_24h"] = btc.btc_24h.reindex(t.values).values
    return df


def table(df, col, bins, labels):
    rows = []
    g = df.copy()
    g["b"] = pd.cut(g[col], bins, labels=labels)
    for b, x in g.groupby("b", observed=True):
        tk, st = (x.result == "тейк").sum(), (x.result == "стоп").sum()
        rows.append({col.replace("btc_", "BTC "): b, "сделок": len(x),
                     "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%", "в среднем R": f"{x.R.mean():+.2f}",
                     "сумма R": f"{x.R.sum():+.1f}"})
    tk, st = (df.result == "тейк").sum(), (df.result == "стоп").sum()
    rows.append({col.replace("btc_", "BTC "): "все", "сделок": len(df),
                 "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%", "в среднем R": f"{df.R.mean():+.2f}",
                 "сумма R": f"{df.R.sum():+.1f}"})
    return pd.DataFrame(rows).to_markdown(index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    btc = btc_table()
    closed = ["тейк", "стоп", "по времени"]

    s = pd.read_csv("results/short_tests/short_tests_trades.csv")
    s = add_btc(s[s.result.isin(closed)], btc)
    v = pd.read_csv("results/variants/variants_trades.csv")
    lg = v[(v.side == "лонг") & (v.delta_pct <= -17) & v.fill_wait_h.notna() & (v.fill_wait_h < 4)
           & v.result.isin(closed)].copy()
    lg["R"] = lg.pnl_pct / 4.3
    lg = add_btc(lg, btc)

    groups = [
        ("Шорт S0 (вход POC−5т, стоп +4,3%, ожидание 24 ч), дельта любая",
         s[(s["сценарий"] == "S0") & (s["ожидание"] == "24 часа")]),
        ("Шорт S0, дельта > 0", s[(s["сценарий"] == "S0") & (s["ожидание"] == "24 часа") & (s.delta_pct > 0)]),
        ("Шорт C1 (подтверждение, по рынку), дельта любая", s[s["сценарий"] == "C1"]),
        ("Шорт C3 (подтверждение, ретест, 24 ч), дельта любая",
         s[(s["сценарий"] == "C3") & (s["ожидание"] == "24 часа")]),
        ("Лонг (x5, дельта ≤ −17%, POC внизу, закрытие выше, ожидание 1 свеча)", lg),
    ]
    with open(f"{OUT_DIR}/btc_filter_summary.md", "w") as f:
        f.write("# Результат в зависимости от BTC в момент сигнала\n\n")
        f.write("BTC 4h — изменение BTC за сигнальную свечу; BTC 24h — за 24 часа до её закрытия. "
                "Безубыточно = 0R. Комиссии не учтены.\n\n")
        for name, g in groups:
            f.write(f"## {name}\n\n")
            f.write(table(g, "btc_24h", B24, L24) + "\n\n")
            f.write(table(g, "btc_4h", B4, L4) + "\n\n")
    pd.concat([s.assign(side="шорт"), lg.assign(side="лонг")]).to_csv(f"{OUT_DIR}/btc_filter_trades.csv", index=False)
    print(open(f"{OUT_DIR}/btc_filter_summary.md").read())


if __name__ == "__main__":
    main()
