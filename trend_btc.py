"""
Шорт по тренду + фильтр по рынку. Сделки — из results/trend_short/trend_short_signals.csv (пробой N дней, ретест 7 дней).
Состояние рынка берётся на момент закрытия свечи пробоя (без заглядывания в будущее):
  BTC относительно своей средней за 20 дней (120 свечей 4h): ниже / выше;
  BTC за 7 дней: <= −5% / −5…0% / > 0%;
  «ширина»: сколько монет пробили N-дневный минимум в ту же 4h-свечу: 1–2 / 3–9 / 10+.
Для каждого фильтра: сделок, винрейт, средний R, сколько месяцев в плюсе.
"""

import os

import numpy as np
import pandas as pd

import history as Hst
from poc import kyiv_to_utc

OUT_DIR = os.environ.get("OUT_DIR", "results/trend_btc")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = pd.read_csv("results/trend_short/trend_short_signals.csv")
    a["t"] = a.time.map(lambda s: kyiv_to_utc(s).timestamp() * 1000)
    Hst.START = "2024-11-01"
    b = Hst.load_klines("BTCUSDT")
    b["close_t"] = b.open_time + 4 * 3600 * 1000
    b["ma20"] = b.close.rolling(120).mean()
    b["chg7"] = (b.close / b.close.shift(42) - 1) * 100
    bt = b.set_index("open_time")
    a["btc_below_ma"] = a.t.map(bt.close).lt(a.t.map(bt.ma20))
    a["btc_chg7"] = a.t.map(bt.chg7)
    a["breadth"] = a.groupby(["N", "t"]).symbol.transform("count")

    filters = [
        ("без фильтра", lambda d: d.N > 0),
        ("BTC ниже средней 20 дн.", lambda d: d.btc_below_ma),
        ("BTC выше средней 20 дн.", lambda d: ~d.btc_below_ma),
        ("BTC за 7 дн. ≤ −5%", lambda d: d.btc_chg7 <= -5),
        ("BTC за 7 дн. −5…0%", lambda d: (d.btc_chg7 > -5) & (d.btc_chg7 <= 0)),
        ("BTC за 7 дн. > 0%", lambda d: d.btc_chg7 > 0),
        ("ширина 1–2 монеты", lambda d: d.breadth <= 2),
        ("ширина 3–9 монет", lambda d: (d.breadth >= 3) & (d.breadth <= 9)),
        ("ширина 10+ монет", lambda d: d.breadth >= 10),
        ("BTC ниже средней + ширина 3+", lambda d: d.btc_below_ma & (d.breadth >= 3)),
    ]
    out = []
    for N in (20, 30):
        for sp in (4.3, 6.0):
            res, Rc = f"res_7_{sp}", f"R_7_{sp}"
            g0 = a[(a.N == N) & a[res].isin(["тейк", "стоп", "по времени"])].copy()
            g0["mo"] = g0.time.str[:7]
            months_total = g0.mo.nunique()
            for name, f in filters:
                g = g0[f(g0)]
                if len(g) == 0:
                    continue
                tk, st = (g[res] == "тейк").sum(), (g[res] == "стоп").sum()
                mo = g.groupby("mo")[Rc].mean()
                out.append({"пробой": f"{N} дн.", "стоп / тейк": f"+{sp:g}% / −{sp * 1.8:.2f}%", "фильтр": name,
                            "сделок": len(g), "в месяц": f"{len(g) / months_total:.0f}",
                            "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%", "в среднем R": f"{g[Rc].mean():+.2f}",
                            "месяцев в плюсе": f"{(mo > 0).sum()} из {len(mo)}",
                            "худший месяц R": f"{mo.min():+.2f}"})
    with open(f"{OUT_DIR}/trend_btc_summary.md", "w") as f:
        f.write("# Шорт по тренду + фильтр по рынку\n\nПробой минимума N дней, шорт на ретесте (ждём 7 дней), тейк = 1,8 × стоп. "
                "Фильтры — на момент закрытия свечи пробоя. Безубыточный винрейт 35,7%, безубыточно = 0R.\n\n")
        f.write(pd.DataFrame(out).to_markdown(index=False) + "\n")
    print(open(f"{OUT_DIR}/trend_btc_summary.md").read())


if __name__ == "__main__":
    main()
