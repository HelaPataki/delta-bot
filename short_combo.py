"""
Шорт: структура свечи «POC в верхних 30% + закрытие ниже POC» с любой дельтой, объём >= x5,
два входа рядом: POC − 5 тиков и максимум свечи + 1%. По минуткам.

Сигналы и POC — из results/variants/variants_trades.csv (шортовые строки).
Стоп +4,3%, тейк −7,74% от входа (1:1,8). Условный ордер ждёт 1 свечу 4h / 24 часа.
Если в одной минутке задеты и стоп, и тейк — считается стоп. Держим до 10 дней. Комиссии не учтены.
"""

import os

os.environ["SIDE"] = "short"
os.environ.setdefault("EXT_PCT", "1")

from concurrent.futures import ThreadPoolExecutor   # noqa: E402

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

import backtest as B                                # noqa: E402

OUT_DIR = os.environ.get("OUT_DIR", "results/short_combo")
BINS = [5, 7, 10, 20, 10 ** 6]
LABELS = ["x5–7", "x7–10", "x10–20", "x20+"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    v = pd.read_csv("results/variants/variants_trades.csv")
    s = v[v.side == "шорт"].drop_duplicates(["symbol", "candle_open_kyiv"]).reset_index(drop=True)
    print(f"шортовых сигналов: {len(s)}")
    parts = []
    for mode, name in (("poc", "POC − 5 тиков"), ("extreme", f"максимум + {B.EXT_PCT:g}%")):
        B.ENTRY_MODE = mode
        with ThreadPoolExecutor(16) as ex:
            res = pd.DataFrame(list(ex.map(B.run_one, s.to_dict("records"))))
        part = pd.concat([s[["symbol", "candle_open_kyiv", "delta_pct", "vol_ratio", "poc", "close"]], res], axis=1)
        part["вход"] = name
        parts.append(part)
        print(f"{name}: готово")
    t = pd.concat(parts, ignore_index=True)
    t["R"] = t.pnl_pct / B.STOP_PCT
    t.round(4).to_csv(f"{OUT_DIR}/short_combo_trades.csv", index=False)

    rows = []
    for dname, dm in (("любая", t.delta_pct.notna()), ("> 0", t.delta_pct > 0), ("≥ +17%", t.delta_pct >= 17)):
        for entry in t["вход"].unique():
            for wname, wmin in (("1 свеча 4h", 240), ("24 часа", 1440)):
                g = t[dm & (t["вход"] == entry) & t.fill_wait_min.notna() & (t.fill_wait_min < wmin)
                      & t.result.isin(["тейк", "стоп", "по времени"])].copy()
                g["объём"] = pd.cut(g.vol_ratio, BINS, right=False, labels=LABELS)
                for b, gg in list(g.groupby("объём", observed=True)) + [("все x5+", g)]:
                    tk, st = (gg.result == "тейк").sum(), (gg.result == "стоп").sum()
                    rows.append({"дельта": dname, "вход": entry, "ожидание": wname, "объём": b, "сделок": len(gg),
                                 "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
                                 "в среднем R": f"{gg.R.mean():+.2f}" if len(gg) else "-",
                                 "в среднем %": f"{gg.pnl_pct.mean():+.2f}%" if len(gg) else "-"})
    signals = s.vol_ratio.ge(5).sum()
    with open(f"{OUT_DIR}/short_combo_summary.md", "w") as f:
        f.write("# Шорт: POC вверху + закрытие ниже POC, любая дельта, два входа\n\n")
        f.write(f"Сигналов (объём ≥ x5): {signals}. Стоп +{B.STOP_PCT:g}%, тейк −{B.STOP_PCT * B.RR:.2f}%. "
                f"Безубыточный винрейт {1 / (1 + B.RR) * 100:.1f}%, безубыточно = 0R. Комиссии не учтены.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(open(f"{OUT_DIR}/short_combo_summary.md").read())


if __name__ == "__main__":
    main()
