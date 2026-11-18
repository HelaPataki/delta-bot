"""
Сделки выбранных вариантов календарных шортов (со временем входа и выхода) — для общей симуляции.
Логика та же, что в cycle_grid.py, только без перебора. Только история. Живых ботов не трогает.
  Разблокировки: вход за 5 дней в 00:00 UTC или 07:00 Киев, стоп +15%, тейк −8% / −10%, выход по времени через 1 день после.
  Листинги: вход на 30-й день в час листинга или в 07:00 Киев, держать до 14 дней, стоп +30%,
            тейк −10% (без безубытка) / −15% (с безубытком после −8%).
"""
import os

import pandas as pd

import cycle_grid as G

G.OUT = os.environ.get("OUT_DIR", "results/cycle_trades")
G.TIMES = ["00 UTC", "07 Киев"]
G.U_BEFORE, G.U_AFTER, G.U_STOPS, G.U_TAKES = [5], [1], [15], [8, 10]
G.L_STOPS, G.L_TAKES, G.L_HOLDS = [30], [10, 15], [14]


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{G.OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{G.OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True)
    ev["выход_t"] = ev.t + (ev["дней"] * 24 * G.H).round().astype("int64")
    ev.to_csv(f"{G.OUT}/trades.csv.gz", index=False)
    s = ev.groupby(["стратегия", "время", "стоп", "тейк", "безубыток"]).agg(
        сделок=("r", "size"), средняя=("r", "mean"), дней=("дней", "mean")).round(2)
    open(f"{G.OUT}/cycle_trades_summary.md", "w").write("# Сделки для симуляции\n\n" + s.to_markdown() + "\n")
    print(s)


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        G.prep()
    elif mode == "combine":
        combine()
    else:
        G.run_shard()
