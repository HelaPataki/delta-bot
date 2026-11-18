"""
Разблокировки: входить ли раньше, чем за 5 дней. Логика cycle_grid.py, только разблокировки (инсайдеры ≥3%).
Вход в 07:00 по Киеву за 14 / 10 / 7 / 5 / 3 дня; выход в 07:00 в день разблокировки / через 1 / 3 дня;
стоп +15%; тейк −8 / −10 / −12 / −15%. Отдельно — «запас по дате»: насколько результат зависит от того,
что выход привязан к дате, которая может сдвинуться (сравниваем выход в день / +1 / +3).
Только история. Живых ботов не трогает.
"""
import os

import pandas as pd

import cycle_grid as G

G.OUT = os.environ.get("OUT_DIR", "results/unlock_early")
G.TIMES = ["07 Киев"]
G.U_BEFORE, G.U_AFTER, G.U_STOPS, G.U_TAKES = [14, 10, 7, 5, 3], [0, 1, 3], [15], [8, 10, 12, 15]
G.L_HOLDS = []                       # листинги не считаем


def prep():
    os.makedirs(G.OUT, exist_ok=True)
    ev = pd.read_csv("results/unlocks_study/events.csv.gz")
    ev = ev[(ev["вид"] == "инсайдеры") & (ev["размер, %"] >= 3)][["symbol", "дата", "размер, %"]]
    ev.to_csv(f"{G.OUT}/ev.csv", index=False)
    pd.Series(sorted(ev.symbol.unique()), name="symbol").to_csv(f"{G.OUT}/syms.csv", index=False)
    print("событий:", len(ev))


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{G.OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{G.OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True)
    ev["период"] = (pd.to_datetime(ev.t, unit="ms").dt.year <= 2024).map({True: "21–24", False: "25–26"})
    rows = []
    for key, g in ev.groupby(["вход", "выход", "тейк"]):
        row = {"вход": key[0], "выход": key[1], "тейк": key[2]}
        for per in ("21–24", "25–26"):
            x = g[g.период == per]
            row[f"{per} сделок"] = len(x)
            row[f"{per} винрейт, %"] = (x.r > 0).mean() * 100
            row[f"{per} средняя, %"] = x.r.mean()
            row[f"{per} дней"] = x["дней"].mean()
            row[f"{per} % в день"] = x.r.sum() / x["дней"].sum()
        rows.append(row)
    rd = pd.DataFrame(rows)
    rd["% в день: хуже из двух"] = rd[["21–24 % в день", "25–26 % в день"]].min(axis=1)
    order = {f"за {d}д": i for i, d in enumerate([14, 10, 7, 5, 3])}
    rd = rd.sort_values(["вход", "выход", "тейк"], key=lambda s: s.map(order) if s.name == "вход" else s)
    w = open(f"{G.OUT}/unlock_early_summary.md", "w").write
    w("# Разблокировки: насколько рано входить\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(rd.round(2).to_markdown(index=False) + "\n\n")
    w("## Лучшие 10 по худшему из двух периодов\n\n" +
      rd.sort_values("% в день: хуже из двух", ascending=False).head(10).round(2).to_markdown(index=False) + "\n")
    print(open(f"{G.OUT}/unlock_early_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        G.run_shard()
