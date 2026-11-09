"""
Эксперимент (не бот): «коробка → выход 4h» — что наступает раньше, тейк или стоп, при разных тейках.

Сигналы — из results/box4h_feat/box4h_feat_all.csv.gz (коробка ≤ 25%, с 2025 года), с признаком фандинга.
Вход: через 15 минут после закрытия сигнальной 4h-свечи (как в Box 4h). Путь — 15-минутные свечи, 30 дней.
Тейки 10 / 15 / 20 / 25 / 30%; стопы 5 / 8 / 10 / 12 / 15 / 20 / 25%. Если тейк и стоп в одной 15-минутке — считаем стоп.
Ни то ни другое за 30 дней — выход по цене через 30 дней. Комиссия 0.1%.
Группы: все сигналы; фандинг за коробку выше стандартного (> 0.0116% за 8 ч).
Просадка до тейка: насколько цена уходила ниже входа, прежде чем дойти до тейка (без стопа).
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-12-01")

import numpy as np
import pandas as pd

import history as Hst

OUT = "results/box4h_tp"
SRC = "results/box4h_feat/box4h_feat_all.csv.gz"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H4, M15 = 4 * 3600 * 1000, 15 * 60 * 1000
DAYS = 30
TPS = [10, 15, 20, 25, 30]
SLS = [5, 8, 10, 12, 15, 20, 25]
FUND = "фандинг средний за коробку, %"
FUND_MIN = 0.0116


def prep():
    os.makedirs(OUT, exist_ok=True)
    a = pd.read_csv(SRC)
    json.dump({s: g[["t", FUND, "год"]].values.tolist() for s, g in a.groupby("монета")}, open(f"{OUT}/todo.json", "w"))
    print("сигналов:", len(a))


def coin(sym, items):
    m = Hst.load_klines(sym, "15m")
    if m is None:
        return []
    mt, mh, ml, mc = (m[x].to_numpy() for x in ("open_time", "high", "low", "close"))
    rows = []
    for t, fund, year in items:
        T = t + H4
        j = np.searchsorted(mt, T)
        if j >= len(mt) or mt[j] != T:
            continue
        px = mc[j]
        end = np.searchsorted(mt, T + M15 + DAYS * 86400 * 1000)
        if end >= len(mt):
            continue
        ph, pl, pc = mh[j + 1:end], ml[j + 1:end], mc[j + 1:end]
        r = {"монета": sym, "t": t, "фандинг": fund, "год": int(year), "close30": (pc[-1] / px - 1) * 100}
        for tp in TPS:
            hit = ph >= px * (1 + tp / 100)
            i = int(np.argmax(hit)) if hit.any() else -1
            r[f"tp{tp}"] = i
            r[f"mae{tp}"] = (pl[:i + 1].min() / px - 1) * 100 if i >= 0 else np.nan
        for sl in SLS:
            hit = pl <= px * (1 - sl / 100)
            r[f"sl{sl}"] = int(np.argmax(hit)) if hit.any() else -1
        rows.append(r)
    return rows


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    todo = json.load(open(f"{OUT}/todo.json"))
    mine = [s for i, s in enumerate(sorted(todo)) if i % NSHARDS == SHARD]

    def work(sym):
        try:
            return coin(sym, todo[sym])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: сигналов {len(rows)}")


def outcome(g, tp, sl):
    ti, si = g[f"tp{tp}"].to_numpy(), g[f"sl{sl}"].to_numpy()
    tpf = (ti >= 0) & ((si < 0) | (ti < si))
    slf = (si >= 0) & ((ti < 0) | (si <= ti))
    nei = ~tpf & ~slf
    res = np.where(tpf, tp, np.where(slf, -sl, g.close30.to_numpy())) - 0.1
    return tpf, slf, nei, res


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True)
    groups = {f"фандинг за коробку > {FUND_MIN}%": a[a["фандинг"] > FUND_MIN], "все сигналы": a}
    w = open(f"{OUT}/box4h_tp_summary.md", "w").write
    w("# Коробка → выход 4h: что раньше — тейк или стоп\n\n"
      "Вход через 15 минут после закрытия сигнальной свечи. Следим 30 дней по 15-минуткам. "
      "Если тейк и стоп в одной 15-минутке — считаем стоп. «Ни то ни другое» — за 30 дней не было ни тейка, ни стопа, "
      "выход по цене через 30 дней. Средний результат — % цены после комиссии 0.1% (при x2 вдвое больше).\n\n")
    for gname, g in groups.items():
        months = 22
        w(f"## {gname}: {len(g)} сигналов (~{len(g) / months:.0f} в месяц)\n\n")
        rows = []
        for tp in TPS:
            reach = (g[f"tp{tp}"] >= 0).mean()
            mae = g[f"mae{tp}"].dropna()
            rows.append({"тейк": f"{tp}%", "дошли до тейка (без стопа)": f"{reach * 100:.0f}%",
                         "просадка до тейка: медиана": f"{mae.median():.1f}%",
                         "у 75% не глубже": f"{mae.quantile(.25):.1f}%",
                         "у 90% не глубже": f"{mae.quantile(.10):.1f}%"})
        w("### Сколько доходят до тейка и какая просадка по пути\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
        rows = []
        for tp in TPS:
            for sl in SLS:
                tpf, slf, nei, res = outcome(g, tp, sl)
                y25, y26 = (g["год"] == 2025).to_numpy(), (g["год"] == 2026).to_numpy()
                rows.append({"тейк": f"{tp}%", "стоп": f"{sl}%",
                             "раньше тейк": f"{tpf.mean() * 100:.0f}%", "раньше стоп": f"{slf.mean() * 100:.0f}%",
                             "ни то ни другое": f"{nei.mean() * 100:.0f}%",
                             "в среднем, %": round(res.mean(), 2),
                             "2025, %": round(res[y25].mean(), 2) if y25.any() else None,
                             "2026, %": round(res[y26].mean(), 2) if y26.any() else None,
                             "безубыточный винрейт": f"{sl / (tp + sl) * 100:.0f}%"})
        w("### Тейк против стопа\n\n«Безубыточный винрейт» — сколько тейков нужно, чтобы выйти в ноль при таком соотношении.\n\n"
          + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    a.round(3).to_csv(f"{OUT}/box4h_tp_all.csv.gz", index=False)
    print(open(f"{OUT}/box4h_tp_summary.md").read()[:3000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
