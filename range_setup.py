"""
Шорт «боковик и сползание» как сетап: вход, стоп, тейк. Только история. Живых ботов не трогает.
Случаи берутся из results/range_study/ranges.csv.gz (боковик 7 дней в коридоре 4–25%, проверка в 00:00 UTC,
одна монета — не чаще раза в 14 дней).

ФИЛЬТРЫ:
  «все боковики» — для сравнения (каждый 4-й случай, чтобы прогон был быстрее);
  «простой» — оборот 24ч ≥ $20M и объём в боковике < 60% от месяца до него;
  «с топами» — простой + топ-трейдеры L/S ≥ 1,8 и выросли за боковик больше чем на 0,1.
ВХОД:
  «сразу» — в момент проверки (00:00 UTC) или в 07:00 по Киеву того же дня;
  «пробой низа» — ждём до 3 суток первую часовую свечу, закрывшуюся ниже минимума боковика, вход на следующей.
ВЫХОД: стоп +8 / +10 / +15%; тейк −8 / −10 / −15 / −20%; держать не дольше 7 / 14 дней.
  Стоп и тейк в одной свече — стоп. Комиссия 0,1%, фандинг учтён.
Мерило — прибыль на день в слоте (как у календарных шортов: разблокировки ≈ 0,9%/день, листинги ≈ 1,6%/день).
"""

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import cycle_grid as G
import pump_life as PL
import short_struct as SS

OUT = os.environ.get("OUT_DIR", "results/range_setup")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, DAYMS = 3600 * 1000, 24 * 3600 * 1000
STOPS, TAKES, HOLDS = [8, 10, 15], [8, 10, 15, 20], [7, 14]


def coin(sym, cases):
    k = PL.klines_any("um", sym, dt.date(2020, 11, 1), dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 24 * 40:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    fund = SS.Fund(*SS.funding_raw(sym))
    idx = {int(x): i for i, x in enumerate(t)}
    rows = []
    for _, s in cases.iterrows():
        j0 = idx.get(int(s.t))
        if j0 is None or j0 < 7 * 24:
            continue
        lo = l[j0 - 7 * 24:j0].min()
        day = pd.Timestamp(int(s.t), unit="ms", tz="UTC").date().isoformat()
        starts = {"сразу 00 UTC": j0, "сразу 07 Киев": idx.get(G.at_time(day, "07 Киев"))}
        br = np.where(c[j0:j0 + 72] < lo)[0]
        starts["пробой низа"] = j0 + int(br[0]) + 1 if len(br) else None
        for en, j in starts.items():
            if j is None or j >= len(t):
                continue
            for hold in HOLDS:
                je = j + hold * 24
                if je > len(t):
                    continue
                for st in STOPS:
                    for tk in TAKES:
                        r, days = G.trade(o, h, l, c, t, j, je, st, tk, None, fund)
                        if np.isfinite(r):
                            rows.append({"symbol": sym, "t": int(t[j]), "фильтр": s.фильтр, "вход": en,
                                         "держать": f"{hold}д", "стоп": st, "тейк": tk, "r": r, "дней": days})
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    e = pd.read_csv("results/range_study/ranges.csv.gz")
    simple = (e["оборот 24ч $"] >= 20e6) & (e["объём боковика к 30д до"] < 0.6)
    tops = simple & (e["топы L/S"] >= 1.8) & (e["топы: изменение"] > 0.1)
    allr = e.iloc[::4].assign(фильтр="все боковики")
    cases = pd.concat([allr, e[simple].assign(фильтр="простой"), e[tops].assign(фильтр="с топами")])
    cases[["symbol", "t", "фильтр"]].to_csv(f"{OUT}/cases.csv", index=False)
    print(cases.фильтр.value_counts().to_dict(), "монет:", cases.symbol.nunique())


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    cs = pd.read_csv(f"{OUT}/cases.csv")
    syms = sorted(cs.symbol.unique())
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, cs[cs.symbol == s])
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/r{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} строк, монет {len(mine)}")


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True)
    ev["период"] = np.where(pd.to_datetime(ev.t, unit="ms").dt.year <= 2024, "21–24", "25–26")
    ev["выход_t"] = ev.t + (ev["дней"] * 24 * H).round().astype("int64")
    ev.to_csv(f"{OUT}/trades.csv.gz", index=False)
    ev[ev.фильтр == "с топами"].to_csv(f"{OUT}/trades_tops.csv.gz", index=False)   # для общей симуляции (combo4.py)
    keys = ["фильтр", "вход", "держать", "стоп", "тейк"]
    rows = []
    for key, g in ev.groupby(keys):
        row = dict(zip(keys, key))
        ok = True
        for per in ("21–24", "25–26"):
            x = g[g.период == per]
            if len(x) < 30:
                ok = False
                break
            row[f"{per} сделок"] = len(x)
            row[f"{per} винрейт, %"] = (x.r > 0).mean() * 100
            row[f"{per} средняя, %"] = x.r.mean()
            row[f"{per} дней"] = x["дней"].mean()
            row[f"{per} % в день в слоте"] = x.r.sum() / x["дней"].sum()
        if ok:
            rows.append(row)
    rd = pd.DataFrame(rows)
    rd["% в день: хуже из двух"] = rd[["21–24 % в день в слоте", "25–26 % в день в слоте"]].min(axis=1)
    w = open(f"{OUT}/range_setup_summary.md", "w").write
    w("# Боковик и сползание — шорт-сетап\n\n" + __doc__.split("\n", 2)[2] + "\n")
    for f in ("с топами", "простой", "все боковики"):
        x = rd[rd.фильтр == f].sort_values("% в день: хуже из двух", ascending=False)
        w(f"## {f}: лучшие 12 по прибыли на день в слоте (по худшему из двух периодов)\n\n")
        w(x.head(12).round(2).to_markdown(index=False) + "\n\n")
    w("## Лучшее по каждому способу входа (все фильтры)\n\n")
    best = rd.sort_values("% в день: хуже из двух", ascending=False).groupby(["фильтр", "вход"]).head(1)
    w(best.round(2).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/range_setup_summary.md").read()[:8000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
