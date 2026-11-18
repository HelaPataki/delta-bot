"""
Календарные шорты (разблокировки и листинги): как сократить цикл сделки. Только история. Живых ботов не трогает.
Главное мерило — ПРИБЫЛЬ НА ДЕНЬ В СЛОТЕ = сумма результатов / сумма дней в сделках (насколько выгодно занимать слот).

РАЗБЛОКИРОВКИ (инсайдеры ≥3%, события из results/unlocks_study):
  вход за 7 / 5 / 3 дня до разблокировки в 00:00, 07:00, 08:00, 10:00 по Киеву или 00:00 UTC;
  выход по времени в тот же час: в день разблокировки / через 1 / через 3 дня после;
  стоп +15 / +20%; тейк нет / −8 / −10 / −12 / −15%; вариант «безубыток» — после −6% стоп переносится на вход.
ЛИСТИНГИ (листинги с 2021-02-01, оборот 24ч перед входом ≥ $50M):
  вход на 30-й день после листинга: в час листинга (как раньше) или в 00:00 / 07:00 / 08:00 / 10:00 по Киеву;
  стоп +20 / +25 / +30%; тейк −10 / −12 / −15 / −20%; держать не дольше 7 / 14 / 30 дней;
  вариант «безубыток» — после −8% стоп переносится на вход.
Стоп и тейк в одной часовой свече — стоп. Комиссия 0,1%, фандинг учтён. Подбор — 2021–2024, проверка — 2025–2026.
"""

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL
import short_struct as SS

OUT = os.environ.get("OUT_DIR", "results/cycle_grid")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, DAYMS = 3600 * 1000, 24 * 3600 * 1000
FEE = 0.1
TIMES = ["00 Киев", "07 Киев", "08 Киев", "10 Киев", "00 UTC"]
U_BEFORE, U_AFTER, U_STOPS, U_TAKES, U_BE = [7, 5, 3], [0, 1, 3], [15, 20], [None, 8, 10, 12, 15], 6
L_STOPS, L_TAKES, L_HOLDS, L_BE = [20, 25, 30], [10, 12, 15, 20], [7, 14, 30], 8


def at_time(day, label):
    """day — дата (YYYY-MM-DD), label — время из TIMES → мс UTC."""
    if label == "00 UTC":
        return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    hh = int(label[:2])
    return int(pd.Timestamp(f"{day} {hh:02d}:00", tz="Europe/Kyiv").tz_convert("UTC").timestamp() * 1000)


def trade(o, h, l, c, t, j, j_end, stop, take, be, fund):
    """Шорт с открытия свечи j до открытия свечи j_end (не включая). → (результат %, дней)."""
    n = j_end - j
    if n < 1 or j_end > len(c):
        return np.nan, np.nan
    e = o[j]
    hh, ll = h[j:j_end], l[j:j_end]
    s_lvl = e * (1 + stop / 100)
    t_lvl = e * (1 - take / 100) if take else -np.inf
    b_lvl = e * (1 - be / 100) if be else -np.inf
    armed = False
    res, k = None, n
    for i in range(n):
        cur_stop = e if armed else s_lvl
        if hh[i] >= cur_stop:
            res, k = (0.0 if armed else -stop), i + 1
            break
        if ll[i] <= t_lvl:
            res, k = take, i + 1
            break
        if be and ll[i] <= b_lvl:
            armed = True
    if res is None:
        res = (1 - c[j_end - 1] / e) * 100
    res += fund.short_gain(t[j], t[j] + k * H) - FEE
    return res, k / 24


def unlock_rows(sym, evs, o, h, l, c, t, fund, idx):
    rows = []
    for _, e in evs.iterrows():
        d0 = pd.Timestamp(e["дата"])
        for tl in TIMES:
            for b in U_BEFORE:
                j = idx.get(at_time((d0 - pd.Timedelta(days=b)).date().isoformat(), tl))
                if j is None:
                    continue
                for a in U_AFTER:
                    je = idx.get(at_time((d0 + pd.Timedelta(days=a)).date().isoformat(), tl))
                    if je is None or je <= j:
                        continue
                    for st in U_STOPS:
                        for tk in U_TAKES:
                            for be in (None, U_BE):
                                if be and tk not in (None, 12):
                                    continue
                                r, days = trade(o, h, l, c, t, j, je, st, tk, be, fund)
                                if np.isfinite(r):
                                    rows.append({"стратегия": "разблокировка", "symbol": sym, "t": int(t[j]),
                                                 "время": tl, "вход": f"за {b}д", "выход": f"+{a}д",
                                                 "стоп": st, "тейк": tk or 0, "безубыток": bool(be),
                                                 "держать": "", "r": r, "дней": days})
    return rows


def listing_rows(sym, o, h, l, c, t, qv, fund, idx):
    rows = []
    t0 = int(t[0])
    if t0 < SS.LIST_FROM:
        return rows
    day30 = pd.Timestamp(t0 + 30 * DAYMS, unit="ms", tz="UTC")
    starts = {"час листинга": int(np.searchsorted(t, t0 + 30 * DAYMS))}
    for tl in TIMES[:4]:
        starts[tl] = idx.get(at_time(day30.tz_convert("Europe/Kyiv").date().isoformat(), tl))
    for tl, j in starts.items():
        if j is None or j >= len(t) - 24 or j < 24:
            continue
        if qv[j - 24:j].sum() < 50e6:
            continue
        for hold in L_HOLDS:
            je = min(j + hold * 24, len(t))
            if je - j < 24:
                continue
            for st in L_STOPS:
                for tk in L_TAKES:
                    for be in (None, L_BE):
                        if be and tk not in (15, 20):
                            continue
                        r, days = trade(o, h, l, c, t, j, je, st, tk, be, fund)
                        if np.isfinite(r):
                            rows.append({"стратегия": "листинг", "symbol": sym, "t": int(t[j]), "время": tl,
                                         "вход": "30д", "выход": "", "стоп": st, "тейк": tk, "безубыток": bool(be),
                                         "держать": f"{hold}д", "r": r, "дней": days})
    return rows


def coin(sym, evs):
    k = PL.klines_any("um", sym, dt.date(2020, 11, 1), dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 48:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c, qv = (k[x].to_numpy() for x in ("open", "high", "low", "close", "quote_volume"))
    fund = SS.Fund(*SS.funding_raw(sym))
    idx = {int(x): i for i, x in enumerate(t)}
    rows = []
    if len(evs):
        rows += unlock_rows(sym, evs, o, h, l, c, t, fund, idx)
    rows += listing_rows(sym, o, h, l, c, t, qv, fund, idx)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    ev = pd.read_csv("results/unlocks_study/events.csv.gz")
    ev = ev[(ev["вид"] == "инсайдеры") & (ev["размер, %"] >= 3)][["symbol", "дата", "размер, %"]]
    ev.to_csv(f"{OUT}/ev.csv", index=False)
    lst = pd.read_csv("results/listing_sim/trades.csv.gz")
    syms = sorted(set(ev.symbol) | set(lst.symbol))
    pd.Series(syms, name="symbol").to_csv(f"{OUT}/syms.csv", index=False)
    print("разблокировок:", len(ev), "монет всего:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    ev = pd.read_csv(f"{OUT}/ev.csv")
    syms = pd.read_csv(f"{OUT}/syms.csv").symbol.tolist()
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, ev[ev.symbol == s])
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/g{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} строк, монет {len(mine)}")


def agg(g):
    r, d = g.r.to_numpy(), g["дней"].to_numpy()
    if not len(r):
        return {}
    return {"сделок": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(),
            "дней в сделке": d.mean(), "% в день в слоте": r.sum() / d.sum()}


KEYS = ["время", "вход", "выход", "держать", "стоп", "тейк", "безубыток"]


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True)
    ev["период"] = np.where(pd.to_datetime(ev.t, unit="ms").dt.year <= 2024, "21–24", "25–26")
    ev = ev.fillna({"выход": "", "держать": ""})
    ev.to_csv(f"{OUT}/grid.csv.gz", index=False)
    w = open(f"{OUT}/cycle_grid_summary.md", "w").write
    w("# Календарные шорты: короче цикл — больше прибыли на слот\n\n" + __doc__.split("\n", 2)[2] + "\n")
    for strat in ("разблокировка", "листинг"):
        g0 = ev[ev.стратегия == strat]
        rows = []
        for key, g in g0.groupby(KEYS, dropna=False):
            a, b = agg(g[g.период == "21–24"]), agg(g[g.период == "25–26"])
            if a.get("сделок", 0) < 25 or b.get("сделок", 0) < 25:
                continue
            rows.append({**dict(zip(KEYS, key)), **{f"21–24 {k}": v for k, v in a.items()},
                         **{f"25–26 {k}": v for k, v in b.items()}})
        rd = pd.DataFrame(rows)
        if not len(rd):
            continue
        rd["% в день: хуже из двух"] = rd[["21–24 % в день в слоте", "25–26 % в день в слоте"]].min(axis=1)
        drop = ["вход", "выход"] if strat == "листинг" else ["держать"]
        w(f"## {strat.capitalize()}\n\n### Лучшие 20 по прибыли на день в слоте (по худшему из двух периодов)\n\n")
        w(rd.sort_values("% в день: хуже из двух", ascending=False).head(20).drop(columns=drop).round(2)
          .to_markdown(index=False) + "\n\n")
        w("### Лучшие 10 по средней сделке (по 2021–2024)\n\n")
        w(rd.sort_values("21–24 средняя, %", ascending=False).head(10).drop(columns=drop).round(2)
          .to_markdown(index=False) + "\n\n")
        best = rd.sort_values("% в день: хуже из двух", ascending=False).iloc[0]
        same = rd
        for k in KEYS:
            if k != "время":
                same = same[same[k] == best[k]]
        w("### Время входа при лучших остальных настройках\n\n")
        w(same[["время", "21–24 средняя, %", "21–24 % в день в слоте", "25–26 средняя, %", "25–26 % в день в слоте",
                "25–26 сделок"]].round(2).to_markdown(index=False) + "\n\n")
        if strat == "разблокировка":
            base = rd[(rd.время == "00 UTC") & (rd.вход == "за 7д") & (rd.выход == "+1д") & (rd.стоп == 20) &
                      (rd.тейк == 0) & (~rd.безубыток)]
        else:
            base = rd[(rd.время == "час листинга") & (rd.держать == "30д") & (rd.стоп == 30) & (rd.тейк == 20) &
                      (~rd.безубыток)]
        w("Для сравнения — текущий вариант:\n\n" + base.drop(columns=drop).round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/cycle_grid_summary.md").read()[:9000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
