"""
Шорт перед разблокировками — проверка против «случайного шорта» и сделки для общей симуляции. Только история.
Живых ботов не трогает. События берутся из results/unlocks_study/events.csv.gz.

СИГНАЛЬНЫЕ СДЕЛКИ (как в unlocks_study): шорт по открытию часа 00:00 UTC за 7 дней до разблокировки,
  выход через 1 или 3 дня после неё, стоп +15% / +20%. Комиссия 0,1%, фандинг учтён. Сохраняем время входа и выхода.
  Варианты: «инсайдеры ≥3%» и «все ≥7%».
СЛУЧАЙНЫЙ ШОРТ: те же монеты, любой день 00:00 UTC с 2021 года, такая же сделка (8 или 10 дней, тот же стоп), но
  только «чистые» дни — в ближайшие 15 дней у монеты нет заметной разблокировки (инсайдеры ≥1% или все ≥3%).
  Если сигнальный шорт заметно лучше случайного — эффект даёт именно разблокировка, а не общее сползание альтов.
"""

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import pump_life as PL
import short_struct as SS

OUT = os.environ.get("OUT_DIR", "results/unlocks_check")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, DAYMS = 3600 * 1000, 24 * 3600 * 1000
D0 = dt.date(2020, 11, 1)
VARIANTS = [(1, 15), (1, 20), (3, 20)]           # (дней после события, стоп)


def coin(sym, evs):
    k = PL.klines_any("um", sym, D0, dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 24 * 30:
        return [], []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    fund = SS.Fund(*SS.funding_raw(sym))
    idx = {int(x): i for i, x in enumerate(t)}
    sig, base = [], []
    for _, e in evs.iterrows():
        ts = int(pd.Timestamp(e["дата"], tz="UTC").timestamp() * 1000)
        if not ((e["вид"] == "инсайдеры" and e["размер, %"] >= 3) or (e["вид"] == "все" and e["размер, %"] >= 7)):
            continue
        j = idx.get(ts - 7 * DAYMS)
        if j is None:
            continue
        for after, stop in VARIANTS:
            r, hrs = SS.short_trade(o, h, l, c, t, j, (7 + after) * 24, stop, None, fund)
            if np.isfinite(r):
                sig.append({"symbol": sym, "вид": e["вид"], "размер, %": e["размер, %"], "дата": e["дата"],
                            "вариант": f"−7д→+{after}д стоп{stop}", "вход": int(t[j]),
                            "выход": int(t[j] + hrs * H), "r": r})
    near = []
    for _, e in evs.iterrows():
        if (e["вид"] == "инсайдеры" and e["размер, %"] >= 1) or (e["вид"] == "все" and e["размер, %"] >= 3):
            near.append(int(pd.Timestamp(e["дата"], tz="UTC").timestamp() * 1000))
    near = np.array(sorted(near))
    start = pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000
    for j in np.where((t % DAYMS == 0) & (t >= start))[0]:
        if len(near):
            a = np.searchsorted(near, t[j])
            if a < len(near) and near[a] - t[j] <= 15 * DAYMS:
                continue
        for after, stop in VARIANTS:
            r, _ = SS.short_trade(o, h, l, c, t, j, (7 + after) * 24, stop, None, fund)
            if np.isfinite(r):
                base.append({"symbol": sym, "вход": int(t[j]), "вариант": f"−7д→+{after}д стоп{stop}", "r": r})
    return sig, base


def prep():
    os.makedirs(OUT, exist_ok=True)
    ev = pd.read_csv("results/unlocks_study/events.csv.gz")[["symbol", "вид", "дата", "размер, %"]]
    ev.to_csv(f"{OUT}/ev.csv", index=False)
    print("событий:", len(ev), "монет:", ev.symbol.nunique())


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    ev = pd.read_csv(f"{OUT}/ev.csv")
    syms = sorted(ev.symbol.unique())
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, ev[ev.symbol == s])
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return [], []

    S, B = [], []
    with ThreadPoolExecutor(3) as ex:
        for a, b in ex.map(work, mine):
            S += a
            B += b
    pd.DataFrame(S).to_csv(f"{OUT}/parts/s{SHARD}.csv.gz", index=False)
    pd.DataFrame(B).to_csv(f"{OUT}/parts/b{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: сигнальных {len(S)}, случайных {len(B)}")


def read(p):
    out = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        if fn.startswith(p):
            try:
                out.append(pd.read_csv(f"{OUT}/parts/{fn}"))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(out, ignore_index=True)


def combine():
    s, b = read("s"), read("b")
    for x in (s, b):
        x["период"] = np.where(pd.to_datetime(x["вход"], unit="ms").dt.year <= 2024, "2021–24", "2025–26")
    s["группа"] = np.where(s["вид"] == "инсайдеры", "инсайдеры ≥3%", "все ≥7%")
    s.to_csv(f"{OUT}/trades.csv.gz", index=False)
    w = open(f"{OUT}/unlocks_check_summary.md", "w").write
    w("# Шорт перед разблокировками против случайного шорта\n\n" + __doc__.split("\n", 2)[2] + "\n")
    rows = []
    for v in sorted(s["вариант"].unique()):
        for per in ("2021–24", "2025–26"):
            bb = b[(b["вариант"] == v) & (b["период"] == per)].r
            row = {"сделка": v, "период": per, "случайный: сделок": len(bb), "случайный: средняя, %": bb.mean(),
                   "случайный: винрейт, %": (bb > 0).mean() * 100}
            for g in ("инсайдеры ≥3%", "все ≥7%"):
                ss = s[(s["вариант"] == v) & (s["период"] == per) & (s["группа"] == g)].r
                row[f"{g}: сделок"] = len(ss)
                row[f"{g}: средняя, %"] = ss.mean()
                row[f"{g}: винрейт, %"] = (ss > 0).mean() * 100
                row[f"{g}: лучше случайного, п.п."] = ss.mean() - bb.mean()
            rows.append(row)
    w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")
    s["год"] = pd.to_datetime(s["вход"], unit="ms").dt.year
    b["год"] = pd.to_datetime(b["вход"], unit="ms").dt.year
    v = "−7д→+1д стоп20"
    by = pd.DataFrame({"инсайдеры ≥3%": s[(s["вариант"] == v) & (s["группа"] == "инсайдеры ≥3%")].groupby("год").r.mean(),
                       "сделок": s[(s["вариант"] == v) & (s["группа"] == "инсайдеры ≥3%")].groupby("год").size(),
                       "случайный": b[b["вариант"] == v].groupby("год").r.mean()})
    w(f"## По годам ({v}), средняя сделка, %\n\n" + by.round(2).to_markdown() + "\n")
    print(open(f"{OUT}/unlocks_check_summary.md").read())


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
