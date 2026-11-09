"""
Эксперимент (не бот): «коробка по телам → выход 4h-свечи». Правила Анастасии.

Коробка: последние 10 дней (60 свечей 4h) ДО сигнальной свечи, по телам свечей без теней:
  верх = максимум из max(open, close), низ = минимум из min(open, close). Ширина = верх / низ − 1.
Сигнал: зелёная 4h-свеча, у которой часть тела выше верха коробки ≥ 40% всего тела:
  (close − max(open, верх)) / (close − open) ≥ 0.4.
  Повторный сигнал той же монеты — не раньше чем через 10 дней (новая коробка).
Вход: через 15 минут после закрытия сигнальной свечи — по закрытию первой 15-минутки после неё.
Выход: тейк +30%, стопа нет; следим 30 дней (15-минутные свечи). Не дошло за 30 дней — смотрим, где цена.
Дополнительно (чтобы понять, какой стоп нужен): что было бы со стопом 10 / 15 / 20 / 25 / 30%.
Фильтры: капа $10M–400M (оценка: текущая капа × цена / текущая цена), оборот сигнальной свечи ≥ $50K.
Сигналы с 2025-01-01. Время — как на TradingView: открытие 4h-свечи, по Киеву.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-11-01")

import numpy as np
import pandas as pd

import history as Hst

OUT = "results/box4h"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
SIG_FROM = pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000
BOX = 60            # 10 дней 4h-свечей
PART = 0.40
TAKE = 0.30
DAYS = 30
STOPS = [10, 15, 20, 25, 30]
H4, M15 = 4 * 3600 * 1000, 15 * 60 * 1000
WIDTHS = [10, 15, 20, 25, 30, 40, None]


def kyiv(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(Hst.KYIV).strftime("%Y-%m-%d %H:%M")


def coin(sym, cap):
    k = Hst.load_klines(sym, "4h")
    if k is None or len(k) < BOX + 10:
        return []
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    t, qv = k.open_time.to_numpy(), k.quote_volume.to_numpy()
    n = len(k)
    btop, bbot = np.maximum(o, c), np.minimum(o, c)
    top = pd.Series(btop).shift(1).rolling(BOX, min_periods=BOX).max().to_numpy()
    bot = pd.Series(bbot).shift(1).rolling(BOX, min_periods=BOX).min().to_numpy()
    vnorm = pd.Series(qv).shift(1).rolling(BOX, min_periods=BOX).median().to_numpy()
    body = c - o
    part = np.where(body > 0, (c - np.maximum(o, top)) / np.where(body > 0, body, 1), 0)
    mcap = cap * c / c[-1]
    sig = (body > 0) & (c > top) & (part >= PART) & (t >= SIG_FROM) & (qv >= Hst.MIN_TURNOVER) \
        & (mcap >= Hst.MCAP_MIN) & (mcap <= Hst.MCAP_MAX)
    idx, last = [], -10 ** 9
    for i in np.flatnonzero(sig):
        if i - last >= BOX:
            idx.append(i)
            last = i
    elig = (t >= SIG_FROM) & (qv >= Hst.MIN_TURNOVER) & (mcap >= Hst.MCAP_MIN) & (mcap <= Hst.MCAP_MAX) & ~np.isnan(top)
    base = [i for i in np.flatnonzero(elig) if i % BOX == 0]       # для сравнения: случайный вход раз в 10 дней
    if not idx and not base:
        return []
    m = Hst.load_klines(sym, "15m")
    if m is None:
        return []
    mt, mh, ml, mc = (m[x].to_numpy() for x in ("open_time", "high", "low", "close"))
    rows = []
    for i, kind in [(i, "сигнал") for i in idx] + [(i, "случайный вход") for i in base]:
        T = t[i] + H4
        j = np.searchsorted(mt, T)                      # первая 15-минутка после закрытия сигнальной
        if j >= len(mt) or mt[j] != T:
            continue
        px = mc[j]                                      # её закрытие = через 15 минут
        end = np.searchsorted(mt, T + M15 + DAYS * 86400 * 1000)
        ph, pl, pc = mh[j + 1:end], ml[j + 1:end], mc[j + 1:end]
        full = end < len(mt) and len(ph) > 0
        if len(ph) == 0:
            continue
        hit = ph >= px * (1 + TAKE)
        ti = int(np.argmax(hit)) if hit.any() else None
        row = {"тип": kind, "монета": sym, "сигнальная свеча (TV 4Ч)": kyiv(t[i]), "год": pd.Timestamp(t[i], unit="ms").year,
               "низ коробки": bot[i], "верх коробки": top[i],
               "ширина коробки, %": round((top[i] / bot[i] - 1) * 100, 1),
               "доля тела за коробкой, %": round(part[i] * 100, 0),
               "закрытие над верхом, %": round((c[i] / top[i] - 1) * 100, 1),
               "объём свечи к норме": round(qv[i] / vnorm[i], 1) if vnorm[i] else None,
               "вход (через 15 мин)": px, "вход над верхом, %": round((px / top[i] - 1) * 100, 1),
               "капа, $M": round(mcap[i] / 1e6, 1), "30 дней прошло": full,
               "тейк 30%": ti is not None,
               "часов до тейка": round((ti + 1) / 4, 1) if ti is not None else None,
               "худшая просадка до тейка, %": round((pl[:ti + 1].min() / px - 1) * 100, 1) if ti is not None else None,
               "макс. рост за 30д, %": round((ph.max() / px - 1) * 100, 1),
               "худшая просадка за 30д, %": round((pl.min() / px - 1) * 100, 1),
               "цена через 30д, %": round((pc[-1] / px - 1) * 100, 1)}
        for s in STOPS:
            sh = pl <= px * (1 - s / 100)
            si = int(np.argmax(sh)) if sh.any() else None
            if ti is not None and (si is None or ti < si):
                res = TAKE * 100
            elif si is not None:
                res = -s
            else:
                res = (pc[-1] / px - 1) * 100
            row[f"стоп {s}%"] = round(res - 0.1, 2)
        rows.append(row)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps[Hst.base_asset(s)] for s in syms if Hst.base_asset(s) in caps}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]
    if LIMIT:
        mine = mine[:LIMIT]

    def work(sym):
        try:
            return coin(sym, uni[sym])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(work, mine), 1):
            rows += r
            if i % 20 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: сигналов {len(rows)}")


def q(x, p):
    x = x.dropna()
    return round(x.quantile(p), 1) if len(x) else None


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    allr = pd.concat(parts, ignore_index=True).sort_values("сигнальная свеча (TV 4Ч)").reset_index(drop=True)
    a = allr[allr["тип"] == "сигнал"]
    rnd = allr[allr["тип"] == "случайный вход"]
    months = max((pd.to_datetime(a["сигнальная свеча (TV 4Ч)"]).max() - pd.to_datetime(a["сигнальная свеча (TV 4Ч)"]).min()).days / 30.4, 1)
    w = open(f"{OUT}/box4h_summary.md", "w").write
    w("# Коробка по телам → выход 4h-свечи (≥ 40% тела над коробкой), тейк +30%, без стопа\n\n"
      "Коробка — 10 дней по телам свечей. Вход через 15 минут после закрытия сигнальной свечи. Следим 30 дней. "
      "Капа $10M–400M, оборот свечи ≥ $50K, с 2025-01-01. Комиссия 0.1% учтена в колонках со стопами. "
      "% — движение цены (при x2 вдвое больше).\n\n")
    rows = []
    for wd in WIDTHS:
        g = a if wd is None else a[a["ширина коробки, %"] <= wd]
        if not len(g):
            continue
        win = g[g["тейк 30%"]]
        lose = g[~g["тейк 30%"]]
        rows.append({"коробка шире не больше": "любая" if wd is None else f"{wd}%", "сигналов": len(g),
                     "в месяц": round(len(g) / months, 1),
                     "дошли до +30% за 30д": f"{g['тейк 30%'].mean() * 100:.0f}%",
                     "2025": f"{g[g['год'] == 2025]['тейк 30%'].mean() * 100:.0f}%",
                     "2026": f"{g[g['год'] == 2026]['тейк 30%'].mean() * 100:.0f}%",
                     "часов до тейка (медиана)": q(win["часов до тейка"], .5),
                     "просадка до тейка: медиана": q(win["худшая просадка до тейка, %"], .5),
                     "просадка до тейка: у 90% не глубже": q(win["худшая просадка до тейка, %"], .1),
                     "не дошли: цена через 30д (медиана)": q(lose["цена через 30д, %"], .5),
                     "не дошли: худшая просадка (медиана)": q(lose["худшая просадка за 30д, %"], .5)})
    rows.append({"коробка шире не больше": "СЛУЧАЙНЫЙ ВХОД (для сравнения)", "сигналов": len(rnd), "в месяц": None,
                 "дошли до +30% за 30д": f"{rnd['тейк 30%'].mean() * 100:.0f}%",
                 "2025": f"{rnd[rnd['год'] == 2025]['тейк 30%'].mean() * 100:.0f}%",
                 "2026": f"{rnd[rnd['год'] == 2026]['тейк 30%'].mean() * 100:.0f}%",
                 "часов до тейка (медиана)": q(rnd[rnd['тейк 30%']]["часов до тейка"], .5),
                 "просадка до тейка: медиана": q(rnd[rnd['тейк 30%']]["худшая просадка до тейка, %"], .5),
                 "просадка до тейка: у 90% не глубже": q(rnd[rnd['тейк 30%']]["худшая просадка до тейка, %"], .1),
                 "не дошли: цена через 30д (медиана)": q(rnd[~rnd['тейк 30%']]["цена через 30д, %"], .5),
                 "не дошли: худшая просадка (медиана)": q(rnd[~rnd['тейк 30%']]["худшая просадка за 30д, %"], .5)})
    w("## Без стопа: сколько коробок дают +30%\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
      "«Просадка до тейка» — насколько цена уходила ниже входа, прежде чем дойти до +30% (это пришлось бы пересидеть). "
      "«Не дошли» — сделки, которые за 30 дней так и не дали +30%.\n\n")
    rows = []
    for wd in WIDTHS:
        g = a if wd is None else a[a["ширина коробки, %"] <= wd]
        for s in STOPS:
            col = f"стоп {s}%"
            for y in (2025, 2026):
                x = g[g["год"] == y][col]
                if len(x):
                    rows.append({"коробка ≤": "любая" if wd is None else f"{wd}%", "стоп": f"{s}%", "год": y,
                                 "сделок": len(x), "винрейт (тейк 30%)": f"{(x >= 29).mean() * 100:.0f}%",
                                 "в среднем на сделку, %": round(x.mean(), 2), "сумма, %": round(x.sum(), 0)})
    for s_ in STOPS:
        for y in (2025, 2026):
            x = rnd[rnd["год"] == y][f"стоп {s_}%"]
            if len(x):
                rows.append({"коробка ≤": "СЛУЧАЙНЫЙ ВХОД", "стоп": f"{s_}%", "год": y, "сделок": len(x),
                             "винрейт (тейк 30%)": f"{(x >= 29).mean() * 100:.0f}%",
                             "в среднем на сделку, %": round(x.mean(), 2), "сумма, %": round(x.sum(), 0)})
    w("## Если всё же поставить стоп (тейк 30%, до 30 дней)\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    tab = a.copy()
    tab.to_csv(f"{OUT}/box4h_signals.csv", index=False)
    print(open(f"{OUT}/box4h_summary.md").read()[:4000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
