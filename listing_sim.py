"""
Шорт новых листингов — симуляция депозита и поиск фильтров. Только история. Живых ботов не трогает.

СДЕЛКА: шорт по открытию часа через N дней после листинга (первая часовая свеча фьючерса в архиве, листинги
  с 2021-02-01), держим до 30 дней. Варианты: вход 30д или 1д; стоп/тейк 30/20, 20/20, 30/30.
  Стоп и тейк в одной свече — стоп. Комиссия 0,1% на круг, фандинг учтён.
  Если монету делистнули раньше срока — закрываем по последней цене (если сделка прожила хотя бы сутки).
  «Проскальзывание стопа» — вариант, где стоп исполняется на 3% хуже (сквизы на новых монетах резкие).
ДЕПОЗИТ: $100, 3 / 5 / 8 / 10 слотов (на сделку — текущий депозит / число слотов), плечо 1x и 2x.
  Все слоты заняты — сигнал пропускаем. Одновременные сигналы — сначала монеты с большим оборотом первых суток.
ФИЛЬТРЫ: признаки на момент входа; подбираем на 2021–2024, проверяем на 2025–2026.
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL
import short_struct as SS

OUT = os.environ.get("OUT_DIR", "results/listing_sim")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
HOLD = 30 * 24
FEE = 0.1
SLIP = 3.0
VARIANTS = [(30, 30, 20), (30, 20, 20), (30, 30, 30), (1, 30, 30), (1, 30, 20)]   # (вход, дней; стоп; тейк)
SLOTS, LEVS = [3, 5, 8, 10], [1, 2]


def trade(o, h, l, c, t, j, stop, take, fund):
    n = min(HOLD, len(c) - j)
    if n < 24:
        return None
    e = o[j]
    hh, ll = h[j:j + n], l[j:j + n]
    up, dn = hh >= e * (1 + stop / 100), ll <= e * (1 - take / 100)
    js = int(np.argmax(up)) if up.any() else n
    jt = int(np.argmax(dn)) if dn.any() else n
    if js == n and jt == n:
        r, k, how = (1 - c[j + n - 1] / e) * 100, n, ("время" if n == HOLD else "делистинг")
    elif js <= jt:
        r, k, how = -stop, js + 1, "стоп"
    else:
        r, k, how = take, jt + 1, "тейк"
    f = fund.short_gain(t[j], t[j] + k * H)
    return {"r": r + f - FEE, "часов": k, "выход": how, "фандинг": f}


def coin(sym, btc):
    k = PL.klines_any("um", sym, dt.date(2021, 1, 15), dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 48:
        return []
    t = k.open_time.to_numpy().astype("int64")
    if t[0] < SS.LIST_FROM:
        return []
    o, h, l, c, qv = (k[x].to_numpy() for x in ("open", "high", "low", "close", "quote_volume"))
    fund = SS.Fund(*SS.funding_raw(sym))
    rows = []
    for days in sorted({v[0] for v in VARIANTS}):
        j = int(np.searchsorted(t, t[0] + days * 24 * H))
        if j >= len(t) - 24:
            continue
        ts = int(t[j])
        bi = np.searchsorted(btc[0], ts, side="right") - 1
        bl = np.searchsorted(btc[0], ts - 30 * 24 * H, side="right") - 1
        ft, fr = fund.t, np.diff(fund.c)
        recent = fr[(ft > ts - 3 * 24 * H) & (ft <= ts)] if len(ft) else np.array([])
        row = {"symbol": sym, "листинг": int(t[0]), "вход через, д": days, "t": ts,
               "оборот 1-х суток $": qv[:24].sum(), "оборот 24ч до входа $": qv[max(0, j - 24):j].sum(),
               "рост от старта, %": (o[j] / o[0] - 1) * 100,
               "от максимума, %": (o[j] / h[:j].max() - 1) * 100,
               "за 7д до входа, %": (o[j] / o[max(0, j - 168)] - 1) * 100,
               "фандинг 3д, %": recent.sum() if len(recent) else np.nan,
               "BTC 30д, %": (btc[1][bi] / btc[1][bl] - 1) * 100 if bi >= 0 and bl >= 0 else np.nan}
        for dd, stop, take in VARIANTS:
            if dd != days:
                continue
            x = trade(o, h, l, c, t, j, stop, take, fund)
            if x is None:
                continue
            key = f"стоп{stop}/тейк{take}"
            row[key], row[f"{key} часов"], row[f"{key} выход"] = x["r"], x["часов"], x["выход"]
            row[f"{key} проск."] = x["r"] - SLIP if x["выход"] == "стоп" else x["r"]
        rows.append(row)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(s for s in Hst.list_symbols() if s.endswith("USDT"))
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    b = PL.klines_any("um", "BTCUSDT", dt.date(2020, 12, 1), dt.datetime.now(dt.timezone.utc).date())
    btc = (b.open_time.to_numpy(), b.open.to_numpy())

    def work(s):
        try:
            return coin(s, btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/l{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} строк, монет {len(mine)}")


def deposit(g, col, slots, lev):
    g = g.dropna(subset=[col]).sort_values(["t", "оборот 1-х суток $"], ascending=[True, False])
    base = col.replace(" проск.", "")
    eq, peak, mdd = 100.0, 100.0, 0.0
    opened, taken, skipped, yearly, monthly = [], 0, 0, {}, {}

    def settle(upto):
        nonlocal eq, peak, mdd
        for b in sorted(x for x in opened if x[0] <= upto):
            opened.remove(b)
            eq += b[1]
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak * 100)
            m = pd.Timestamp(b[0], unit="ms").strftime("%Y-%m")
            monthly[m] = monthly.get(m, 0) + b[1] / max(eq - b[1], 1e-9) * 100
            yearly.setdefault(pd.Timestamp(b[0], unit="ms").year, []).append(b[1])

    for t, r, hrs in zip(g.t, g[col], g[f"{base} часов"]):
        settle(t)
        if len(opened) >= slots:
            skipped += 1
            continue
        if eq <= 1:
            break
        opened.append((t + hrs * H, eq / slots * lev * r / 100))
        taken += 1
    settle(float("inf"))
    return eq, mdd, taken, skipped, yearly, (min(monthly.values()) if monthly else np.nan)


def st(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"сделок": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {"сделок": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(),
            "PF": pos / neg if neg > 0 else np.inf, "серия минусов": best}


FEATS = ["оборот 1-х суток $", "оборот 24ч до входа $", "рост от старта, %", "от максимума, %",
         "за 7д до входа, %", "фандинг 3д, %", "BTC 30д, %"]


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["год"] = pd.to_datetime(ev.t, unit="ms").dt.year
    ev["время входа (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/trades.csv.gz", index=False)
    w = open(f"{OUT}/listing_sim_summary.md", "w").write
    w("# Шорт новых листингов — депозит и фильтры\n\n" + __doc__.split("\n", 2)[2] + "\n")

    w("## 1. Сделки по годам (средняя, % цены без плеча; в скобках винрейт)\n\n")
    rows = []
    for dd, stop, take in VARIANTS:
        key = f"стоп{stop}/тейк{take}"
        g = ev[ev["вход через, д"] == dd].dropna(subset=[key])
        for c in (key, f"{key} проск."):
            row = {"вход": f"{dd}д", "вариант": c}
            for y, x in g.groupby("год"):
                row[str(y)] = f"{x[c].mean():+.2f} ({(x[c] > 0).mean() * 100:.0f}%) n{len(x)}"
            row["всего"] = f"{g[c].mean():+.2f}"
            rows.append(row)
    w(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    g = ev[ev["вход через, д"] == 30]
    w("Как закрываются сделки (вход 30д, стоп30/тейк20): " +
      ", ".join(f"{k} {v / len(g) * 100:.0f}%" for k, v in g["стоп30/тейк20 выход"].value_counts().items()) +
      f"; медиана в сделке {g['стоп30/тейк20 часов'].median() / 24:.0f} дн.\n\n")

    w("## 2. Депозит $100 (2021-02 — 2026-11)\n\n"
      "итог — во что превратились $100; просадка — худшее падение от пика; худший месяц — % депозита; "
      "колонки по годам — прибыль/убыток в $ за год (сложный процент).\n\n")
    rows = []
    for dd, stop, take in VARIANTS:
        key = f"стоп{stop}/тейк{take}"
        g = ev[ev["вход через, д"] == dd]
        for c in (key, f"{key} проск."):
            for lev in LEVS:
                for sl in SLOTS:
                    eq, mdd, n, sk, yearly, wm = deposit(g, c, sl, lev)
                    rows.append({"вход": f"{dd}д", "вариант": c, "плечо": f"x{lev}", "слотов": sl, "сделок": n,
                                 "пропущено": sk, "итог $": eq, "просадка, %": mdd, "худший месяц, %": wm,
                                 **{str(y): sum(v) for y, v in sorted(yearly.items())}})
    w(pd.DataFrame(rows).round(1).to_markdown(index=False) + "\n\n")

    w("## 3. Фильтры (вход 30д, стоп30/тейк20): 5 групп по каждому признаку\n\n")
    g = ev[(ev["вход через, д"] == 30)].dropna(subset=["стоп30/тейк20"])
    summ = []
    for f in FEATS:
        x = g.dropna(subset=[f]).copy()
        try:
            x["группа"] = pd.qcut(x[f], 5, duplicates="drop")
        except ValueError:
            continue
        rows = []
        for gn, y in x.groupby("группа", observed=True):
            a, b = y[y.год <= 2024]["стоп30/тейк20"], y[y.год >= 2025]["стоп30/тейк20"]
            rows.append({"группа": str(gn), "сделок 21–24": len(a), "21–24 средняя": a.mean(),
                         "21–24 винрейт": (a > 0).mean() * 100, "сделок 25–26": len(b), "25–26 средняя": b.mean(),
                         "25–26 винрейт": (b > 0).mean() * 100})
        tb = pd.DataFrame(rows)
        if len(tb) < 2:
            continue
        corr = np.corrcoef(tb["21–24 средняя"], tb["25–26 средняя"])[0, 1] if len(tb) > 2 else np.nan
        summ.append({"признак": f, "разброс 21–24": tb["21–24 средняя"].max() - tb["21–24 средняя"].min(),
                     "лучшая группа по 21–24": tb.loc[tb["21–24 средняя"].idxmax(), "группа"],
                     "она же в 25–26": tb.loc[tb["21–24 средняя"].idxmax(), "25–26 средняя"],
                     "все в 25–26": g[g.год >= 2025]["стоп30/тейк20"].mean(),
                     "согласие лет (корр.)": corr})
        w(f"### {f}\n\n" + tb.round(2).to_markdown(index=False) + "\n\n")
    w("### Сводка по признакам\n\nЛучшая группа выбрана по 2021–2024; помогает, если в 2025–26 она лучше, "
      "чем «все в 25–26».\n\n" + pd.DataFrame(summ).round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/listing_sim_summary.md").read()[:9000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
