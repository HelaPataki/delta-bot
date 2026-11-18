"""
Разблокировки токенов (данные DefiLlama из results/unlocks) × цена фьючерса Binance. Только история.
Живых ботов не трогает.

СОБЫТИЕ: день, когда разблокировалось много сразу (клифф), а не обычная ежедневная линейная выдача:
  размер = прирост за день / всё разблокированное до этого дня (≈ предложение в обороте), %.
  Отдельно: «инсайдеры» (команда, инвесторы, советники, private/seed…) и «все категории».
  Группы размера: 1–3%, 3–7%, ≥7%. События одной монеты ближе 14 дней друг к другу — берём первое.
  Монета сопоставляется с фьючерсом Binance по тикеру из CoinGecko (TICKERUSDT или 1000TICKERUSDT).
ЦЕНА: 1ч-свечи фьючерса; день события = 00:00 UTC (02:00 Киев зимой).
  Изменение цены за 14 / 7 / 3 / 1 день ДО события и за 1 / 3 / 7 / 14 дней ПОСЛЕ — и то же «минус BTC»
  (чтобы убрать общий рынок).
ШОРТ: вход за 7 / 3 / 1 день до события или в день события, выход через 1 / 3 / 7 / 14 дней после события;
  стоп +15% или +20%; комиссия 0,1%, фандинг учтён (как в short_struct).
Проверка: 2021–2024 отдельно от 2025–2026.
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

OUT = os.environ.get("OUT_DIR", "results/unlocks_study")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, DAYMS = 3600 * 1000, 24 * 3600 * 1000
D0 = dt.date(2020, 11, 1)
MIN_SIZE = 1.0
ENTRIES = {"−7д": -7, "−3д": -3, "−1д": -1, "день": 0}
EXITS = {"+1д": 1, "+3д": 3, "+7д": 7, "+14д": 14}
STOPS = [15, 20]
FEE = 0.1


def events():
    meta = pd.read_csv("results/unlocks/unlocks_meta.csv", low_memory=False)
    daily = pd.read_csv("results/unlocks/unlocks_daily.csv.gz")
    perps = set(Hst.list_symbols())
    gsym = {}
    for _ in range(6):
        r = Hst.get("https://api.coingecko.com/api/v3/coins/list")
        if r is not None and r.status_code == 200:
            gsym = {x["id"]: x["symbol"] for x in r.json()}
            break
        import time
        time.sleep(20)
    print("CoinGecko монет:", len(gsym))
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    rows = []
    for _, m in meta.iterrows():
        tk = str(gsym.get(str(m.get("gecko_id")), "") or m.get("metadata.ticker") or "").upper().strip()
        if not tk or tk == "NAN":
            continue
        sym = next((s for s in (f"{tk}USDT", f"1000{tk}USDT") if s in perps), None)
        if sym is None:
            continue
        d = daily[daily.protocol == m.protocol].sort_values("date")
        d = d[(d.date >= "2020-06-01") & (d.date < today)]
        if len(d) < 30:
            continue
        for kind, col in (("инсайдеры", "unlocked_insiders"), ("все", "unlocked_total")):
            v = d[col].to_numpy(float)
            tot = d["unlocked_total"].to_numpy(float)
            inc = np.r_[0, np.diff(v)]
            prev = np.r_[np.nan, tot[:-1]]
            size = np.where(prev > 0, inc / prev * 100, np.nan)
            last = None
            for i in np.where(size >= MIN_SIZE)[0]:
                day = pd.Timestamp(d.date.iloc[i])
                if day < pd.Timestamp("2021-01-01") or (last is not None and (day - last).days < 14):
                    continue
                last = day
                rows.append({"protocol": m.protocol, "symbol": sym, "вид": kind, "дата": d.date.iloc[i],
                             "размер, %": size[i]})
    ev = pd.DataFrame(rows)
    return ev


def coin(sym, evs, btc):
    k = PL.klines_any("um", sym, D0, dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 24 * 30:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    fund = SS.Fund(*SS.funding_raw(sym))

    def px(ts):
        i = np.searchsorted(t, ts)
        return o[i] if i < len(t) and abs(t[i] - ts) < 2 * H else np.nan

    def bpx(ts):
        i = np.searchsorted(btc[0], ts)
        return btc[1][i] if i < len(btc[0]) else np.nan

    rows = []
    for _, e in evs.iterrows():
        ts = int(pd.Timestamp(e["дата"], tz="UTC").timestamp() * 1000)
        p0, b0 = px(ts), bpx(ts)
        if not np.isfinite(p0):
            continue
        r = dict(e)
        r["t"] = ts
        for n in (14, 7, 3, 1):
            p, b = px(ts - n * DAYMS), bpx(ts - n * DAYMS)
            r[f"до {n}д, %"] = (p0 / p - 1) * 100 if p else np.nan
            r[f"до {n}д − BTC"] = r[f"до {n}д, %"] - ((b0 / b - 1) * 100 if b else np.nan)
        for n in (1, 3, 7, 14):
            p, b = px(ts + n * DAYMS), bpx(ts + n * DAYMS)
            r[f"после {n}д, %"] = (p / p0 - 1) * 100 if np.isfinite(p) else np.nan
            r[f"после {n}д − BTC"] = r[f"после {n}д, %"] - ((b / b0 - 1) * 100 if b0 else np.nan)
        for en, de in ENTRIES.items():
            j = int(np.searchsorted(t, ts + de * DAYMS))
            if j >= len(t) or abs(t[j] - (ts + de * DAYMS)) > 2 * H:
                continue
            for ex, dx in EXITS.items():
                hours = (dx - de) * 24
                for st in STOPS:
                    rr, _ = SS.short_trade(o, h, l, c, t, j, hours, st, None, fund)
                    r[f"шорт {en}→{ex} стоп{st}"] = rr
        rows.append(r)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    ev = events()
    ev.to_csv(f"{OUT}/events_list.csv", index=False)
    print("событий:", len(ev), "монет:", ev.symbol.nunique() if len(ev) else 0)


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    ev = pd.read_csv(f"{OUT}/events_list.csv")
    syms = sorted(ev.symbol.unique())
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    b = PL.klines_any("um", "BTCUSDT", D0, dt.datetime.now(dt.timezone.utc).date())
    btc = (b.open_time.to_numpy(), b.open.to_numpy())

    def work(s):
        try:
            return coin(s, ev[ev.symbol == s], btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/u{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} событий, монет {len(mine)}")


def st(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"сделок": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {"сделок": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(),
            "медиана, %": float(np.median(r)), "PF": pos / neg if neg > 0 else np.inf}


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["год"] = pd.to_datetime(ev.t, unit="ms").dt.year
    ev["период"] = np.where(ev.год <= 2024, "2021–24", "2025–26")
    ev["группа"] = pd.cut(ev["размер, %"], [1, 3, 7, 1e9], right=False, labels=["1–3%", "3–7%", "≥7%"])
    ev.to_csv(f"{OUT}/events.csv.gz", index=False)
    w = open(f"{OUT}/unlocks_study_summary.md", "w").write
    w("# Разблокировки и цена\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Событий: {len(ev)}, монет: {ev.symbol.nunique()}.\n\n")
    w("Количество событий: " + ", ".join(f"{k}: {v}" for k, v in
                                        ev.groupby(["вид", "группа"], observed=True).size().items()) + "\n\n")

    w("## 1. Что делает цена вокруг разблокировки (медиана, % — минус BTC)\n\n")
    cols = [f"до {n}д − BTC" for n in (14, 7, 3, 1)] + [f"после {n}д − BTC" for n in (1, 3, 7, 14)]
    tb = ev.groupby(["вид", "группа", "период"], observed=True)[cols].median().round(2)
    tb.insert(0, "событий", ev.groupby(["вид", "группа", "период"], observed=True).size())
    w(tb.to_markdown() + "\n\n")
    w("Доля событий, где монета через 7 дней хуже BTC, %:\n\n")
    tb2 = ev.groupby(["вид", "группа", "период"], observed=True)["после 7д − BTC"].apply(
        lambda x: (x.dropna() < 0).mean() * 100).round(0)
    w(tb2.to_frame("хуже BTC через 7д, %").to_markdown() + "\n\n")

    w("## 2. Шорт вокруг разблокировки (средняя сделка, % цены без плеча, с комиссией и фандингом)\n\n")
    trade_cols = [c for c in ev.columns if c.startswith("шорт ")]
    rows = []
    for kind in ("инсайдеры", "все"):
        for grp in ("1–3%", "3–7%", "≥7%", "≥3%"):
            g = ev[(ev.вид == kind) & ((ev.группа == grp) if grp != "≥3%" else (ev["размер, %"] >= 3))]
            for c in trade_cols:
                a, b = st(g[g.год <= 2024][c]), st(g[g.год >= 2025][c])
                if a.get("сделок", 0) < 20 or b.get("сделок", 0) < 20:
                    continue
                rows.append({"вид": kind, "размер": grp, "сделка": c.replace("шорт ", ""),
                             **{f"21–24 {k}": v for k, v in a.items()}, **{f"25–26 {k}": v for k, v in b.items()}})
    rd = pd.DataFrame(rows)
    if len(rd):
        rd["хуже из двух"] = rd[["21–24 средняя, %", "25–26 средняя, %"]].min(axis=1)
        rd = rd.sort_values("21–24 средняя, %", ascending=False)
        w("### Лучшие 25 по 2021–2024 (рядом 2025–2026)\n\n" + rd.head(25).round(2).to_markdown(index=False) + "\n\n")
        w("### Лучшие 15 по худшему из двух периодов\n\n" +
          rd.sort_values("хуже из двух", ascending=False).head(15).round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/unlocks_study_summary.md").read()[:9000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
