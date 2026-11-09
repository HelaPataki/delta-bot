"""
Список: все USDT-фьючерсы Binance, которые в 2026 году сделали +50% и больше за 48 часов (1h-свечи).

Рост считается от минимума (low) часовой свечи до максимума (high) в следующие 48 ч.
Если так выходит у нескольких свечей подряд — это один эпизод; его старт — свеча с самым низким low,
пик — максимум за 48 ч от старта. Что было после пика (падение) — тоже записывается, монета остаётся в списке.
Время — как на TradingView: время открытия часовой свечи, по Киеву.
Капа — оценка: текущая капа CoinGecko × цена старта / текущая цена (если монеты нет на CoinGecko — пусто).
Для поиска коробок: ширина диапазона (max high / min low − 1) за 10 дней до старта и рост за 7 дней до старта.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2025-12-01")

import numpy as np
import pandas as pd

import history as Hst

OUT = "results/pump_list"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
FROM = pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000
W = 48
GAIN = 0.50


def kyiv(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(Hst.KYIV).strftime("%Y-%m-%d %H:%M")


def coin(sym, cap):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < W + 2:
        return []
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    t, qv = k.open_time.to_numpy(), k.quote_volume.to_numpy()
    n = len(k)
    pad = np.concatenate([h, np.full(W, np.nan)])
    fmax = np.lib.stride_tricks.sliding_window_view(pad, W + 1)[:n].max(axis=1)   # high от свечи i до i+48
    hit = (fmax / l - 1 >= GAIN) & (t >= FROM)
    rows = []
    i = 0
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hit[j + 1]:
            j += 1
        s = i + int(np.argmin(l[i:j + 1]))
        win = h[s:s + W + 1]
        p = s + int(np.argmax(win))
        after = l[p + 1:p + 1 + W]
        after7 = l[p + 1:p + 1 + 168]
        b10 = slice(max(0, s - 240), s)
        row = {"монета": sym,
               "старт (TV 1Ч)": kyiv(t[s]), "цена старта (low)": l[s],
               "пик (TV 1Ч)": kyiv(t[p]), "цена пика (high)": h[p],
               "рост, %": round((h[p] / l[s] - 1) * 100, 1),
               "часов до пика": p - s,
               "падение от пика за 48ч, %": round((after.min() / h[p] - 1) * 100, 1) if len(after) else None,
               "падение от пика за 7д, %": round((after7.min() / h[p] - 1) * 100, 1) if len(after7) else None,
               "цена через 7д от старта, %": round((c[s + 168] / l[s] - 1) * 100, 1) if s + 168 < n else None,
               "коробка 10д до старта, %": round((h[b10].max() / l[b10].min() - 1) * 100, 1) if s >= 240 else None,
               "рост за 7д до старта, %": round((c[s - 1] / c[s - 168] - 1) * 100, 1) if s >= 168 else None,
               "оборот за 48ч, $M": round(qv[s:s + W + 1].sum() / 1e6, 1),
               "капа старта, $M": round(cap * l[s] / c[-1] / 1e6, 1) if cap else None}
        rows.append(row)
        i = max(j + 1, p + 1)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    try:
        caps = Hst.coingecko_mcaps()
    except Exception as e:
        print("CoinGecko:", e)
        caps = {}
    uni = {s: caps.get(Hst.base_asset(s)) for s in syms}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni), "с капой:", sum(v is not None for v in uni.values()))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]

    def work(sym):
        try:
            return coin(sym, uni[sym])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, эпизодов {len(rows)}")


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True).sort_values("старт (TV 1Ч)").reset_index(drop=True)
    a.to_csv(f"{OUT}/pump_list_2026.csv", index=False)
    cap_ok = a["капа старта, $M"].between(10, 400)
    w = open(f"{OUT}/pump_list_2026.md", "w").write
    w("# Монеты с ростом +50% и больше за 48 часов, 2026 год (Binance USDT-фьючерсы, 1h)\n\n"
      "Рост — от минимума часовой свечи старта до максимума в следующие 48 ч. Время — как на TradingView "
      "(открытие часовой свечи, по Киеву). Капа — оценка на момент старта.\n\n"
      f"Всего эпизодов: {len(a)}, монет: {a['монета'].nunique()}. "
      f"С капой $10M–400M: {cap_ok.sum()} эпизодов, {a[cap_ok]['монета'].nunique()} монет.\n\n"
      "По месяцам: " + ", ".join(f"{m}: {g}" for m, g in a.groupby(a['старт (TV 1Ч)'].str[:7]).size().items()) + "\n\n"
      + a.to_markdown(index=False) + "\n")
    print(a.head(30).to_string())


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
