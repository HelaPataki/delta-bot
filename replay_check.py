"""
ВРЕМЕННАЯ проверка (удалить после). Живой бот не меняется — используются только его функции.
Все монеты бота, все 4ч-свечи с 20.09 по 31.10.2026: какие шорты нашла бы НОВАЯ схема фандинга
(сигнальная свеча — прогнозная ставка, прошлая — фактические выплаты) и сравнение с архивом Binance:
 • совпали с архивом, • ЛИШНИЕ (бот прислал бы, а в архиве сигнала нет), • пропущенные.
Coinalyze делится с живым ботом: в окно его работы (за 2ч20 до закрытия свечи и 15 мин после) проверка ждёт.
"""

import datetime as dt
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live as L
from replay_util import interval_as_live, oi_at

KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600
START = int(dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc).timestamp())
END = int(dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc).timestamp())

_orig_cz = L.cz


def polite_cz(*a, **k):
    """Не мешать живому боту: ждать, если сейчас его окно работы."""
    while True:
        ph = time.time() % H4
        if 15 * 60 <= ph < 100 * 60:
            break
        wait = (15 * 60 - ph) % H4 + 5
        L.log(f"окно живого бота — жду {wait / 60:.0f} мин")
        time.sleep(wait)
    return _orig_cz(*a, **k)


L.cz = polite_cz


def kv(T):
    return dt.datetime.fromtimestamp(T, KYIV).strftime("%d.%m %H:%M")


def main():
    arch = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    arch = arch[(arch.open_time >= (START - H4) * 1000) & (arch.open_time < (END - H4) * 1000)]
    arch_set = {(r.symbol, int(r.open_time // 1000) + H4) for r in arch.itertuples()}
    arch_syms = set(pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").symbol)

    uni = L.universe()
    syms = list(uni)
    Ts = list(range(START + H4, END + 1, H4))

    # шаг 1: грубый отбор по 4ч-свечам и 4ч OI (дёшево) — OI 24ч ≥ +40%, цена ≥ −2%, оборот ≥ $50k
    ohlc = L.history("ohlcv-history", syms, "4hour", START - 24 * H4, END + 60)
    oi4 = L.history("open-interest-history", syms, "4hour", START - 12 * H4, END + 60, convert_to_usd="false")
    rough = {}
    for s in syms:
        ob = {b["t"] + H4: b["c"] for b in oi4.get(s, [])}
        for T in Ts:
            c = L.candle_feats(ohlc.get(s, []), T)
            if not c or c["turnover"] < L.MIN_TURNOVER or not (c["px24"] >= -2):
                continue
            o0, o24 = ob.get(T), ob.get(T - 24 * H)
            if o0 and o24 and (o0 / o24 - 1) * 100 >= 40:
                rough.setdefault(s, []).append((T, c))
    L.log(f"монет {len(syms)}, грубых кандидатов: {sum(map(len, rough.values()))} свечей у {len(rough)} монет")

    # шаг 2: точный расчёт по часовым данным. Фандинг = прогноз в моменты выплат по сетке интервала.
    #   A «до закрытия»: выплаты в [T−24ч, T) — выплата ровно в момент закрытия идёт уже в следующую свечу
    #   B «с закрытием»: выплаты в (T−24ч, T] — как в прошлой проверке
    found = {"A": [], "B": []}
    cs = list(rough)
    oi1 = L.history("open-interest-history", cs, "1hour", START - 30 * H, END + 60, convert_to_usd="false")
    pr = L.history("predicted-funding-rate-history", cs, "1hour", START - 32 * H, END + 60)
    for s in cs:
        pm = {x["t"] + H: x["c"] for x in pr.get(s, []) if x.get("c") is not None}
        for T, c in rough[s]:
            hours = interval_as_live(uni[s], T)
            o0, o24, o4, o28 = (oi_at(oi1.get(s, []), x) for x in (T, T - 24 * H, T - 4 * H, T - 28 * H))
            for v in ("A", "B"):
                sf = {"fund24": fgrid(pm, T, hours, v), "oi_chg24": (o0 / o24 - 1) * 100,
                      "fund24_prev": fgrid(pm, T - 4 * H, hours, v), "oi_chg24_prev": (o4 / o28 - 1) * 100}
                if L.is_short(c, sf):
                    found[v].append((uni[s], T, sf, c))

    lines = ["🧪 Все монеты, свечи 20.09–31.10, два способа счёта фандинга против архива"]
    best = None
    for v, name in (("A", "A — выплаты до закрытия свечи"), ("B", "B — включая выплату в момент закрытия")):
        got = {(b, T) for b, T, _, _ in found[v]}
        extra = [f for f in found[v] if (f[0], f[1]) not in arch_set]
        miss = sorted(arch_set - got, key=lambda x: x[1])
        lines.append(f"{name}: совпали {len(got & arch_set)} из {len(arch_set)}, лишних {len(extra)}, пропущено {len(miss)}")
        score = len(got & arch_set) - len(extra)
        if best is None or score > best[0]:
            best = (score, v, extra, miss)
    _, v, extra, miss = best
    lines.append(f"Подробно по способу {v}:")
    for b, T, sf, c in sorted(extra, key=lambda x: x[1]):
        note = "" if b in arch_syms else " · монеты нет в архиве"
        lines.append(f"  ➕ {b} {kv(T)}: фандинг {sf['fund24']:.2f}%, OI {sf['oi_chg24']:+.0f}%, цена {c['px24']:+.1f}%, "
                     f"прошлая: фандинг {sf['fund24_prev']:.2f}%, OI {sf['oi_chg24_prev']:+.0f}%{note}")
    for b, T in miss:
        why = "" if b in uni.values() else " · сейчас вне списка монет бота (капитализация)"
        lines.append(f"  ➖ {b} {kv(T)}{why}")
    print("\n".join(lines), flush=True)
    chunk = ""
    for ln in lines:
        if len(chunk) + len(ln) > 3800:
            L.tg(chunk)
            chunk = ""
        chunk += ln + "\n"
    L.tg(chunk)


def fgrid(pm, X, hours, v):
    """Средний фандинг (% на 8 ч) по выплатам на сетке интервала: прогноз в момент выплаты."""
    step = int(hours * H)
    top = (X - 1) // step * step if v == "A" else X // step * step
    lo = X - 24 * H
    rates, t = [], top
    while (t >= lo if v == "A" else t > lo):
        if t in pm:
            rates.append(pm[t] * 8 / hours)
        t -= step
    return float(np.mean(rates)) if rates else np.nan


if __name__ == "__main__":
    main()
