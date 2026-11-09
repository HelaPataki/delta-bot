"""
ВРЕМЕННАЯ проверка (удалить после). Бот не меняется.
Ранний отбор лонгов (за ~42 мин до закрытия) сейчас берёт все монеты, где объём неполной свечи ≥ x2.
При общем падении рынка это почти все монеты (09.11 02:00 — 313 из 386), и проверка после закрытия идёт 20 мин.
Идея: добавить условие на дельту неполной свечи. Проверка на свечах 20.09–09.11 (данные Coinalyze):
сколько монет попадёт в список при разных порогах дельты и не потеряем ли сигналы
(этап 1: объём ≥ x5 и дельта ≤ −17% на закрытой свече; архивные лонги — для сверки).
Неполная свеча = первые 3 часа (бот видит 3 ч 18 мин).
"""

import datetime as dt
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live as L

KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600
START = int(dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc).timestamp())
END = int(dt.datetime(2026, 11, 9, 4, tzinfo=dt.timezone.utc).timestamp())
HEAVY = int(dt.datetime(2026, 11, 9, 0, tzinfo=dt.timezone.utc).timestamp())  # свеча 02:00 по Киеву
XS = [None, 0, -5, -8, -10]
_orig_cz = L.cz


def polite_cz(*a, **k):
    while True:
        ph = time.time() % H4
        if 15 * 60 <= ph < 100 * 60:
            break
        wait = (15 * 60 - ph) % H4 + 5
        L.log(f"окно живого бота — жду {wait / 60:.0f} мин")
        time.sleep(wait)
    return _orig_cz(*a, **k)


L.cz = polite_cz


def qv(b):
    return (b["o"] + b["h"] + b["l"] + b["c"]) / 4 * (b.get("v") or 0)


def kv(T):
    return dt.datetime.fromtimestamp(T, KYIV).strftime("%d.%m %H:%M")


def send(lines):
    print("\n".join(lines), flush=True)
    chunk = ""
    for ln in lines:
        if len(chunk) + len(ln) > 3800:
            L.tg(chunk)
            chunk = ""
        chunk += ln + "\n"
    if chunk:
        L.tg(chunk)


def main():
    uni = L.universe()
    syms = list(uni)
    Ts = list(range(START + H4, END + 1, H4))
    b4 = L.history("ohlcv-history", syms, "4hour", START - 14 * H4, END + 60)
    b1 = L.history("ohlcv-history", syms, "1hour", START - H4, END + 60)

    part = {}     # (символ, T) -> (объём x неполной, дельта % неполной)
    final = {}    # (символ, T) -> признаки закрытой свечи
    for s in syms:
        bars4, bars1 = b4.get(s, []), b1.get(s, [])
        if not bars4 or not bars1:
            continue
        i4 = {b["t"]: i for i, b in enumerate(bars4)}
        h1 = {b["t"]: b for b in bars1}
        q4 = [qv(b) for b in bars4]
        for T in Ts:
            i = i4.get(T - H4)
            if i is None or i < 12:
                continue
            norm = float(np.median(q4[i - 12:i]))
            hs = [h1.get(T - H4 + k * H) for k in range(3)]
            if norm <= 0 or any(h is None for h in hs):
                continue
            v = sum(h.get("v") or 0 for h in hs)
            bv = sum(h.get("bv") or 0 for h in hs)
            part[(s, T)] = (sum(qv(h) for h in hs) / norm, (2 * bv - v) / v * 100 if v else np.nan)
            final[(s, T)] = L.candle_feats(bars4, T)

    sig = {k for k, f in final.items() if f and f["vol_ratio"] >= 5 and f["delta_pct"] <= -17 and f["turnover"] >= L.MIN_TURNOVER}
    p = pd.read_csv("results/poc_signals.csv")
    tt = pd.to_datetime(p.candle_close_kyiv).dt.tz_localize(KYIV, ambiguous="NaT", nonexistent="NaT")
    p = p[tt.notna()].copy()
    p["T"] = (tt[tt.notna()] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    p = p[(p["T"] > START) & (p["T"] <= END) & (p.vol_ratio >= 5) & (p.delta_pct <= -17) & (p.poc_pos <= 30)
          & (p.close_below_poc == "нет") & (p.turnover_usd >= L.MIN_TURNOVER)]
    rev = {b: s for s, b in uni.items()}
    arch = {(rev[r.symbol], int(r.T)) for r in p.itertuples() if r.symbol in rev}

    lines = ["🧪 Ранний отбор лонгов: условие на дельту неполной свечи (20.09–09.11, Coinalyze)",
             f"Сигналов этапа 1 (объём ≥ x5, дельта ≤ −17%): {len(sig)} · архивных лонгов в списке монет бота: {len(arch)}",
             "Порог дельты · монет в основном списке: в среднем / 95% свечей / максимум / свеча 09.11 02:00 · "
             "этап 1: основной / второй проход / потеряно · архив: потеряно"]
    for X in XS:
        ok = lambda k: X is None or (part[k][1] <= X)  # noqa: E731
        sizes = {}
        for k, (vr, dp) in part.items():
            if vr >= 2 and ok(k):
                sizes[k[1]] = sizes.get(k[1], 0) + 1
        arr = np.array([sizes.get(T, 0) for T in Ts])

        def where(k):
            if k not in part:
                return "нет данных"
            vr = part[k][0]
            if vr >= 2 and ok(k):
                return "осн"
            if vr >= 1.2 and ok(k):
                return "втор"
            return "потеряно"
        w = [where(k) for k in sig]
        wa = [where(k) for k in arch]
        name = "без условия (как сейчас)" if X is None else f"дельта ≤ {X}%"
        lines.append(f"• {name}: {arr.mean():.0f} / {np.percentile(arr, 95):.0f} / {arr.max()} / {sizes.get(HEAVY, 0)} · "
                     f"{w.count('осн')} / {w.count('втор')} / {w.count('потеряно')} · архив {wa.count('потеряно')} из {len(wa)}")
        lost = [k for k in sig if where(k) == "потеряно"]
        for s, T in sorted(lost, key=lambda x: x[1])[:6]:
            vr, dp = part[(s, T)]
            lines.append(f"    потерян {uni[s]} {kv(T)}: за 3 ч объём x{vr:.1f}, дельта {dp:+.0f}% → итог дельта {final[(s, T)]['delta_pct']:.0f}%")
    send(lines)


if __name__ == "__main__":
    main()
