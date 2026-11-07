"""
ВРЕМЕННАЯ проверка (удалить после): как правильно читать фандинг из Coinalyze.
Для 5 прошлых шортов сравнивает фандинг 24ч, посчитанный разными способами, с настоящими выплатами из архива Binance,
и показывает детали правила «только первый сигнал подряд». Живой бот не меняется.
"""

import io
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import live as L

KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600
SHORTS = [("BLURUSDT", "2026-10-29 06:00"), ("POLYXUSDT", "2026-10-25 14:00"), ("TUSDT", "2026-10-25 06:00"),
          ("ORDERUSDT", "2026-10-24 15:00"), ("MEWUSDT", "2026-10-21 07:00")]


def archive_funding(sym):
    r = requests.get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-2026-10.zip", timeout=60)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip().lower() for c in df.columns]
    return [(int(t) // 1000, float(h), float(v) * 100) for t, h, v in
            zip(df.calc_time, df.funding_interval_hours, df.last_funding_rate)]


def by_t(bars):
    return {b["t"]: b for b in bars}


def oi_at(bars, X, step):
    best = None
    for b in bars:
        if b["t"] + step <= X and X - (b["t"] + step) <= max(1800, step):
            best = b
    return best["c"] if best else np.nan


def main():
    for sym, ts in SHORTS:
        T = int(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=KYIV).timestamp())
        cs = sym + "_PERP.A"
        arch = archive_funding(sym)
        win = [(t, h, v) for t, h, v in arch if T - 24 * H < t <= T + 59]
        hours = win[-1][1] if win else L.funding_interval(sym)
        settles = [t for t, _, _ in win]
        a24 = np.mean([v * 8 / h for _, h, v in win]) if win else np.nan
        h1 = by_t(L.history("funding-rate-history", [cs], "1hour", T - 30 * H, T + 2 * H).get(cs, []))
        m15 = by_t(L.history("funding-rate-history", [cs], "15min", T - 26 * H, T + 2 * H).get(cs, []))
        m5 = by_t(L.history("funding-rate-history", [cs], "5min", T - 25 * H, T + H).get(cs, []))

        def mean(get):
            vals = [get(s) for s in settles]
            vals = [v for v in vals if v is not None]
            return (np.mean(vals) * 8 / hours, len(vals)) if vals else (np.nan, 0)

        def g(d, t, k):
            b = d.get(t)
            return b[k] if b and b.get(k) is not None else None

        methods = {
            "A сейчас: 1ч свеча до выплаты, close": mean(lambda s: g(h1, s - H, "c")),
            "B 1ч свеча до выплаты, low": mean(lambda s: g(h1, s - H, "l")),
            "C 15м свеча за 15 мин до выплаты, close": mean(lambda s: g(m15, s - 30 * 60, "c")),
            "D 15м свеча до выплаты, close": mean(lambda s: g(m15, s - 15 * 60, "c")),
            "E 5м свеча за 5 мин до выплаты, close": mean(lambda s: g(m5, s - 10 * 60, "c")),
            "F 1ч свеча ПОСЛЕ выплаты, open": mean(lambda s: g(h1, s, "o")),
        }
        lines = [f"🔎 {sym} {ts[5:]} — фандинг 24ч (на 8ч), интервал {hours:g} ч, выплат {len(win)}",
                 f"АРХИВ: {a24:.3f}%"]
        for k, (v, n) in methods.items():
            lines.append(f"{k}: {v:.3f}% ({n}/{len(win)})")
        last = win[-1] if win else None
        if last:
            s = last[0]
            lines.append(f"Последняя выплата {dt.datetime.fromtimestamp(s, KYIV):%d.%m %H:%M}: архив {last[2]:.4f}%")
            for name, d, step in (("1ч", h1, H), ("15м", m15, 900), ("5м", m5, 300)):
                row = []
                for t in (s - 2 * step, s - step, s):
                    b = d.get(t)
                    row.append(f"{dt.datetime.fromtimestamp(t, KYIV):%H:%M} o{b['o']:.4f} c{b['c']:.4f}" if b else "—")
                lines.append(f"  {name}: " + " | ".join(row))
        # правило «первый подряд»
        bars = L.history("ohlcv-history", [cs], "4hour", T - 20 * H4, T + 60).get(cs, [])
        c = L.candle_feats(bars, T)
        ob = L.history("open-interest-history", [cs], "1hour", T - 30 * H, T + 60, convert_to_usd="false").get(cs, [])
        o4, o28 = oi_at(ob, T - 4 * H, H), oi_at(ob, T - 28 * H, H)
        prev_win = [(t, h, v) for t, h, v in arch if T - 28 * H < t <= T - 4 * H + 59]
        a24p = np.mean([v * 8 / h for _, h, v in prev_win]) if prev_win else np.nan
        if c:
            lines.append(f"Прошлая свеча: оборот ${c['turnover_prev']:,.0f}, цена 24ч {c['px24_prev']:+.1f}%, "
                         f"OI 24ч {(o4 / o28 - 1) * 100:+.0f}%, фандинг (архив) {a24p:.3f}% → "
                         f"{'G1 (сигнал НЕ первый)' if c['turnover_prev'] >= 50000 and a24p <= -0.03 and (o4 / o28 - 1) * 100 >= 20 and c['px24_prev'] >= -2 else 'не G1 (сигнал первый)'}")
        msg = "\n".join(lines)
        print(msg, flush=True)
        L.tg(msg)


if __name__ == "__main__":
    main()
