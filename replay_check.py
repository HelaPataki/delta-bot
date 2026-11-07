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

        def g(d, t, k):
            b = d.get(t)
            return b[k] if b and b.get(k) is not None else None

        # прогнозная ставка (Coinalyze predicted-funding-rate-history)
        try:
            p1 = by_t(L.history("predicted-funding-rate-history", [cs], "1hour", T - 3 * H, T + 2 * H).get(cs, []))
            p15 = by_t(L.history("predicted-funding-rate-history", [cs], "15min", T - 3 * H, T + H).get(cs, []))
            perr = ""
        except Exception as e:  # noqa: BLE001
            p1, p15, perr = {}, {}, str(e)[:120]
        last = win[-1] if win else None
        lines = [f"🔎 {sym} {ts[5:]} — интервал {hours:g} ч, выплат за 24ч {len(win)}"]
        if last:
            sT = last[0]
            lines.append(f"Выплата {dt.datetime.fromtimestamp(sT, KYIV):%d.%m %H:%M}: архив {last[2]:.4f}%")
            cand = {
                "прогноз 15м свеча до выплаты, close": g(p15, sT - 900, "c"),
                "прогноз 15м за 15 мин до выплаты, close": g(p15, sT - 1800, "c"),
                "прогноз 1ч свеча до выплаты, close": g(p1, sT - H, "c"),
            }
            for k, v in cand.items():
                lines.append(f"  {k}: {v:.4f}%" if v is not None else f"  {k}: —")
            row = []
            for t in (sT - 2 * 900, sT - 900, sT, sT + 900):
                b = p15.get(t)
                row.append(f"{dt.datetime.fromtimestamp(t, KYIV):%H:%M} o{b['o']:.4f} c{b['c']:.4f}" if b else "—")
            lines.append("  прогноз 15м: " + " | ".join(row))
            if perr:
                lines.append("  ошибка прогноза: " + perr)
        # новая схема: прошлые выплаты — из истории выплат (close часовой свечи, начатой в момент выплаты),
        # выплата в момент закрытия свечи — из прогноза (close 15м свечи перед ней, иначе часовой)
        vals = []
        for t, h, v in win:
            if last and t == last[0] and abs(t - T) < 60:
                pv = g(p15, t - 900, "c")
                if pv is None:
                    pv = g(p1, t - H, "c")
            else:
                pv = g(h1, t, "c")
            if pv is not None:
                vals.append(pv * 8 / hours)
        new24 = np.mean(vals) if vals else np.nan
        lines.insert(1, f"АРХИВ фандинг 24ч: {a24:.3f}% · НОВАЯ СХЕМА: {new24:.3f}% ({len(vals)}/{len(win)}) · сейчас в боте: "
                        f"{np.mean([h1[t - H]['c'] for t in settles if (t - H) in h1]) * 8 / hours:.3f}%")
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
