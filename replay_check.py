"""
ВРЕМЕННАЯ проверка (удалить после): находит ли живой бот по данным Coinalyze прошлые сигналы, которые точно были в архиве.
Живой бот не меняется — отсюда только вызываются его функции (live.py).
Старые даты: Coinalyze хранит 5-мин OI и минутки недолго, поэтому для них берутся часовой OI и 5-мин свечи для POC (приблизительно).
"""

import os
import time
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live as L

KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600

SIGNALS = [  # (сторона, монета, свеча закрылась — Киев)
    ("шорт", "BLURUSDT", "2026-10-29 06:00"), ("шорт", "POLYXUSDT", "2026-10-25 14:00"),
    ("шорт", "TUSDT", "2026-10-25 06:00"), ("шорт", "ORDERUSDT", "2026-10-24 15:00"),
    ("шорт", "MEWUSDT", "2026-10-21 07:00"),
    ("лонг", "DOLOUSDT", "2026-11-03 02:00"), ("лонг", "SECZUSDT", "2026-11-02 22:00"),
    ("лонг", "CLOUSDT", "2026-11-02 10:00"), ("лонг", "NOWUSDT", "2026-11-02 06:00"),
]


def oi_at(bars, X, step):
    best = None
    for b in bars:
        end = b["t"] + step
        if end <= X and X - end <= max(1800, step):
            best = b
    return best["c"] if best else np.nan


def main():
    arch_s = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    arch_l = pd.read_csv("results/delay8/long/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    now = time.time()
    lines, found = ["🧪 Проверка бота на прошлых сигналах (данные Coinalyze)"], 0
    for side, sym, ts in SIGNALS:
        T = int(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=KYIV).timestamp())
        cs = sym + "_PERP.A"
        bars = L.history("ohlcv-history", [cs], "4hour", T - 20 * H4, T + 60).get(cs, [])
        c = L.candle_feats(bars, T)
        ms = (T - H4) * 1000
        if side == "шорт":
            step = 300 if now - T < 4 * 86400 else 3600
            iv = "5min" if step == 300 else "1hour"
            fb = L.history("funding-rate-history", [cs], "1hour", T - 30 * H, T + 60).get(cs, [])
            ob = L.history("open-interest-history", [cs], iv, T - 30 * H, T + 60, convert_to_usd="false").get(cs, [])
            hours = L.funding_interval(sym)
            o0, o24, o4, o28 = (oi_at(ob, x, step) for x in (T, T - 24 * H, T - 4 * H, T - 28 * H))
            sf = {"fund24": L.fund24(fb, T, hours), "oi_chg24": (o0 / o24 - 1) * 100,
                  "fund24_prev": L.fund24(fb, T - 4 * H, hours), "oi_chg24_prev": (o4 / o28 - 1) * 100}
            ok = L.is_short(c, sf)
            a = arch_s[(arch_s.symbol == sym) & (arch_s.open_time == ms)]
            av = f"архив: фандинг {a.fund24.iloc[0]:.2f}%, OI {a.oi_chg24.iloc[0]:+.0f}%" if len(a) else "архив: —"
            bv = (f"бот: фандинг {sf['fund24']:.2f}%, OI {sf['oi_chg24']:+.0f}%, цена {c['px24']:+.1f}%" if c else "бот: нет свечей")
            lines.append(f"{'✅' if ok else '❌'} 🔴 {sym} {ts[5:]}\n   {bv}\n   {av} (интервал фандинга {hours:g} ч, OI {iv})")
        else:
            p = None
            if c:
                mb = L.history("ohlcv-history", [cs], "1min", T - H4, T - 60).get(cs, [])
                src = "1 мин"
                if len(mb) < 60:
                    mb = L.history("ohlcv-history", [cs], "5min", T - H4, T - 300).get(cs, [])
                    src = "5 мин, приблизительно"
                    mb = [dict(b) for b in mb]
                    p = None
                    w = [b for b in mb if T - H4 <= b["t"] < T]
                    if len(w) >= 12:
                        top = max(w, key=lambda b: b.get("v") or 0)
                        poc = (top["h"] + top["l"]) / 2
                        hi, lo = max(b["h"] for b in w), min(b["l"] for b in w)
                        p = {"poc": poc, "poc_pos": (poc - lo) / (hi - lo) * 100 if hi > lo else 50, "tick": L.infer_tick(w)}
                else:
                    p = L.poc_info(mb, T)
            ok = L.is_long(c, p)
            a = arch_l[(arch_l.symbol == sym) & (arch_l.open_time == ms)]
            av = f"архив: объём {a.vol_ratio.iloc[0]:.1f}x, дельта {a.delta_pct.iloc[0]:.1f}%" if len(a) else "архив: —"
            bv = (f"бот: объём {c['vol_ratio']:.1f}x, дельта {c['delta_pct']:.1f}%, POC {p['poc_pos']:.0f}% снизу ({src})"
                  if c and p else "бот: нет данных")
            lines.append(f"{'✅' if ok else '❌'} 🟢 {sym} {ts[5:]}\n   {bv}\n   {av}")
        found += ok
        print(lines[-1], flush=True)
    lines.insert(1, f"Найдено {found} из {len(SIGNALS)}")
    L.tg("\n".join(lines))


if __name__ == "__main__":
    main()
