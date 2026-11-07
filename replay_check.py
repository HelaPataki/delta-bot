"""
ВРЕМЕННАЯ проверка (удалить после). Живой бот не меняется — только его функции.
16 шортов из архива (с 20.09.2026): нашёл бы их бот, если читать фандинг из истории ПРОГНОЗНОЙ ставки
(predicted-funding-rate-history), а не из истории выплат, как сейчас. Всё остальное — как в боте.
OI для старых дат — по часовым свечам (5-мин Coinalyze хранит недолго).
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live as L

KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600
SHORTS = [("CELRUSDT", "2026-09-20 19:00"), ("KERNELUSDT", "2026-09-22 07:00"), ("RLCUSDT", "2026-10-05 19:00"),
          ("NMRUSDT", "2026-10-06 19:00"), ("OGNUSDT", "2026-10-08 15:00"), ("CTSIUSDT", "2026-10-08 19:00"),
          ("KAIAUSDT", "2026-10-09 19:00"), ("RSRUSDT", "2026-10-12 15:00"), ("GTCUSDT", "2026-10-12 23:00"),
          ("CTKUSDT", "2026-10-13 15:00"), ("BIRBUSDT", "2026-10-21 07:00"), ("MEWUSDT", "2026-10-21 07:00"),
          ("ORDERUSDT", "2026-10-24 15:00"), ("TUSDT", "2026-10-25 06:00"), ("POLYXUSDT", "2026-10-25 14:00"),
          ("BLURUSDT", "2026-10-29 06:00")]


def oi_at(bars, X, step=H):
    best = None
    for b in bars:
        if b["t"] + step <= X and X - (b["t"] + step) <= max(1800, step):
            best = b
    return best["c"] if best else np.nan


_iv = {}


def interval_as_live(sym, T):
    """Интервал так, как его увидел бы живой бот в день сигнала: последний в архиве за предыдущий месяц, иначе 4 ч."""
    d = dt.datetime.fromtimestamp(T, dt.timezone.utc).date().replace(day=1)
    pm = (d - dt.timedelta(days=1)).replace(day=1)
    key = (sym, pm)
    if key not in _iv:
        import io, zipfile, requests
        hours = None
        try:
            r = requests.get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{pm:%Y-%m}.zip", timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]))
                df.columns = [c.strip().lower() for c in df.columns]
                hours = float(df.funding_interval_hours.iloc[-1])
        except Exception:  # noqa: BLE001
            pass
        _iv[key] = hours or 4.0
    return _iv[key]


def detect_interval(settled, X, fallback):
    """Интервал фандинга по истории выплат за 48 ч до X: ставка меняется в момент выплаты."""
    bars = [b for b in settled if X - 48 * H <= b["t"] and b["t"] + H <= X]
    ch = [bars[i]["t"] for i in range(1, len(bars)) if abs(bars[i]["c"] - bars[i - 1]["c"]) > 1e-9]
    gaps = np.diff(ch) / H if len(ch) >= 3 else []
    if len(gaps) < 2:
        return fallback, "архив"
    g = float(np.median(gaps))
    return (1.0 if g < 1.5 else 2.0 if g < 3 else 4.0 if g < 6 else 8.0), "по выплатам"


def main():
    arch = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    lines, n_new, n_old = [], 0, 0
    for sym, ts in SHORTS:
        T = int(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=KYIV).timestamp())
        cs = sym + "_PERP.A"
        c = L.candle_feats(L.history("ohlcv-history", [cs], "4hour", T - 20 * H4, T + 60).get(cs, []), T)
        ob = L.history("open-interest-history", [cs], "1hour", T - 30 * H, T + 60, convert_to_usd="false").get(cs, [])
        pr = L.history("predicted-funding-rate-history", [cs], "1hour", T - 30 * H, T + 60).get(cs, [])
        st = L.history("funding-rate-history", [cs], "1hour", T - 56 * H, T + 60).get(cs, [])
        hours_old = L.funding_interval(sym)
        hours = hours_p = interval_as_live(sym, T)
        src = "как у живого бота в тот день"
        o0, o24, o4, o28 = (oi_at(ob, x) for x in (T, T - 24 * H, T - 4 * H, T - 28 * H))
        res = {}
        # прошлая свеча: средняя ФАКТИЧЕСКАЯ выплаченная ставка за (T−28ч, T−4ч] (история выплат, ступеньками), в пересчёте на 8 ч
        sv = [b["c"] for b in st if T - 28 * H <= b["t"] and b["t"] + H <= T - 4 * H and b.get("c") is not None]
        prev_settled = float(np.mean(sv)) * 8 / hours if sv else np.nan
        for name, fb, h, fprev in (("new", pr, hours, prev_settled), ("old", st, hours_old, L.fund24(st, T - 4 * H, hours_old))):
            sf = {"fund24": L.fund24(fb, T, h), "oi_chg24": (o0 / o24 - 1) * 100,
                  "fund24_prev": fprev, "oi_chg24_prev": (o4 / o28 - 1) * 100}
            res[name] = (L.is_short(c, sf), sf)
        a = arch[(arch.symbol == sym) & (arch.open_time == (T - H4) * 1000)]
        ok_new, sf = res["new"]
        n_new += ok_new
        n_old += res["old"][0]
        px = c["px24"] if c else np.nan
        lines.append(f"{'✅' if ok_new else '❌'} {sym} {ts[5:]}: фандинг {sf['fund24']:.2f}% (архив {a.fund24.iloc[0]:.2f}%), "
                     f"OI {sf['oi_chg24']:+.0f}% (архив {a.oi_chg24.iloc[0]:+.0f}%), цена {px:+.1f}%, интервал {hours:g} ч ({src})"
                     + ("" if ok_new else f" · прошлая свеча: фандинг {sf['fund24_prev']:.2f}%, OI {sf['oi_chg24_prev']:+.0f}%"))
        print(lines[-1], flush=True)
    head = (f"🧪 Шорты с 20.09: сигнальная свеча — прогноз, прошлая — фактические выплаты\nНОВАЯ СХЕМА: найдено {n_new} из {len(SHORTS)} · "
            f"КАК СЕЙЧАС В БОТЕ: {n_old} из {len(SHORTS)}")
    L.tg(head + "\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
