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


def main():
    arch = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    lines, n_new, n_old = [], 0, 0
    for sym, ts in SHORTS:
        T = int(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=KYIV).timestamp())
        cs = sym + "_PERP.A"
        c = L.candle_feats(L.history("ohlcv-history", [cs], "4hour", T - 20 * H4, T + 60).get(cs, []), T)
        ob = L.history("open-interest-history", [cs], "1hour", T - 30 * H, T + 60, convert_to_usd="false").get(cs, [])
        pr = L.history("predicted-funding-rate-history", [cs], "1hour", T - 30 * H, T + 60).get(cs, [])
        st = L.history("funding-rate-history", [cs], "1hour", T - 30 * H, T + 60).get(cs, [])
        hours = L.funding_interval(sym)
        o0, o24, o4, o28 = (oi_at(ob, x) for x in (T, T - 24 * H, T - 4 * H, T - 28 * H))
        res = {}
        for name, fb in (("new", pr), ("old", st)):
            sf = {"fund24": L.fund24(fb, T, hours), "oi_chg24": (o0 / o24 - 1) * 100,
                  "fund24_prev": L.fund24(fb, T - 4 * H, hours), "oi_chg24_prev": (o4 / o28 - 1) * 100}
            res[name] = (L.is_short(c, sf), sf)
        a = arch[(arch.symbol == sym) & (arch.open_time == (T - H4) * 1000)]
        ok_new, sf = res["new"]
        n_new += ok_new
        n_old += res["old"][0]
        px = c["px24"] if c else np.nan
        lines.append(f"{'✅' if ok_new else '❌'} {sym} {ts[5:]}: фандинг {sf['fund24']:.2f}% (архив {a.fund24.iloc[0]:.2f}%), "
                     f"OI {sf['oi_chg24']:+.0f}% (архив {a.oi_chg24.iloc[0]:+.0f}%), цена {px:+.1f}%"
                     + ("" if ok_new else f" · прошлая свеча: фандинг {sf['fund24_prev']:.2f}%, OI {sf['oi_chg24_prev']:+.0f}%"))
        print(lines[-1], flush=True)
    head = (f"🧪 Шорты с 20.09 по новой схеме фандинга\nНОВАЯ СХЕМА: найдено {n_new} из {len(SHORTS)} · "
            f"КАК СЕЙЧАС В БОТЕ: {n_old} из {len(SHORTS)}")
    L.tg(head + "\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
