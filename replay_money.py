"""
ВРЕМЕННО (удалить вместе с replay_check.py). Только архив Binance — можно запускать в любое время.
Способ B даёт 8 лишних шортов и теряет 5 архивных. Как бы сыграли эти сделки по деньгам:
вход через 8 мин после закрытия свечи, стоп +4,3%, тейк −7,74%, выход через 24 ч; фандинг из архива, комиссия 0,1%.
Пропущенные считаются тем же кодом — для сверки с архивом.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live as L
from replay_extras import A, funding, zcsv

KYIV = ZoneInfo("Europe/Kyiv")
H = 3600
EXTRA = "API3USDT 06.10 11:00|KAIAUSDT 09.10 15:00|ERAUSDT 10.10 15:00|RSRUSDT 12.10 11:00|GTCUSDT 12.10 19:00|" \
        "CTKUSDT 12.10 19:00|WALUSDT 14.10 11:00|HOTUSDT 31.10 02:00"
MISS = "NMRUSDT 06.10 19:00|KAIAUSDT 09.10 19:00|RSRUSDT 12.10 15:00|GTCUSDT 12.10 23:00|BLURUSDT 29.10 06:00"


def parse(s):
    for item in s.split("|"):
        sym, d, hm = item.split()
        yield sym, f"{d} {hm}", int(dt.datetime.strptime(f"2026.{d} {hm}", "%Y.%d.%m %H:%M").replace(tzinfo=KYIV).timestamp())


def minutes(sym, T):
    d = dt.datetime.fromtimestamp(T, dt.timezone.utc).date()
    parts = [zcsv(f"{A}/daily/klines/{sym}/1m/{sym}-1m-{d + dt.timedelta(days=i)}.zip") for i in (0, 1, 2)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    k = pd.concat(parts).iloc[:, :5]
    k.columns = ["ot", "o", "h", "l", "c"]
    return k.astype(float).drop_duplicates("ot").sort_values("ot").set_index("ot")


def trade(sym, T):
    k = minutes(sym, T)
    if k is None:
        return None
    t0 = (T + L.ENTRY_DELAY_MIN * 60) * 1000
    w = k[(k.index >= t0) & (k.index < t0 + 24 * H * 1000)]
    if len(w) < 60:
        return None
    entry = w.o.iloc[0]
    stop, take = entry * (1 + L.STOP / 100), entry * (1 - L.SHORT_TAKE / 100)
    res, ex, t_exit = "по времени", w.c.iloc[-1], w.index[-1] + 60000
    for t, r in w.iterrows():
        if r.h >= stop:
            res, ex, t_exit = "стоп", stop, t + 60000
            break
        if r.l <= take:
            res, ex, t_exit = "тейк", take, t + 60000
            break
    pnl = (entry / ex - 1) * 100 if res == "по времени" else (L.SHORT_TAKE if res == "тейк" else -L.STOP)
    f = funding(sym)
    fund = 0.0 if f is None else f[(f.t > t0) & (f.t <= t_exit)].rate.sum() * 100
    return res, pnl, fund, pnl + fund - 0.1


def main():
    arch = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    lines, tot = [], {}
    for kind, src in (("➕ лишние (бот прислал бы, в архиве нет)", EXTRA), ("➖ пропущенные (в архиве есть, бот не прислал бы)", MISS)):
        lines.append(kind)
        s = []
        for sym, label, T in parse(src):
            r = trade(sym, T)
            if r is None:
                lines.append(f"  {sym} {label}: нет данных")
                continue
            res, pnl, fund, net = r
            s.append(net)
            a = arch[(arch.symbol == sym) & (arch.open_time == (T - 4 * H) * 1000)]
            chk = f" (архив: {a.result.iloc[0]}, {a.pnl_pct.iloc[0]:+.2f}%)" if len(a) else ""
            lines.append(f"  {sym} {label}: {res}, цена {pnl:+.2f}%, фандинг {fund:+.2f}%, итого {net:+.2f}% "
                         f"(с x2 {net * 2:+.1f}%){chk}")
        tot[kind[0]] = (sum(s), len(s))
        lines.append(f"  сумма: {sum(s):+.2f}% на {len(s)} сделок (с x2 {sum(s) * 2:+.1f}%)")
    (pe, ne), (pm, nm) = tot["➕"], tot["➖"]
    head = ["🧪 Способ B по деньгам: лишние против пропущенных (% от цены, x2 — ROI)",
            f"Итог для B относительно архива: {pe - pm:+.2f}% (с x2 {(pe - pm) * 2:+.1f}%), сделок {ne - nm:+d}"]
    msg = "\n".join(head + lines)
    print(msg)
    L.tg(msg)


if __name__ == "__main__":
    main()
