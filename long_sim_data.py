"""
Данные для симуляции лонгового сетапа (такой же формат, как у oi_sim_data.py).

Сигналы: results/poc_signals.csv — дельта <= −17%, объём >= x5, POC в нижних 30%, закрытие выше POC (с 2025-01-01).
Вход: условный ордер на POC + 5 тиков, начинает действовать через DELAY минут после закрытия свечи,
  срабатывает при касании уровня с любой стороны, ждёт до конца следующей 4h-свечи.
Выход: стоп −4,3%, тейк = 4,3% x соотношение, или по времени (HOLD_LIST часов).
Фандинг: лонг платит плюсовой и получает минусовой (знак противоположный шорту).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as B
import short_tests as ST
from oi_sim_data import funding_series
from poc import kyiv_to_utc

OUT_DIR = os.environ.get("OUT_DIR", "results/long_sim")
DELAY = int(os.environ.get("DELAY", "15"))
RRS = [float(x) for x in os.environ.get("RR_LIST", "1.5 1.8 2.5 3 4").split()]
HOLDS = [float(x) for x in os.environ.get("HOLD_LIST", "24 240").split()]
STOP, TICKS = 4.3, 5
H4, MIN = 4 * 3600 * 1000, 60_000


def run(r):
    t_open = kyiv_to_utc(r["candle_open_kyiv"]).timestamp() * 1000
    t0 = t_open + H4
    t_order = t0 + DELAY * MIN
    mins = ST.Mins(r["symbol"], t_open)
    if not list(mins.rows(t_open, t0)) or not mins.parts:
        return []
    tick = B.infer_tick(mins.parts[0])
    lvl = r["poc"] + TICKS * tick
    prev = next((x[4] for x in mins.rows(t_order - MIN, t_order)), r["close"])
    t_in = None
    for t, o, h, l, c in mins.rows(t_order, t0 + H4):
        if (l <= lvl <= h) or (prev < lvl < l) or (h < lvl < prev):
            t_in = t
            break
        prev = c
    if t_in is None:
        return []
    ft, fr = funding_series(r["symbol"])
    out = []
    for rr in RRS:
        for hold in HOLDS:
            stop, take = lvl * (1 - STOP / 100), lvl * (1 + STOP * rr / 100)
            res, t_out, px, first = None, None, None, True
            for t, o, h, l, c in mins.rows(t_in, t_in + int(hold * 3600 * 1000)):
                px, t_out = c, t + MIN
                if l <= stop:
                    res, px = "стоп", stop
                    break
                if not first and h >= take:
                    res, px = "тейк", take
                    break
                first = False
            if px is None:
                continue
            sel = (ft > t_in) & (ft <= t_out)
            out.append({"symbol": r["symbol"], "open_time": t_open, "entry_time": t_in, "exit_time": t_out,
                        "rr": rr, "hold_h": hold, "result": res or "по времени", "entry": lvl, "exit": px,
                        "pnl_pct": (px - lvl) / lvl * 100, "funding_pct": -fr[sel].sum() * 100,
                        "vol_ratio": r["vol_ratio"], "delta_pct": r["delta_pct"], "strategy": "Лонг (x5, дельта ≤ −17%, POC внизу)"})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p = pd.read_csv("results/poc_signals.csv")
    s = p[(p.delta_sign == "−") & (p.vol_ratio >= 5) & (p.poc_pos <= 30) & (p.close_below_poc == "нет") & p.poc.notna()]
    print(f"лонговых сигналов: {len(s)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, s.to_dict("records")))
    t = pd.DataFrame([x for r in res for x in r])
    t.to_csv(f"{OUT_DIR}/oi_sim_trades.csv.gz", index=False)
    print(t.groupby(["rr", "hold_h"]).agg(n=("pnl_pct", "size"), pnl=("pnl_pct", "mean"), fund=("funding_pct", "mean")))


if __name__ == "__main__":
    main()
