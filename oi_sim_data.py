"""
Данные для симуляции торговли шорта по фандингу и открытому интересу.

Сигналы G1 и G1b (с 2025-01-01) из results/oi_funding_2025/parts.
Для каждого сигнала и соотношения (стоп 4,3%): вход по рынку через DELAY минут после закрытия свечи
(цена открытия той минутки), выход по стопу / тейку / через 10 дней. Записываются время входа, время выхода,
результат в % от цены и фандинг за время сделки (шорт получает плюсовой фандинг и платит минусовой).
Комиссия здесь НЕ вычитается — её учитывает симуляция.
"""

import glob
import io
import os
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import short_tests as ST
from oi_funding import first_only

IN_DIR = os.environ.get("IN_DIR", "results/oi_funding_2025/parts")
OUT_DIR = os.environ.get("OUT_DIR", "results/oi_sim")
DELAY = int(os.environ.get("DELAY", "15"))
RRS = [float(x) for x in os.environ.get("RR_LIST", "1.8 2.5 3 4").split()]
HOLDS = [float(x) for x in os.environ.get("HOLD_LIST", "240").split()]   # максимум часов в сделке
STOP = 4.3
H4, DAY, MIN = 4 * 3600 * 1000, 24 * 3600 * 1000, 60_000

_fund = {}


def funding_series(sym):
    if sym in _fund:
        return _fund[sym]
    frames = []
    for m in Hst.months(dt.date(2024, 12, 1), dt.datetime.now(dt.timezone.utc).date()):
        r = Hst.get(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if r is None or r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "last_funding_rate" not in df.columns:
            df = df.iloc[:, :3]
            df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        frames.append(df[["calc_time", "last_funding_rate"]].astype(float))
    f = pd.concat(frames).sort_values("calc_time") if frames else pd.DataFrame(columns=["calc_time", "last_funding_rate"])
    _fund[sym] = (f.calc_time.to_numpy(), f.last_funding_rate.to_numpy())
    return _fund[sym]


def run(r):
    t_open = int(r["open_time"])
    t_in = t_open + H4 + DELAY * MIN
    mins = ST.Mins(r["symbol"], t_open)
    m = list(mins.rows(t_in, t_in + MIN))
    if not m:
        return []
    entry = m[0][1]
    ft, fr = funding_series(r["symbol"])
    out = []
    for rr in RRS:
      for hold in HOLDS:
        stop, take = entry * (1 + STOP / 100), entry * (1 - STOP * rr / 100)
        res, t_out, px, first = None, None, None, True
        for t, o, h, l, c in mins.rows(t_in, t_in + int(hold * 3600 * 1000)):
            px, t_out = c, t + MIN
            if h >= stop:
                res, px = "стоп", stop
                break
            if not first and l <= take:
                res, px = "тейк", take
                break
            first = False
        if px is None:
            continue
        res = res or "по времени"
        sel = (ft > t_in) & (ft <= t_out)
        out.append({"symbol": r["symbol"], "open_time": t_open, "entry_time": t_in, "exit_time": t_out,
                    "rr": rr, "hold_h": hold, "result": res, "entry": entry, "exit": px,
                    "pnl_pct": (entry - px) / entry * 100,
                    "funding_pct": fr[sel].sum() * 100,
                    "fund24": r["fund24"], "oi_chg24": r["oi_chg24"], "G1": r["G1"], "G1b": r["G1b"]})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{IN_DIR}/*.csv.gz"))])
    a = a.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    a = a[a.fund24.notna() & a.oi_chg24.notna() & a.px24.notna()].reset_index(drop=True)
    g1 = first_only(a, (a.fund24 <= -0.03) & (a.oi_chg24 >= 20) & (a.px24 >= -2))
    g1b = first_only(a, (a.fund24 <= -0.1) & (a.oi_chg24 >= 20) & (a.px24 >= -2))
    s = a[g1 | g1b].copy()
    s["G1"], s["G1b"] = g1[g1 | g1b].to_numpy(), g1b[g1 | g1b].to_numpy()
    if os.environ.get("ONLY_FILTER"):
        s = s[s.G1 & (s.fund24 <= -0.2) & (s.oi_chg24 >= 60)]
    print(f"сигналов: {len(s)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, s.to_dict("records")))
    t = pd.DataFrame([x for r in res for x in r])
    t.to_csv(f"{OUT_DIR}/oi_sim_trades.csv.gz", index=False)
    print(t.groupby(["rr", "hold_h"]).agg(n=("pnl_pct", "size"), pnl=("pnl_pct", "mean"), fund=("funding_pct", "mean"),
                              hours=("exit_time", lambda x: ((x - t.loc[x.index, "entry_time"]) / 3.6e6).median())))


if __name__ == "__main__":
    main()
