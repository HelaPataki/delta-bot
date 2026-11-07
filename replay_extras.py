"""
ВРЕМЕННО (удалить вместе с replay_check.py). Только архив Binance, без Coinalyze — можно запускать в любое время.
26 «лишних» шортов новой схемы: что по ним в архиве и почему архив сигнал не дал.
"""

import datetime as dt
import io
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import live as L
from replay_util import interval_as_live

KYIV = ZoneInfo("Europe/Kyiv")
A = "https://data.binance.vision/data/futures/um"
H = 3600
EXTRAS = """CELRUSDT 20.09 23:00|2ZUSDT 26.09 19:00|MEWUSDT 30.09 15:00|ENJUSDT 03.10 03:00|RLCUSDT 05.10 23:00|RVNUSDT 05.10 23:00|
API3USDT 06.10 11:00|METUSDT 08.10 03:00|SKLUSDT 08.10 23:00|CTSIUSDT 08.10 23:00|KAIAUSDT 09.10 15:00|ERAUSDT 10.10 15:00|
RSRUSDT 12.10 11:00|GTCUSDT 12.10 19:00|CTKUSDT 12.10 19:00|WALUSDT 14.10 11:00|MEWUSDT 21.10 11:00|BIRBUSDT 21.10 11:00|
IOSTUSDT 21.10 15:00|ORDERUSDT 24.10 19:00|ANIMEUSDT 24.10 23:00|TUSDT 25.10 10:00|STEEMUSDT 25.10 10:00|
1000XECUSDT 25.10 14:00|HOTUSDT 31.10 02:00|COWUSDT 31.10 22:00"""
_cache = {}


def zcsv(url):
    if url not in _cache:
        df = None
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
                if not str(df.iloc[0, 0]).replace(".", "").isdigit():
                    df.columns = [str(c).strip().lower() for c in df.iloc[0]]
                    df = df.iloc[1:].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            df = None
        _cache[url] = df
    return _cache[url]


def funding(sym):
    fr = [zcsv(f"{A}/monthly/fundingRate/{sym}/{sym}-fundingRate-2026-{m:02d}.zip") for m in (8, 9, 10)]
    fr = [f for f in fr if f is not None]
    if not fr:
        return None
    f = pd.concat(fr)
    f.columns = ["t", "hours", "rate"][:len(f.columns)] if "calc_time" not in f.columns else f.columns
    f = f.rename(columns={"calc_time": "t", "funding_interval_hours": "hours", "last_funding_rate": "rate"})
    f = f.astype({"t": float, "hours": float, "rate": float})
    return f.sort_values("t")


def fund24(f, X):
    w = f[(f.t > (X - 24 * H) * 1000) & (f.t <= X * 1000)]
    return (w.rate * 8 / w.hours).mean() * 100 if len(w) else np.nan, w


def days(T, back):
    d = dt.datetime.fromtimestamp(T, dt.timezone.utc).date()
    return [d - dt.timedelta(days=i) for i in range(back, -1, -1)]


def oi(sym, T):
    parts = [zcsv(f"{A}/daily/metrics/{sym}/{sym}-metrics-{d}.zip") for d in days(T, 2)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    m = pd.concat(parts)
    t = (pd.to_datetime(m.create_time, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    s = pd.Series(m.sum_open_interest.astype(float).to_numpy(), index=t.to_numpy()).sort_index()

    def at(x):
        v = s[s.index <= x]
        return v.iloc[-1] if len(v) else np.nan
    return at


def klines(sym, T):
    parts = [zcsv(f"{A}/daily/klines/{sym}/4h/{sym}-4h-{d}.zip") for d in days(T, 2)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    k = pd.concat(parts).iloc[:, :8]
    k.columns = ["ot", "o", "h", "l", "c", "v", "ct", "qv"]
    k = k.astype(float).drop_duplicates("ot").set_index("ot")
    return k


def main():
    lines = ["🧪 26 лишних шортов: что в архиве Binance"]
    cnt = {}
    for item in EXTRAS.replace("\n", "").split("|"):
        sym, d, hm = item.split()
        T = int(dt.datetime.strptime(f"2026.{d} {hm}", "%Y.%d.%m %H:%M").replace(tzinfo=KYIV).timestamp())
        f = funding(sym)
        if f is None:
            lines.append(f"• {sym} {d} {hm}: архива фандинга нет")
            cnt["нет архива"] = cnt.get("нет архива", 0) + 1
            continue
        f24, w = fund24(f, T)
        f24p, _ = fund24(f, T - 4 * H)
        hrs = sorted(set(w.hours.astype(int)))
        bot_h = interval_as_live(sym, T)
        o = oi(sym, T)
        k = klines(sym, T)
        oc = (o(T) / o(T - 24 * H) - 1) * 100 if o else np.nan
        ocp = (o(T - 4 * H) / o(T - 28 * H) - 1) * 100 if o else np.nan
        px = pxp = tv = np.nan
        if k is not None:
            ot = (T - 4 * H) * 1000
            try:
                px = (k.loc[ot, "c"] / k.loc[ot - 24 * H * 1000, "c"] - 1) * 100
                pxp = (k.loc[ot - 4 * H * 1000, "c"] / k.loc[ot - 28 * H * 1000, "c"] - 1) * 100
                tv = k.loc[ot, "qv"]
            except KeyError:
                pass
        sig = f24 <= -0.2 and oc >= 60 and px >= -2 and tv >= L.MIN_TURNOVER
        prev = f24p <= -0.03 and ocp >= 20 and pxp >= -2
        if not f24 <= -0.2:
            why = "фандинг в архиве слабее порога"
        elif not oc >= 60:
            why = "OI в архиве ниже +60%"
        elif prev:
            why = "в архиве не первый подряд (прошлая свеча уже «горячая»)"
        elif sig:
            why = "в архиве всё проходит → отсеяла капитализация"
        else:
            why = "цена/оборот"
        cnt[why] = cnt.get(why, 0) + 1
        lines.append(f"• {sym} {d} {hm}: архив фандинг {f24:.2f}% (прошлая {f24p:.2f}%), OI {oc:+.0f}% (прошлая {ocp:+.0f}%), "
                     f"интервал в архиве {'/'.join(map(str, hrs))} ч, бот считал {bot_h:g} ч → {why}")
        print(lines[-1], flush=True)
    lines.insert(1, "Итого: " + "; ".join(f"{k} — {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
    msg = "\n".join(lines)
    print(msg)
    L.tg(msg)


if __name__ == "__main__":
    main()
