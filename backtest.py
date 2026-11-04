"""
Бэктест лонга по минутным свечам Binance.

Сигналы: results/poc_signals.csv, объём >= x5, дельта <= -17%.
  Условие («гипотеза»): POC в нижних 30% свечи и закрытие выше POC.
  Остальные x5 с дельтой <= -17% считаются для сравнения по тем же правилам.

Сделка:
  вход   — лимитный ордер на покупку по POC + 5 тиков, выставляется на закрытии сигнальной свечи
           (если закрытие уже ниже этой цены — вход сразу по цене закрытия);
  стоп   — -STOP_PCT% от входа, тейк — +STOP_PCT * RR % от входа;
  если в одной минутке задеты и стоп, и тейк — считаем стоп (осторожный вариант);
  в минутку входа проверяется только стоп;
  если за 10 дней после входа ни стопа, ни тейка — закрытие по рынку, результат в %.
Время ожидания лимитки: результат считается для 1 свечи 4h, 24 часов и 10 дней.
Тик: определяется по минутным ценам дня сигнала (наименьший шаг цены).
Комиссии не учитываются.
"""

import io
import os
import time
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from poc import kyiv_to_utc

IN_FILE = os.environ.get("IN_FILE", "results/poc_signals.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results")
STOP_PCT = float(os.environ.get("STOP_PCT", "4.3"))
RR = float(os.environ.get("RR", "1.8"))
TICKS = int(os.environ.get("TICKS", "5"))
VOL_MIN = float(os.environ.get("VOL_MIN", "5"))
POC_LOW_MAX = 30.0
WAIT = {"1 свеча 4h": 4 * 60, "24 часа": 24 * 60, "10 дней": 10 * 24 * 60}   # минуты
HOLD_MIN = 10 * 24 * 60
ARCHIVE = "https://data.binance.vision"
MS_MIN = 60_000

session = requests.Session()
session.headers["User-Agent"] = "delta-bot-backtest"


def get(url, tries=4):
    for i in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return None


def day_minutes(sym, day):
    url = f"{ARCHIVE}/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day:%Y-%m-%d}.zip"
    r = get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    if isinstance(df.iloc[0, 0], str):
        df = df.iloc[1:]
    return df.iloc[:, :5].astype(float).to_numpy()   # open_time, open, high, low, close


def infer_tick(arr):
    prices = np.unique(np.round(arr[:, 1:5].ravel(), 10))
    diffs = np.diff(prices)
    diffs = diffs[diffs > 0]
    return float(np.round(diffs.min(), 10)) if len(diffs) else np.nan


def minute_stream(sym, start_ms):
    """Минутки начиная с start_ms, день за днём (до 21 дня вперёд)."""
    d = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).date()
    for _ in range(22):
        arr = day_minutes(sym, d)
        if arr is not None:
            arr = arr[arr[:, 0] >= start_ms]
            for row in arr:
                yield row
        d += dt.timedelta(days=1)


def run_one(row):
    out = {"fill_wait_min": np.nan, "entry": np.nan, "tick": np.nan, "result": "нет данных",
           "pnl_pct": np.nan, "hours_in_trade": np.nan, "filled_at_close": False}
    open_utc = kyiv_to_utc(row["candle_open_kyiv"])
    sig_day = day_minutes(row["symbol"], open_utc.date())
    if sig_day is None:
        return out
    tick = infer_tick(sig_day)
    out["tick"] = tick
    entry = row["poc"] + TICKS * tick
    stop = entry * (1 - STOP_PCT / 100)
    take = entry * (1 + STOP_PCT * RR / 100)
    t0 = open_utc.timestamp() * 1000 + 4 * 3600 * 1000   # закрытие сигнальной свечи

    filled, fill_t = False, None
    if row["close"] <= entry:                               # цена уже ниже лимитки
        filled, fill_t, entry = True, t0, row["close"]
        stop = entry * (1 - STOP_PCT / 100)
        take = entry * (1 + STOP_PCT * RR / 100)
        out["filled_at_close"] = True
    out["entry"] = entry
    last_close = None
    for t, o, h, l, c in minute_stream(row["symbol"], t0):
        last_close = c
        if not filled:
            if t - t0 >= WAIT["10 дней"] * MS_MIN:
                out["result"] = "не налило"
                return out
            if l <= entry:
                filled, fill_t = True, t
                if l <= stop:
                    out.update(result="стоп", pnl_pct=-STOP_PCT, hours_in_trade=0)
                    break
            continue
        if t - fill_t >= HOLD_MIN * MS_MIN:
            out.update(result="по времени", pnl_pct=(c / entry - 1) * 100, hours_in_trade=(t - fill_t) / 3.6e6)
            break
        if l <= stop:
            out.update(result="стоп", pnl_pct=-STOP_PCT, hours_in_trade=(t - fill_t) / 3.6e6)
            break
        if h >= take:
            out.update(result="тейк", pnl_pct=STOP_PCT * RR, hours_in_trade=(t - fill_t) / 3.6e6)
            break
    else:
        if filled:
            out["result"] = "ещё открыта" if last_close is not None else "нет данных"
            if last_close is not None:
                out["pnl_pct"] = (last_close / entry - 1) * 100
        else:
            out["result"] = "не налило (данные кончились)"
    if filled:
        out["fill_wait_min"] = (fill_t - t0) / MS_MIN
    return out


def summary(df, name):
    rows = []
    for wname, wmin in WAIT.items():
        filled = df[df.fill_wait_min.notna() & (df.fill_wait_min < wmin)]
        closed = filled[filled.result.isin(["тейк", "стоп", "по времени"])]
        tk, st, tm = (closed.result == "тейк").sum(), (closed.result == "стоп").sum(), (closed.result == "по времени").sum()
        rows.append({
            "ожидание лимитки": wname,
            "сигналов": len(df),
            "налило": f"{len(filled)} ({len(filled) / max(len(df), 1) * 100:.0f}%)",
            "тейк": tk, "стоп": st, "по времени": tm,
            "винрейт (тейк / (тейк+стоп))": f"{tk / max(tk + st, 1) * 100:.0f}%",
            "средний результат на сделку": f"{closed.pnl_pct.mean():+.2f}%" if len(closed) else "-",
            "сумма результатов": f"{closed.pnl_pct.sum():+.1f}%",
        })
    return f"### {name}\n\n" + pd.DataFrame(rows).set_index("ожидание лимитки").T.to_markdown() + "\n\n"


def main():
    s = pd.read_csv(IN_FILE)
    s = s[(s.delta_sign == "−") & (s.vol_ratio >= VOL_MIN) & s.poc.notna()].copy()
    s["условие"] = np.where((s.poc_pos <= POC_LOW_MAX) & (s.close_below_poc == "нет"), "да", "нет")
    print(f"сигналов: {len(s)}, из них по условию: {(s['условие'] == 'да').sum()}")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run_one, s.to_dict("records")))
    r = pd.DataFrame(res, index=s.index)
    s = pd.concat([s, r], axis=1)
    keep = ["symbol", "candle_open_kyiv", "условие", "delta_pct", "vol_ratio", "poc", "poc_pos", "close",
            "tick", "entry", "filled_at_close", "fill_wait_min", "result", "pnl_pct", "hours_in_trade"]
    out = s[keep].copy()
    out["fill_wait_h"] = (out.pop("fill_wait_min") / 60).round(1)
    out["hours_in_trade"] = out.hours_in_trade.round(1)
    out["pnl_pct"] = out.pnl_pct.round(2)
    out.to_csv(f"{OUT_DIR}/backtest_trades.csv", index=False)

    be = STOP_PCT / (STOP_PCT + STOP_PCT * RR) * 100
    with open(f"{OUT_DIR}/backtest_summary.md", "w") as f:
        f.write("# Бэктест лонга: вход POC + 5 тиков\n\n")
        f.write(f"Сигналы: объём ≥ x{VOL_MIN:g}, дельта ≤ −17%. Стоп −{STOP_PCT:g}%, тейк +{STOP_PCT * RR:.2f}% (1:{RR:g}).\n")
        f.write(f"Безубыточный винрейт при 1:{RR:g} — {be:.1f}%. Комиссии не учтены.\n")
        f.write("Если в одной минутке задеты и стоп, и тейк — считается стоп.\n\n")
        f.write(summary(s.assign(fill_wait_min=r.fill_wait_min)[s["условие"] == "да"],
                        "Условие: POC в нижних 30% + закрытие выше POC"))
        f.write(summary(s.assign(fill_wait_min=r.fill_wait_min)[s["условие"] == "нет"],
                        "Для сравнения: остальные сигналы x5 с дельтой ≤ −17%"))
    print(open(f"{OUT_DIR}/backtest_summary.md").read())


if __name__ == "__main__":
    main()
