"""
Бэктест лонга по минутным свечам Binance.

Сигналы: results/poc_signals.csv, объём >= x5, дельта <= -17%.
  Условие («гипотеза»): POC в нижних 30% свечи и закрытие выше POC.
  Остальные x5 с дельтой <= -17% считаются для сравнения по тем же правилам.

Сделка:
  вход   — условный ордер на POC + 5 тиков (лонг) / POC − 5 тиков (шорт), выставляется после закрытия
           сигнальной свечи и срабатывает, когда цена первый раз дошла до уровня с любой стороны;
           вход всегда по цене уровня;
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
VOL_MIN = float(os.environ.get("VOL_MIN", "3"))
VOL_GROUPS = [3, 5, 10]
POC_LOW_MAX = 30.0
SIDE = os.environ.get("SIDE", "long")          # long или short
ENTRY_MODE = os.environ.get("ENTRY_MODE", "poc")   # poc = от POC; extreme = от максимума (шорт) / минимума (лонг) свечи
EXT_TICKS = int(os.environ.get("EXT_TICKS", "3"))
EXT_PCT = float(os.environ.get("EXT_PCT", "0"))   # сдвиг входа за экстремум свечи в %, вместо тиков
SFX = ("" if SIDE == "long" else "_short") + (
    (f"_extreme_plus{EXT_PCT:g}pct" if EXT_PCT else ("_extreme" if EXT_TICKS == 3 else f"_extreme{EXT_TICKS}"))
    if ENTRY_MODE == "extreme" else "")
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
           "pnl_pct": np.nan, "hours_in_trade": np.nan}
    open_utc = kyiv_to_utc(row["candle_open_kyiv"])
    sig_day = day_minutes(row["symbol"], open_utc.date())
    if sig_day is None:
        return out
    tick = infer_tick(sig_day)
    out["tick"] = tick
    sh = SIDE == "short"
    tp_pct = STOP_PCT * RR

    def levels(e):
        if sh:
            return e * (1 + STOP_PCT / 100), e * (1 - tp_pct / 100)
        return e * (1 - STOP_PCT / 100), e * (1 + tp_pct / 100)

    def pnl(e, price):
        return (e / price - 1) * 100 if sh else (price / e - 1) * 100

    if ENTRY_MODE == "extreme":
        t_open = open_utc.timestamp() * 1000
        w = sig_day[(sig_day[:, 0] >= t_open) & (sig_day[:, 0] < t_open + 4 * 3600 * 1000)]
        if len(w) == 0:
            return out
        if EXT_PCT:
            entry = w[:, 2].max() * (1 + EXT_PCT / 100) if sh else w[:, 3].min() * (1 - EXT_PCT / 100)
        else:
            entry = w[:, 2].max() - EXT_TICKS * tick if sh else w[:, 3].min() + EXT_TICKS * tick
    else:
        entry = row["poc"] - TICKS * tick if sh else row["poc"] + TICKS * tick
    stop, take = levels(entry)
    t0 = open_utc.timestamp() * 1000 + 4 * 3600 * 1000   # закрытие сигнальной свечи

    filled, fill_t = False, None
    out["entry"] = entry
    last_close = None
    prev_c = row["close"]                                   # цена на закрытии сигнальной свечи
    for t, o, h, l, c in minute_stream(row["symbol"], t0):
        last_close = c
        hit_stop = h >= stop if sh else l <= stop
        hit_take = l <= take if sh else h >= take
        if not filled:
            if t - t0 >= WAIT["10 дней"] * MS_MIN:
                out["result"] = "не налило"
                return out
            # условный ордер срабатывает, когда цена дошла до уровня с любой стороны
            touched = (l <= entry <= h) or (prev_c < entry < l) or (h < entry < prev_c)
            prev_c = c
            if touched:
                filled, fill_t = True, t
                if hit_stop:
                    out.update(result="стоп", pnl_pct=-STOP_PCT, hours_in_trade=0)
                    break
            continue
        if t - fill_t >= HOLD_MIN * MS_MIN:
            out.update(result="по времени", pnl_pct=pnl(entry, c), hours_in_trade=(t - fill_t) / 3.6e6)
            break
        if hit_stop:
            out.update(result="стоп", pnl_pct=-STOP_PCT, hours_in_trade=(t - fill_t) / 3.6e6)
            break
        if hit_take:
            out.update(result="тейк", pnl_pct=tp_pct, hours_in_trade=(t - fill_t) / 3.6e6)
            break
    else:
        if filled:
            out["result"] = "ещё открыта" if last_close is not None else "нет данных"
            if last_close is not None:
                out["pnl_pct"] = pnl(entry, last_close)
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
    if SIDE == "short":
        s = s[(s.delta_sign == "+") & (s.vol_ratio >= VOL_MIN) & s.poc.notna()].copy()
        s["условие"] = np.where((s.poc_pos >= 100 - POC_LOW_MAX) & (s.close_below_poc == "да"), "да", "нет")
        cond_name, sig_name = "POC в верхних 30% + закрытие ниже POC", "дельта ≥ +17%"
        entry_name = ((f"шорт, вход максимум свечи + {EXT_PCT:g}%" if EXT_PCT else
                       (f"шорт, вход максимум свечи − {EXT_TICKS} тика" if EXT_TICKS else "шорт, вход на максимуме свечи"))
                      if ENTRY_MODE == "extreme"
                      else f"шорт, вход POC − {TICKS} тиков")
        stop_txt, take_txt = f"+{STOP_PCT:g}%", f"−{STOP_PCT * RR:.2f}%"
    else:
        s = s[(s.delta_sign == "−") & (s.vol_ratio >= VOL_MIN) & s.poc.notna()].copy()
        s["условие"] = np.where((s.poc_pos <= POC_LOW_MAX) & (s.close_below_poc == "нет"), "да", "нет")
        cond_name, sig_name = "POC в нижних 30% + закрытие выше POC", "дельта ≤ −17%"
        entry_name = (f"лонг, вход минимум свечи + {EXT_TICKS} тика" if ENTRY_MODE == "extreme"
                      else f"лонг, вход POC + {TICKS} тиков")
        stop_txt, take_txt = f"−{STOP_PCT:g}%", f"+{STOP_PCT * RR:.2f}%"
    print(f"сигналов: {len(s)}, из них по условию: {(s['условие'] == 'да').sum()}")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run_one, s.to_dict("records")))
    r = pd.DataFrame(res, index=s.index)
    s = pd.concat([s, r], axis=1)
    keep = ["symbol", "candle_open_kyiv", "условие", "delta_pct", "vol_ratio", "poc", "poc_pos", "close",
            "tick", "entry", "fill_wait_min", "result", "pnl_pct", "hours_in_trade"]
    out = s[keep].copy()
    out["fill_wait_h"] = (out.pop("fill_wait_min") / 60).round(1)
    out["hours_in_trade"] = out.hours_in_trade.round(1)
    out["pnl_pct"] = out.pnl_pct.round(2)
    out.to_csv(f"{OUT_DIR}/backtest_trades{SFX}.csv", index=False)

    be = STOP_PCT / (STOP_PCT + STOP_PCT * RR) * 100
    with open(f"{OUT_DIR}/backtest_summary{SFX}.md", "w") as f:
        f.write(f"# Бэктест: {entry_name}\n\n")
        f.write(f"Сигналы: {sig_name}, объём ≥ x3 / x5 / x10 (отдельные таблицы). Стоп {stop_txt}, тейк {take_txt} (1:{RR:g}).\n")
        f.write(f"Безубыточный винрейт при 1:{RR:g} — {be:.1f}%. Комиссии не учтены.\n")
        f.write("Если в одной минутке задеты и стоп, и тейк — считается стоп.\n\n")
        for m in VOL_GROUPS:
            if m < VOL_MIN:
                continue
            g = s[s.vol_ratio >= m]
            f.write(f"## Объём ≥ x{m}\n\n")
            f.write(summary(g[g["условие"] == "да"], f"x{m}. Условие: {cond_name}"))
            f.write(summary(g[g["условие"] == "нет"], f"x{m}. Для сравнения: остальные сигналы, {sig_name}"))
        f.write("## По диапазонам объёма (условие, ожидание 1 свеча 4h)\n\n")
        c = s[(s["условие"] == "да") & s.fill_wait_min.notna() & (s.fill_wait_min < WAIT["1 свеча 4h"])
              & s.result.isin(["тейк", "стоп", "по времени"])].copy()
        c["объём"] = pd.cut(c.vol_ratio, [3, 4, 5, 6, 7, 10, 10 ** 6], right=False,
                            labels=["x3–4", "x4–5", "x5–6", "x6–7", "x7–10", "x10+"])
        rows = []
        for b, g in c.groupby("объём", observed=True):
            tk, st = (g.result == "тейк").sum(), (g.result == "стоп").sum()
            rows.append({"объём": b, "сделок": len(g), "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
                         "средний результат": f"{g.pnl_pct.mean():+.2f}%"})
        if rows:
            f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(open(f"{OUT_DIR}/backtest_summary{SFX}.md").read())


if __name__ == "__main__":
    main()
