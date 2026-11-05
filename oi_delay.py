"""
Шорт по фандингу и открытому интересу: что будет, если алерт приходит с задержкой. По минуткам.

Сигналы — G1 и G1b из results/oi_funding/parts (с 2026-01-01), как в oi_backtest.py.
Входы:
  «на закрытии» — по цене закрытия сигнальной свечи (как в прошлом бэктесте);
  «рынок +D мин» — по цене открытия минутки через D минут после закрытия свечи (D = 5 / 15 / 30);
  «ордер на закрытие +D мин» — через D минут ставится условный ордер на цене закрытия свечи; срабатывает,
      когда цена до неё дошла (ждём до 4 часов).
Стоп +4,3% от входа, тейк = 4,3% x соотношение (1:1,8 и 1:2,5). Если стоп и тейк в одной минутке — стоп.
Держим до 10 дней. Комиссии не учтены.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import short_tests as ST
from oi_funding import first_only

IN_DIR = os.environ.get("IN_DIR", "results/oi_funding/parts")
OUT_DIR = os.environ.get("OUT_DIR", "results/oi_delay")
H4, DAY, MIN = 4 * 3600 * 1000, 24 * 3600 * 1000, 60_000
STOP = 4.3
RRS = [1.8, 2.5]
DELAYS = [5, 15, 30]


def trade_from(mins, t_start, entry, rr):
    """Шорт с фиксированной ценой входа entry, начиная с минутки t_start."""
    stop, take = entry * (1 + STOP / 100), entry * (1 - STOP * rr / 100)
    first, last_c = True, None
    for t, o, h, l, c in mins.rows(t_start, t_start + 10 * DAY):
        last_c = c
        if h >= stop:
            return "стоп", -1.0
        if not first and l <= take:
            return "тейк", rr
        first = False
    if last_c is None:
        return "нет данных", np.nan
    return "по времени", (entry - last_c) / (stop - entry)


def run(r):
    t_open = int(r["open_time"])
    t0 = t_open + H4
    mins = ST.Mins(r["symbol"], t_open)
    if not list(mins.rows(t_open, t0)):
        return {}
    out = {}
    close = r["close"]
    for rr in RRS:
        out[f"на закрытии|{rr:g}"] = trade_from(mins, t0, close, rr)[1]
    for d in DELAYS:
        t_d = t0 + d * MIN
        m = list(mins.rows(t_d, t_d + MIN))
        if not m:
            continue
        px = m[0][1]                                            # открытие минутки через D минут
        out[f"сдвиг цены через {d} мин, %"] = (px / close - 1) * 100
        for rr in RRS:
            out[f"рынок +{d} мин|{rr:g}"] = trade_from(mins, t_d, px, rr)[1]
        # условный ордер на цене закрытия, выставленный через D минут
        prev, fill = px, None
        for t, o, h, l, c in mins.rows(t_d, t_d + H4):
            if (l <= close <= h) or (prev < close < l) or (h < close < prev):
                fill = t
                break
            prev = c
        for rr in RRS:
            out[f"ордер на закрытие +{d} мин|{rr:g}"] = trade_from(mins, fill, close, rr)[1] if fill else np.nan
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{IN_DIR}/*.csv.gz"))])
    a = a.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    a = a[a.fund24.notna() & a.oi_chg24.notna() & a.px24.notna()].reset_index(drop=True)
    g1 = first_only(a, (a.fund24 <= -0.03) & (a.oi_chg24 >= 20) & (a.px24 >= -2))
    g1b = first_only(a, (a.fund24 <= -0.1) & (a.oi_chg24 >= 20) & (a.px24 >= -2))
    s = a[g1 | g1b].reset_index(drop=True)
    s["G1"], s["G1b"] = g1[g1 | g1b].to_numpy(), g1b[g1 | g1b].to_numpy()
    print(f"сигналов: {len(s)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = pd.DataFrame(list(ex.map(run, s.to_dict("records"))))
    s = pd.concat([s, res], axis=1)
    s.to_csv(f"{OUT_DIR}/oi_delay_trades.csv.gz", index=False)

    entries = ["на закрытии"] + [f"рынок +{d} мин" for d in DELAYS] + [f"ордер на закрытие +{d} мин" for d in DELAYS]
    rows = []
    for grp in ("G1", "G1b"):
        g = s[s[grp]]
        for rr in RRS:
            for e in entries:
                R = g[f"{e}|{rr:g}"].dropna()
                rows.append({"группа": grp, "соотношение": f"1:{rr:g}", "вход": e, "сделок": len(R),
                             "налило": f"{len(R) / len(g) * 100:.0f}%",
                             "винрейт": f"{(R >= rr - 1e-9).sum() / max(((R >= rr - 1e-9) | (R <= -1 + 1e-9)).sum(), 1) * 100:.0f}%",
                             "в среднем R": f"{R.mean():+.2f}", "в среднем %": f"{R.mean() * STOP:+.2f}%"})
    shifts = {d: s[f"сдвиг цены через {d} мин, %"] for d in DELAYS}
    with open(f"{OUT_DIR}/oi_delay_summary.md", "w") as f:
        f.write("# Шорт по фандингу и открытому интересу: вход с задержкой после алерта (с 2026)\n\n")
        f.write("G1: фандинг ≤ −0,03% + OI за 24 ч ≥ +20% + цена стоит. G1b: то же, фандинг ≤ −0,1%. "
                f"Стоп +{STOP:g}%, тейк = стоп × соотношение. По минуткам, комиссии не учтены.\n\n")
        f.write("Насколько цена успевает уйти от закрытия свечи (для шорта минус — цена ушла вниз, вход хуже):\n\n")
        for d, x in shifts.items():
            f.write(f"- через {d} мин: медиана {x.median():+.2f}%, у половины сигналов от {x.quantile(.25):+.2f}% до {x.quantile(.75):+.2f}%\n")
        f.write("\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(open(f"{OUT_DIR}/oi_delay_summary.md").read())


if __name__ == "__main__":
    main()
