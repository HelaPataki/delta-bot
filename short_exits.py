"""
Шорт: какой выход даёт больше прибыли. Вход как сейчас — по рынку через 8 мин после закрытия сигнальной свечи.
Сигналы — финальный шорт (фандинг ≤ −0,2% + OI ≥ +60% + цена стоит): 2025–2026 и для проверки 2021–2024
(капитализация $10M–$400M). Старые скрипты не меняются.

Варианты (стоп всегда +4,3%, выход по времени через 24 ч, комиссия 0,1%, фандинг учтён):
  тейк 1:1 / 1:1,2 / 1:1,5 / 1:1,8 (как сейчас);
  безубыток: после хода в нашу сторону на 0,5R или 1R стоп переносится на вход (тейк 1:1,8);
  частичный тейк: половина на 1R (4,3%), вторая половина на 1:1,8 — со стопом на прежнем месте или в безубытке;
  фильтр: только сигналы, где OI за последние 4 ч вырос на +10…25% (выход как сейчас).
В одной минуте стоп и тейк — считаем стоп; в первую минуту тейк не засчитывается (как в прежних прогонах).
Итог — таблица по каждому периоду и симуляция $100 (шорт + лонг, 2 слота).
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import short_tests as ST
import combo_sim as CS
from short_mid import funding_series

OUT = "results/short_exits"
H, MIN = 3600 * 1000, 60_000
H4 = 4 * H
DELAY = 8 * MIN
R = 4.3

VARIANTS = {
    "тейк 1:1,8 (как сейчас)": dict(take=1.8),
    "тейк 1:1": dict(take=1.0),
    "тейк 1:1,2": dict(take=1.2),
    "тейк 1:1,5": dict(take=1.5),
    "безубыток после 0,5R": dict(take=1.8, be=0.5),
    "безубыток после 1R": dict(take=1.8, be=1.0),
    "½ на 1R, ½ на 1:1,8": dict(take=1.8, part=1.0),
    "½ на 1R, ½ на 1:1,8, потом безубыток": dict(take=1.8, part=1.0, part_be=True),
}


def exit_trade(rows, entry, take, be=None, part=None, part_be=False):
    """rows — минутки с момента входа (t, o, h, l, c). Возвращает результат шорта."""
    stop0 = entry * (1 + R / 100)
    take_px = entry * (1 - R * take / 100)
    be_px = entry * (1 - R * be / 100) if be else None
    part_px = entry * (1 - R * part / 100) if part else None
    left, pnl, t_part, be_on, first = 1.0, 0.0, None, False, True
    res, t_out, px = None, None, None
    for t, o, h, l, c in rows:
        t_out, px = t + MIN, c
        stop = entry if be_on else stop0
        if h >= stop:
            pnl += left * (entry - stop) / entry * 100
            res, left = ("безубыток" if be_on else "стоп"), 0.0
            break
        if not first:
            if part_px and t_part is None and l <= part_px:
                pnl += 0.5 * (entry - part_px) / entry * 100
                left, t_part = 0.5, t
                if part_be:
                    be_on = True
            if l <= take_px:
                pnl += left * (entry - take_px) / entry * 100
                res, left = "тейк", 0.0
                break
        if be_px and l <= be_px:
            be_on = True
        first = False
    if t_out is None:
        return None
    if left > 0:
        pnl += left * (entry - px) / entry * 100
        res = "по времени"
    return {"result": res, "pnl_pct": pnl, "exit_time": int(t_out), "t_part": t_part}


def one(sig):
    sym, t_open = sig["symbol"], int(sig["open_time"])
    T = t_open + H4
    t_in = T + DELAY
    mins = ST.Mins(sym, t_open)
    rows = list(mins.rows(t_in, t_in + 24 * H))
    if not rows:
        return []
    entry = rows[0][1]
    ft, fr = funding_series(sym)
    out = []
    for name, p in VARIANTS.items():
        r = exit_trade(rows, entry, **p)
        if r is None:
            continue
        tp = r.pop("t_part")
        if tp is None:
            f = fr[(ft > t_in) & (ft <= r["exit_time"])].sum() * 100
        else:
            f = fr[(ft > t_in) & (ft <= tp)].sum() * 100 + 0.5 * fr[(ft > tp) & (ft <= r["exit_time"])].sum() * 100
        out.append({**sig, "variant": name, "entry_time": int(t_in), "entry": entry, "funding_pct": f, **r})
    return out


def signals():
    a = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz")
    a = a[(a.rr == 1.8) & (a.hold_h == 24)][["symbol", "open_time"]].assign(period="2025–26")
    b = pd.read_csv("results/y2021/short_trades_all.csv.gz")
    b = b[b.mcap.between(10e6, 400e6)][["symbol", "open_time"]].assign(period="2021–24")
    s = pd.concat([a, b]).drop_duplicates(["symbol", "open_time"])
    parts = [pd.read_csv(f, usecols=["symbol", "open_time", "oi_chg4"])
             for f in glob.glob("results/oi_funding_2025/parts/*.csv.gz") + glob.glob("results/y2021/short_parts/*.csv.gz")]
    p = pd.concat(parts).drop_duplicates(["symbol", "open_time"])
    return s.merge(p, on=["symbol", "open_time"], how="left")


def sim(sh, lg):
    ev = sorted(CS.events(sh, "шорт") + CS.events(lg, "лонг"), key=lambda e: e[0])
    o = {}
    for lev in (1, 2):
        r = [CS.simulate(ev, lev, np.random.default_rng(i)) for i in range(CS.N_RUNS)]
        fin = np.array([x[0] for x in r])
        o[f"x{lev} средний"] = f"${np.median(fin):,.0f}"
        o[f"x{lev} плохой (5%)"] = f"${np.percentile(fin, 5):,.0f}"
        o[f"x{lev} просадка"] = f"{np.percentile([x[1] for x in r], 5) * 100:.0f}%"
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    sig = signals()
    print("сигналов:", len(sig), flush=True)
    with ThreadPoolExecutor(12) as ex:
        t = pd.DataFrame([x for r in ex.map(one, sig.to_dict("records")) for x in r])
    t.to_csv(f"{OUT}/short_exits_trades.csv.gz", index=False)
    t["net"] = t.pnl_pct + t.funding_pct - 0.1
    lg = {"2025–26": pd.read_csv("results/delay8/long/oi_sim_trades.csv.gz").query("rr == 4 and hold_h == 240"),
          "2021–24": pd.read_csv("results/y2021/long_trades_all.csv.gz").query("mcap >= 10e6 and mcap <= 400e6")}
    out = ["# Шорт: какой выход даёт больше прибыли\n",
           "Вход по рынку через 8 мин, стоп +4,3%, выход через 24 ч, комиссия 0,1%, фандинг учтён. "
           "«В плюсе» — доля сделок с итогом > 0 после комиссии и фандинга.\n"]
    for period in ("2025–26", "2021–24"):
        p = t[t.period == period]
        rows = []
        variants = list(VARIANTS) + ["фильтр: OI за 4 ч +10…25% (выход как сейчас)"]
        for v in variants:
            if v.startswith("фильтр"):
                d = p[(p.variant == "тейк 1:1,8 (как сейчас)") & p.oi_chg4.between(10, 25)]
            else:
                d = p[p.variant == v]
            row = {"вариант": v, "сделок": len(d), "тейк": f"{(d.result == 'тейк').mean() * 100:.0f}%",
                   "в плюсе": f"{(d.net > 0).mean() * 100:.0f}%", "% на сделку": f"{d.net.mean():+.2f}%",
                   "сумма %": f"{d.net.sum():+.0f}%"}
            row.update(sim(d, lg[period]))
            rows.append(row)
            print(period, row, flush=True)
        out.append(f"## {period}\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    open(f"{OUT}/short_exits_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
