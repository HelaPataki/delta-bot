"""
Совместная симуляция шорта и лонга на одном депозите: $100, 2 общих слота.

Шорт: фильтр (фандинг <= −0,2% + OI 24ч >= +60%), из results/oi_sim2 (вход по рынку через 15 мин).
Лонг: x5, дельта <= −17%, POC внизу, из results/long_sim (условный ордер на POC + 5 тиков с 15-й минуты).
События идут по времени входа; если есть свободный слот — сделка открывается (одна монета на сигнальную свечу
каждой стратегии, случайная). Позиция = баланс слота x плечо. Комиссия 0,1%, фандинг учтён.
"""

import os

import numpy as np
import pandas as pd

OUT_DIR = os.environ.get("OUT_DIR", "results/combo_sim")
N_RUNS = int(os.environ.get("N_RUNS", "1000"))
FEE, SLOTS, START = 0.1, 2, 100.0


def events(df, side):
    ev = []
    for t_open, c in df.groupby("open_time", sort=True):
        c = c.sort_values("entry_time")
        ev.append((c.entry_time.iloc[0], side, c.exit_time.to_numpy(), (c.pnl_pct + c.funding_pct).to_numpy(), c))
    return ev


def simulate(ev, lev, rng, log=False):
    bal, busy = np.full(SLOTS, START / SLOTS), np.zeros(SLOTS)
    peak, dd, cnt, journal = START, 0.0, {"шорт": 0, "лонг": 0}, []
    for t_in, side, exits, net, c in ev:
        free = np.where(busy <= t_in)[0]
        if len(free) == 0:
            continue
        i, k = rng.integers(len(exits)), free[0]
        before = bal[k]
        bal[k] = max(bal[k] * (1 + lev * (net[i] - FEE) / 100), 0.0)
        busy[k] = exits[i]
        cnt[side] += 1
        eq = bal.sum()
        peak, dd = max(peak, eq), min(dd, eq / max(peak, eq) - 1)
        if log:
            tr = c.iloc[i]
            journal.append({"вход (Киев)": pd.to_datetime(t_in, unit="ms", utc=True).tz_convert("Europe/Kyiv").strftime("%Y-%m-%d %H:%M"),
                            "сторона": side, "монета": tr.symbol, "слот": k + 1, "результат": tr.result,
                            "цена %": round(tr.pnl_pct, 2), "фандинг %": round(tr.funding_pct, 2),
                            "слот до $": round(before, 2), "слот после $": round(bal[k], 2), "всего $": round(eq, 2)})
    return bal.sum(), dd, cnt, journal


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sh = pd.read_csv("results/oi_sim2/oi_sim_trades.csv.gz")
    sh = sh[sh.G1 & (sh.fund24 <= -0.2) & (sh.oi_chg24 >= 60) & (sh.rr == 1.8) & (sh.hold_h == 24)]
    lg_all = pd.read_csv("results/long_sim/oi_sim_trades.csv.gz")
    start = max(sh.open_time.min(), lg_all.open_time.min())
    rows = []
    variants = [("только шорт 1:1,8 / 24 ч", sh, None),
                ("только лонг 1:4 / 10 дн.", None, (4.0, 240.0)),
                ("шорт + лонг 1:4 / 10 дн.", sh, (4.0, 240.0)),
                ("шорт + лонг 1:1,8 / 10 дн.", sh, (1.8, 240.0))]
    for name, s_df, lcfg in variants:
        ev = []
        if s_df is not None:
            ev += events(s_df, "шорт")
        if lcfg:
            lg = lg_all[(lg_all.rr == lcfg[0]) & (lg_all.hold_h == lcfg[1])]
            ev += events(lg, "лонг")
        ev.sort(key=lambda e: e[0])
        for lev in (1, 2):
            res = [simulate(ev, lev, np.random.default_rng(i)) for i in range(N_RUNS)]
            fin = np.array([r[0] for r in res]); dd = np.array([r[1] for r in res])
            ns = np.median([r[2]["шорт"] for r in res]); nl = np.median([r[2]["лонг"] for r in res])
            rows.append({"вариант": name, "плечо": f"x{lev}", "сделок шорт / лонг": f"{ns:.0f} / {nl:.0f}",
                         "итог: средний": f"${np.median(fin):,.0f}", "плохой (5%)": f"${np.percentile(fin, 5):,.0f}",
                         "хороший (95%)": f"${np.percentile(fin, 95):,.0f}", "в минусе": f"{(fin < START).mean() * 100:.0f}%",
                         "просадка: обычная": f"{np.median(dd) * 100:.0f}%", "тяжёлая (5%)": f"{np.percentile(dd, 5) * 100:.0f}%"})
            if name == "шорт + лонг 1:4 / 10 дн." and lev == 1:
                pd.DataFrame(simulate(ev, 1, np.random.default_rng(0), log=True)[3]).to_csv(
                    f"{OUT_DIR}/combo_example_x1.csv", index=False)
            print(rows[-1], flush=True)
    with open(f"{OUT_DIR}/combo_sim_summary.md", "w") as f:
        f.write("# Шорт + лонг на одном депозите: $100, 2 общих слота\n\n")
        f.write("Шорт: фандинг ≤ −0,2% + OI 24ч ≥ +60%, вход по рынку через 15 мин, стоп +4,3%, тейк −7,74%, выход через 24 ч. "
                "Лонг: x5, дельта ≤ −17%, POC внизу, условный ордер на POC + 5 тиков, стоп −4,3%. "
                f"Кто первый пришёл при свободном слоте — тот и в сделке. Комиссия {FEE:g}%, фандинг учтён, {N_RUNS} повторов.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\nПример одного прогона (шорт + лонг 1:4, без плеча) — combo_example_x1.csv\n")
    print(open(f"{OUT_DIR}/combo_sim_summary.md").read())


if __name__ == "__main__":
    main()
