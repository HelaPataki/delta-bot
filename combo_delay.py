"""
Финальная связка шорт + лонг ($100, 2 общих слота) с другой задержкой входа.
Сравнивает новый прогон (DELAY минут) с прежним (15 минут). Старые файлы только читаются.

Шорт: фандинг ≤ −0,2% + OI 24ч ≥ +60%, 1:1,8, выход через 24 ч.
Лонг: x5, дельта ≤ −17%, POC внизу, условный ордер POC + 5 тиков, 1:4, до 10 дней.
"""

import os

import numpy as np
import pandas as pd

from combo_sim import N_RUNS, START, events, simulate

DELAY = os.environ.get("DELAY", "8")
NEW_DIR = os.environ.get("NEW_DIR", f"results/delay{DELAY}")


def load(short_file, long_file):
    sh = pd.read_csv(short_file)
    sh = sh[sh.G1 & (sh.fund24 <= -0.2) & (sh.oi_chg24 >= 60) & (sh.rr == 1.8) & (sh.hold_h == 24)]
    lg = pd.read_csv(long_file)
    lg = lg[(lg.rr == 4.0) & (lg.hold_h == 240.0)]
    return sh, lg


def per_trade(df):
    net = df.pnl_pct + df.funding_pct - 0.1
    return f"{len(df)} / {(df.result == 'тейк').mean() * 100:.0f}% / {net.mean():+.2f}%"


def main():
    runs = [("через 15 мин (как было)", "results/oi_sim2/oi_sim_trades.csv.gz", "results/long_sim/oi_sim_trades.csv.gz"),
            (f"через {DELAY} мин", f"{NEW_DIR}/short/oi_sim_trades.csv.gz", f"{NEW_DIR}/long/oi_sim_trades.csv.gz")]
    rows = []
    for name, sf, lf in runs:
        sh, lg = load(sf, lf)
        for vname, use_s, use_l in [("только шорт", True, False), ("только лонг", False, True), ("шорт + лонг", True, True)]:
            ev = (events(sh, "шорт") if use_s else []) + (events(lg, "лонг") if use_l else [])
            ev.sort(key=lambda e: e[0])
            row = {"вход": name, "вариант": vname,
                   "шорт: сделок / винрейт / % на сделку": per_trade(sh) if use_s else "",
                   "лонг: сделок / винрейт / % на сделку": per_trade(lg) if use_l else ""}
            for lev in (1, 2):
                res = [simulate(ev, lev, np.random.default_rng(i)) for i in range(N_RUNS)]
                fin, dd = np.array([r[0] for r in res]), np.array([r[1] for r in res])
                row[f"x{lev} средний"] = f"${np.median(fin):,.0f}"
                row[f"x{lev} плохой (5%)"] = f"${np.percentile(fin, 5):,.0f}"
                row[f"x{lev} просадка тяжёлая"] = f"{np.percentile(dd, 5) * 100:.0f}%"
            rows.append(row)
            print(row, flush=True)
    with open(f"{NEW_DIR}/combo_delay_summary.md", "w") as f:
        f.write(f"# Шорт + лонг: вход через {DELAY} мин против 15 мин\n\n"
                "$100, 2 общих слота, одна случайная монета на свечу, комиссия 0,1%, фандинг учтён, "
                f"{N_RUNS} повторов. Шорт — по рынку; лонг — условный ордер на POC + 5 тиков, активен с этой минуты.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(open(f"{NEW_DIR}/combo_delay_summary.md").read())


if __name__ == "__main__":
    main()
