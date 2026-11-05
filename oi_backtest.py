"""
Бэктест шорта по сигналам фандинга и открытого интереса (по минуткам). Сигналы — из results/oi_funding/parts.

Группы (всё на момент закрытия 4h-свечи, только первая свеча серии):
  G1  фандинг <= −0,03% + OI за 24 ч >= +20% + цена за 24 ч >= −2%   («перегретые шорты» — оказалось, падают)
  G1b то же с фандингом <= −0,1%
  G2  фандинг >= +0,05% + OI за 24 ч >= +20% + OI за 4 ч >= +5% + цена за 24 ч <= +2%  («перегретые лонги»)
  G3  только OI за 24 ч >= +20%
Входы в шорт:
  E1 — по цене закрытия сигнальной свечи сразу после алерта;
  E2 — условный ордер на максимуме свечи + 1% (ждём 1 свечу 4h / 24 часа).
Стоп +4,3%, тейк −7,74% (1:1,8). Если стоп и тейк в одной минутке — стоп. Держим до 10 дней. Комиссии не учтены.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import short_tests as ST
from oi_funding import first_only

OUT_DIR = os.environ.get("OUT_DIR", "results/oi_backtest")
H4, DAY = 4 * 3600 * 1000, 24 * 3600 * 1000
STOP, TAKE = 4.3, 7.74
ST.RR = TAKE / STOP
stop_of = lambda e: e * (1 + STOP / 100)
take_of = lambda e: e * (1 - TAKE / 100)


def run(r):
    t_open = int(r["open_time"])
    t0 = t_open + H4
    mins = ST.Mins(r["symbol"], t_open)
    sig = list(mins.rows(t_open, t0))
    out = {}
    if not sig:
        return out
    hi = max(x[2] for x in sig)
    res, R, _ = ST.trade(mins, t0, r["close"], None, 0, stop_of, take_of, immediate=True)
    out["E1|—|result"], out["E1|—|R"] = res, R
    for wn, w in (("1 свеча 4h", H4), ("24 часа", DAY)):
        res, R, wh = ST.trade(mins, t0, r["close"], hi * 1.01, w, stop_of, take_of)
        out[f"E2|{wn}|result"], out[f"E2|{wn}|R"] = res, R
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("results/oi_funding/parts/*.csv.gz"))])
    a = a.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    a = a[a.fund24.notna() & a.oi_chg24.notna() & a.px24.notna()].reset_index(drop=True)
    groups = {
        "G1: фандинг ≤ −0,03% + OI 24ч ≥ +20% + цена стоит": first_only(a, (a.fund24 <= -0.03) & (a.oi_chg24 >= 20) & (a.px24 >= -2)),
        "G1b: то же, фандинг ≤ −0,1%": first_only(a, (a.fund24 <= -0.1) & (a.oi_chg24 >= 20) & (a.px24 >= -2)),
        "G2: фандинг ≥ +0,05% + OI 24ч ≥ +20% + OI 4ч ≥ +5% + цена стоит":
            first_only(a, (a.fund24 >= 0.05) & (a.oi_chg24 >= 20) & (a.oi_chg4 >= 5) & (a.px24 <= 2)),
        "G3: только OI 24ч ≥ +20%": first_only(a, a.oi_chg24 >= 20),
    }
    any_sig = np.zeros(len(a), bool)
    for m in groups.values():
        any_sig |= m.to_numpy()
    s = a[any_sig].reset_index(drop=True)
    print(f"сигналов к расчёту: {len(s)}")
    with ThreadPoolExecutor(16) as ex:
        res = pd.DataFrame(list(ex.map(run, s.to_dict("records"))))
    s = pd.concat([s, res], axis=1)
    for name, m in groups.items():
        s[name[:3].strip(":")] = m.to_numpy()[any_sig]
    s.to_csv(f"{OUT_DIR}/oi_backtest_trades.csv.gz", index=False)

    months = (a.open_time.max() - a.open_time.min()) / 86_400_000 / 30.4
    rows = []
    for name in groups:
        code = name[:3].strip(":")
        g = s[s[code]]
        for e, wn in (("E1", "—"), ("E2", "1 свеча 4h"), ("E2", "24 часа")):
            r_, R = g[f"{e}|{wn}|result"], g[f"{e}|{wn}|R"]
            closed = r_.isin(["тейк", "стоп", "по времени"])
            tk, st = (r_ == "тейк").sum(), (r_ == "стоп").sum()
            rows.append({"группа": name, "вход": "по закрытию" if e == "E1" else "максимум + 1%", "ожидание": wn,
                         "сигналов": len(g), "сделок": int(closed.sum()), "в месяц": f"{closed.sum() / months:.1f}",
                         "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
                         "в среднем R": f"{R[closed].mean():+.2f}" if closed.any() else "-",
                         "в среднем %": f"{R[closed].mean() * STOP:+.2f}%" if closed.any() else "-"})
    with open(f"{OUT_DIR}/oi_backtest_summary.md", "w") as f:
        f.write("# Бэктест шорта: фандинг + открытый интерес (с 01.01.2026)\n\n")
        f.write(f"Стоп +{STOP:g}%, тейк −{TAKE:g}% (1:{TAKE / STOP:.1f}). Безубыточный винрейт {STOP / (STOP + TAKE) * 100:.1f}%, "
                "безубыточно = 0R. По минуткам, комиссии не учтены. Сигналы последних 10 дней могут быть не завершены.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(open(f"{OUT_DIR}/oi_backtest_summary.md").read())


if __name__ == "__main__":
    main()
