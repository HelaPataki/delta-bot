"""
Нужна ли дельта? Лонг и зеркальный шорт с тремя вариантами условия по дельте.

Вход: results/variants/history_signals.csv — ВСЕ 4h-свечи с объёмом >= x5 (любая дельта),
его делает history.py с DELTA_PCT=0, VOL_MIN=5, OUT_DIR=results/variants.

Для каждой свечи считается POC (минутка с наибольшим объёмом), затем:
  лонг:  POC в нижних 30% + закрытие выше POC; вход POC + 5 тиков; стоп −4,3%, тейк +7,74%
  шорт:  POC в верхних 30% + закрытие ниже POC; вход POC − 5 тиков; стоп +4,3%, тейк −7,74%
Варианты дельты:
  лонг: ≤ −17% / < 0 / любая;   шорт: ≥ +17% / > 0 / любая
Правила сделки — как в backtest.py (условный ордер, срабатывание при касании уровня).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import poc as P
import backtest as B


class NoCache(dict):
    """Не держим минутки в памяти: свечей десятки тысяч."""
    def __setitem__(self, k, v):
        pass


P._cache = NoCache()

IN_FILE = os.environ.get("IN_FILE", "results/variants/history_signals.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results/variants")
THREADS = 16
VOL_BINS = [5, 6, 7, 10, 10 ** 6]
VOL_LABELS = ["x5–6", "x6–7", "x7–10", "x10+"]


def main():
    s = pd.read_csv(IN_FILE)
    s = s[s.vol_ratio >= 5].reset_index(drop=True)
    print(f"свечей с объёмом от x5: {len(s)}")

    with ThreadPoolExecutor(THREADS) as ex:
        pocs = list(ex.map(P.poc_for, s.to_dict("records")))
    p = pd.DataFrame(pocs)
    s["poc"], s["poc_pos"] = p.poc.values, p.poc_pos.values
    s = s[s.poc.notna()].copy()
    s["close_below_poc"] = np.where(s.close < s.poc, "да", "нет")
    s["лонг_условие"] = (s.poc_pos <= 30) & (s.close_below_poc == "нет")
    s["шорт_условие"] = (s.poc_pos >= 70) & (s.close_below_poc == "да")
    print(f"POC посчитан: {len(s)}; под лонг: {s['лонг_условие'].sum()}, под шорт: {s['шорт_условие'].sum()}")

    trades = []
    for side, cond in (("long", "лонг_условие"), ("short", "шорт_условие")):
        B.SIDE = side
        part = s[s[cond]].copy()
        with ThreadPoolExecutor(THREADS) as ex:
            res = list(ex.map(B.run_one, part.to_dict("records")))
        r = pd.DataFrame(res, index=part.index)
        part = pd.concat([part, r], axis=1)
        part["side"] = "лонг" if side == "long" else "шорт"
        trades.append(part)
        print(f"{side}: сделок посчитано {len(part)}")
    t = pd.concat(trades)

    keep = ["side", "symbol", "candle_open_kyiv", "delta_pct", "vol_ratio", "turnover_usd", "change_pct",
            "poc", "poc_pos", "close", "tick", "entry", "fill_wait_min", "result", "pnl_pct", "hours_in_trade"]
    out = t[keep].copy()
    out["fill_wait_h"] = (out.pop("fill_wait_min") / 60).round(1)
    for c in ("poc_pos", "pnl_pct", "hours_in_trade", "delta_pct", "change_pct", "vol_ratio"):
        out[c] = out[c].round(2)
    out.to_csv(f"{OUT_DIR}/variants_trades.csv", index=False)

    be = B.STOP_PCT / (B.STOP_PCT + B.STOP_PCT * B.RR) * 100
    with open(f"{OUT_DIR}/variants_summary.md", "w") as f:
        f.write("# Нужна ли дельта? Лонг и шорт по трём вариантам\n\n")
        f.write(f"Объём ≥ x5, вход POC ± 5 тиков условным ордером, стоп {B.STOP_PCT:g}%, тейк {B.STOP_PCT * B.RR:.2f}% "
                f"(1:{B.RR:g}). Безубыточный винрейт {be:.1f}%. Комиссии не учтены.\n\n")
        variants = {
            "лонг": [("дельта ≤ −17%", t.delta_pct <= -17), ("дельта < 0", t.delta_pct < 0), ("дельта любая", t.delta_pct.notna())],
            "шорт": [("дельта ≥ +17%", t.delta_pct >= 17), ("дельта > 0", t.delta_pct > 0), ("дельта любая", t.delta_pct.notna())],
        }
        for side, vs in variants.items():
            cond_txt = ("POC в нижних 30% + закрытие выше POC, вход POC + 5 тиков" if side == "лонг"
                        else "POC в верхних 30% + закрытие ниже POC, вход POC − 5 тиков")
            f.write(f"## {side.capitalize()}: {cond_txt}\n\n")
            for name, mask in vs:
                g = t[(t.side == side) & mask]
                f.write(B.summary(g, f"{side}, {name}"))
            f.write(f"### {side}: по диапазонам объёма (ожидание 1 свеча 4h)\n\n")
            rows = []
            for name, mask in vs:
                g = t[(t.side == side) & mask & t.fill_wait_min.notna() & (t.fill_wait_min < B.WAIT["1 свеча 4h"])
                      & t.result.isin(["тейк", "стоп", "по времени"])].copy()
                g["объём"] = pd.cut(g.vol_ratio, VOL_BINS, right=False, labels=VOL_LABELS)
                for b, gg in g.groupby("объём", observed=True):
                    tk, st = (gg.result == "тейк").sum(), (gg.result == "стоп").sum()
                    rows.append({"вариант": name, "объём": b, "сделок": len(gg),
                                 "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
                                 "средний результат": f"{gg.pnl_pct.mean():+.2f}%"})
            if rows:
                f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT_DIR}/variants_summary.md").read())


if __name__ == "__main__":
    main()
