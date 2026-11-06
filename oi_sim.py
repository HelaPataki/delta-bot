"""
Симуляция торговли шортом по фандингу и открытому интересу: $100, 2 слота, одна случайная монета на сигнальную свечу.

Правила:
  депозит делится на 2 слота по $50; у каждого слота свой баланс (прибыль/убыток остаются в слоте);
  на каждой сигнальной 4h-свече, если есть свободный слот, берём ОДНУ случайную монету из сигналов этой свечи;
  вход по рынку через 15 минут после закрытия свечи (из oi_sim_data.py), позиция = баланс слота x плечо;
  выход по стопу / тейку / через 10 дней; слот свободен с момента выхода;
  комиссия 0,1% от позиции за сделку (вход + выход по рынку), фандинг за время сделки — от позиции.
Повторяется N_RUNS раз с разным случайным выбором монет.
"""

import os

import numpy as np
import pandas as pd

IN_FILE = os.environ.get("IN_FILE", "results/oi_sim/oi_sim_trades.csv.gz")
OUT_DIR = os.environ.get("OUT_DIR", "results/oi_sim")
N_RUNS = int(os.environ.get("N_RUNS", "1000"))
FEE = 0.1
SLOTS, START = 2, 100.0


def prepare(trades):
    """Сигнальные свечи по времени: для каждой — время входа и массивы сделок-кандидатов."""
    groups = []
    for t_open, c in trades.groupby("open_time", sort=True):
        groups.append((t_open, c.entry_time.iloc[0], c.exit_time.to_numpy(),
                       (c.pnl_pct + c.funding_pct).to_numpy(), c))
    return groups


def simulate(groups, lev, rng, log=False):
    """Возвращает итог, макс. просадку, число сделок, журнал."""
    bal = np.full(SLOTS, START / SLOTS)
    busy = np.zeros(SLOTS)
    equity_peak, max_dd, n, journal = START, 0.0, 0, []
    for t_open, t_in, exits, net, cands in groups:
        free = np.where(busy <= t_in)[0]
        if len(free) == 0:
            continue
        i = rng.integers(len(exits))
        k = free[0]
        before = bal[k]
        bal[k] = max(bal[k] * (1 + lev * (net[i] - FEE) / 100), 0.0)
        busy[k] = exits[i]
        n += 1
        eq = bal.sum()
        equity_peak = max(equity_peak, eq)
        max_dd = min(max_dd, eq / equity_peak - 1)
        if log:
            tr = cands.iloc[i]
            journal.append({"сигнал (Киев)": pd.to_datetime(t_open + 4 * 3600 * 1000, unit="ms", utc=True)
                            .tz_convert("Europe/Kyiv").strftime("%Y-%m-%d %H:%M"),
                            "монета": tr.symbol, "из сигналов": len(exits), "слот": k + 1, "результат": tr.result,
                            "цена %": round(tr.pnl_pct, 2), "фандинг %": round(tr.funding_pct, 2),
                            "слот до $": round(before, 2), "слот после $": round(bal[k], 2), "всего $": round(eq, 2)})
    return bal.sum(), max_dd, n, journal


def main():
    t = pd.read_csv(IN_FILE)
    if "strategy" in t:
        strategies = {name: t[t.strategy == name] for name in t.strategy.unique()}
    else:
        strategies = {
            "G1b (фандинг ≤ −0,1% + OI 24ч ≥ +20%)": t[t.G1b],
            "Фильтр (фандинг ≤ −0,2% + OI 24ч ≥ +60%)": t[t.G1 & (t.fund24 <= -0.2) & (t.oi_chg24 >= 60)],
        }
        if os.environ.get("ONLY_FILTER"):
            strategies.pop("G1b (фандинг ≤ −0,1% + OI 24ч ≥ +20%)")
    rows, examples = [], []
    for sname, st in strategies.items():
        if "hold_h" not in t:
            t["hold_h"] = 240.0
        for rr, hold in sorted(set(zip(t.rr, t.hold_h))):
            tr = prepare(st[(st.rr == rr) & (st.hold_h == hold)])
            for lev in (1, 2):
                res = [simulate(tr, lev, np.random.default_rng(i)) for i in range(N_RUNS)]
                fin = np.array([r[0] for r in res]); dd = np.array([r[1] for r in res]); n = np.array([r[2] for r in res])
                rows.append({"стратегия": sname, "соотношение": f"1:{rr:g}",
                             "макс. в сделке": "10 дней" if hold >= 240 else f"{hold:g} ч", "плечо": f"x{lev}",
                             "сделок за период": int(np.median(n)),
                             "итог: средний": f"${np.median(fin):,.0f}", "плохой (5%)": f"${np.percentile(fin, 5):,.0f}",
                             "хороший (95%)": f"${np.percentile(fin, 95):,.0f}",
                             "в минусе": f"{(fin < START).mean() * 100:.0f}%",
                             "просадка: обычная": f"{np.median(dd) * 100:.0f}%",
                             "тяжёлая (5%)": f"{np.percentile(dd, 5) * 100:.0f}%"})
                if lev == 1 and rr == 3 and hold >= 240:
                    _, _, _, j = simulate(tr, lev, np.random.default_rng(0), log=True)
                    ex = pd.DataFrame(j)
                    ex.insert(0, "стратегия", sname)
                    examples.append(ex)
                print(sname, rr, lev, "готово", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if examples:
        pd.concat(examples).to_csv(f"{OUT_DIR}/oi_sim_example_1to3_x1.csv", index=False)
    span = (t.open_time.max() - t.open_time.min()) / 86_400_000
    with open(f"{OUT_DIR}/oi_sim_summary.md", "w") as f:
        f.write("# Симуляция: $100, 2 слота, одна случайная монета на сигнальную свечу\n\n")
        f.write(f"Период: {pd.to_datetime(t.open_time.min(), unit='ms'):%Y-%m-%d} – {pd.to_datetime(t.open_time.max(), unit='ms'):%Y-%m-%d} "
                f"(~{span / 30.4:.0f} мес.). Вход через 15 мин после закрытия свечи (шорт — по рынку, лонг — условный ордер на POC + 5 тиков), стоп 4,3%. "
                f"Комиссия {FEE:g}% от позиции, фандинг за время сделки учтён. {N_RUNS} повторов со случайным выбором монет.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
        if examples:
            f.write("Пример одного прогона (1:3, без плеча) — в файле oi_sim_example_1to3_x1.csv.\n")
    print(open(f"{OUT_DIR}/oi_sim_summary.md").read())


if __name__ == "__main__":
    main()
