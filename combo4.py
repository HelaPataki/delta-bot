"""
Все живые стратегии + шорт «боковик с топами» на одном депозите: $100, 4 общих слота, кто первый пришёл при свободном слоте —
тот в сделке. Считается локально из уже готовых сделок (новых загрузок нет). Живых ботов не трогает.

  ШОРТ (живой): фандинг 24ч ≤ −0,2% + OI 24ч ≥ +60%, 1:1,8 (стоп 4,3%), 24 ч — results/oi_sim2.
  ЛОНГ дельта (живой): x5, дельта ≤ −17%, POC внизу, условный ордер, 1:4, до 10 дней — results/long_sim.
  L3 (живой): карта ликвидаций + толпа в шортах, капа $10M–$2B, тейк 20 / стоп 10, до 7 дней — results/l3_hourly.
  КАЛЕНДАРЬ (живой) — results/cycle_trades:
    листинг: 30-й день, оборот 24ч ≥ $50M, стоп 30 / тейк 15, безубыток после −8%, до 14 дней;
    разблокировка: инсайдеры ≥3%, вход 07:00 Киев за 5 дней, стоп 15 / тейк 10, выход 07:00 на следующий день после.
  БОКОВИК (кандидат): оборот 24ч ≥ $20M + объём боковика < 60% месяца до него + топы L/S ≥ 1,8 и выросли > 0,1;
    шорт на пробое низа боковика — results/range_setup/trades_tops.csv.gz. Два варианта: 7д стоп 8 / тейк 8
    и 14д стоп 10 / тейк 15.
Плечо x2 у всех. Позиция = текущий депозит / 4 × 2. Комиссия учтена, фандинг — где был в тесте.
Если по стратегии в один момент несколько монет — берём случайную; 500 повторов.
"""

import os

import numpy as np
import pandas as pd

SLOTS, START, RUNS, LEV = 4, 100.0, 500, 2
H = 3600 * 1000
RANGE_FILE = "results/range_setup/trades_tops.csv.gz"
STARTS = {"с 2025-01-01": "2025-01-01", "с 2026-01-01": "2026-01-01"}


def groups(df, tcol, exits, nets, syms):
    return [(t, g[exits].to_numpy(), g[nets].to_numpy(), g[syms].to_numpy()) for t, g in df.groupby(tcol)]


def load():
    ev = {}
    s = pd.read_csv("results/oi_sim2/oi_sim_trades.csv.gz")
    s = s[s.G1 & (s.fund24 <= -0.2) & (s.oi_chg24 >= 60) & (s.rr == 1.8) & (s.hold_h == 24)]
    s = s.assign(net=s.pnl_pct + s.funding_pct - 0.1)
    ev["шорт"] = [(g.entry_time.min(), g.exit_time.to_numpy(), g.net.to_numpy(), g.symbol.to_numpy())
                  for _, g in s.groupby("open_time")]
    l = pd.read_csv("results/long_sim/oi_sim_trades.csv.gz")
    l = l[(l.rr == 4.0) & (l.hold_h == 240)]
    l = l.assign(net=l.pnl_pct + l.funding_pct - 0.1)
    ev["лонг"] = [(g.entry_time.min(), g.exit_time.to_numpy(), g.net.to_numpy(), g.symbol.to_numpy())
                  for _, g in l.groupby("open_time")]
    q = pd.read_csv("results/l3_hourly/trades.csv.gz")
    q = q[(q["тейк/стоп"] == "20/10") & q["капа $M"].between(10, 2000)]
    q = q.assign(exit=q["вход"] + q["часов"] * H)
    ev["L3"] = groups(q, "вход", "exit", "pnl %", "монета")
    c = pd.read_csv("results/cycle_trades/trades.csv.gz")
    a = c[(c.стратегия == "листинг") & (c.время == "час листинга") & (c.тейк == 15) & c.безубыток]
    ev["листинг"] = groups(a, "t", "выход_t", "r", "symbol")
    u = c[(c.стратегия == "разблокировка") & (c.время == "07 Киев") & (c.тейк == 10)]
    ev["разлок"] = groups(u, "t", "выход_t", "r", "symbol")
    if os.path.exists(RANGE_FILE):
        r = pd.read_csv(RANGE_FILE)
        r = r[r.вход == "пробой низа"]
        r = r.assign(exit=r.t + (r["дней"] * 24 * H).round())
        for name, (hold, st, tk) in {"боковик 7д 8/8": ("7д", 8, 8), "боковик 14д 10/15": ("14д", 10, 15)}.items():
            ev[name] = groups(r[(r.держать == hold) & (r.стоп == st) & (r.тейк == tk)], "t", "exit", "r", "symbol")
    return ev


def simulate(ev, use, rng, frm, log=False):
    events = sorted((e[0], name, e) for name in use for e in ev[name] if e[0] >= frm)
    eq, peak, mdd, opened = START, START, 0.0, []
    cnt = {n: 0 for n in use}
    pnl_by = {n: 0.0 for n in use}
    monthly, journal, skipped = {}, [], {n: 0 for n in use}

    def settle(upto):
        nonlocal eq, peak, mdd
        for b in sorted(x for x in opened if x[0] <= upto):
            opened.remove(b)
            eq += b[1]
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak * 100)
            m = pd.Timestamp(b[0], unit="ms").strftime("%Y-%m")
            monthly[m] = monthly.get(m, 0) + b[1]

    for t, name, (_, exits, nets, syms) in events:
        settle(t)
        if len(opened) >= SLOTS:
            skipped[name] += 1
            continue
        i = rng.integers(len(nets))
        p = eq / SLOTS * LEV * nets[i] / 100
        opened.append((exits[i], p))
        cnt[name] += 1
        pnl_by[name] += p
        if log:
            journal.append({"вход (Киев)": pd.Timestamp(t, unit="ms", tz="UTC").tz_convert("Europe/Kyiv")
                            .strftime("%Y-%m-%d %H:%M"), "стратегия": name, "монета": syms[i],
                            "результат, % цены": round(nets[i], 2), "$": round(p, 2)})
    settle(float("inf"))
    return eq, mdd, cnt, pnl_by, monthly, journal, skipped


def main():
    ev = load()
    live = ["шорт", "лонг", "L3", "листинг", "разлок"]
    combos = [live]
    rng_names = [n for n in ev if n.startswith("боковик")]
    combos += [live + [n] for n in rng_names] + [[n] for n in rng_names]
    out = open("results/combo4_summary.md", "w")
    out.write("# Все стратегии + шорт «боковик с топами»: $100, 4 общих слота\n\n" + __doc__.split("\n", 2)[2] + "\n")
    if not rng_names:
        out.write("**Нет файла сделок боковика — сначала запусти workflow Range setup.**\n\n")
    for label, day in STARTS.items():
        frm = pd.Timestamp(day, tz="UTC").timestamp() * 1000
        rows, mons = [], {}
        for use in combos:
            res = [simulate(ev, use, np.random.default_rng(i), frm) for i in range(RUNS)]
            fin = np.array([r[0] for r in res])
            dd = np.array([r[1] for r in res])
            cnt = {n: np.median([r[2][n] for r in res]) for n in use}
            pb = {n: np.median([r[3][n] for r in res]) for n in use}
            sk = {n: np.median([r[6][n] for r in res]) for n in use}
            mon = pd.DataFrame([r[4] for r in res]).median()
            key = " + ".join(use)
            mons[key] = mon
            rows.append({"стратегии": key, "сделок": " / ".join(f"{n} {cnt[n]:.0f}" for n in use),
                         "пропущено (слоты заняты)": " / ".join(f"{n} {sk[n]:.0f}" for n in use),
                         "итог: медиана $": np.median(fin), "плохой (5%) $": np.percentile(fin, 5),
                         "хороший (95%) $": np.percentile(fin, 95), "просадка: обычная, %": np.median(dd),
                         "тяжёлая (5%), %": np.percentile(dd, 95),
                         "вклад, $": " / ".join(f"{n} {pb[n]:+.0f}" for n in use),
                         "худший месяц, $": mon.min() if len(mon) else np.nan,
                         "месяцев в минусе": int((mon < 0).sum())})
        df = pd.DataFrame(rows)
        out.write(f"## {label}\n\n" + df.round(1).to_markdown(index=False) + "\n\n")
        out.write("Помесячно (медиана по повторам, $):\n\n" + pd.DataFrame(mons).round(1).to_markdown() + "\n\n")
        print(label)
        print(df.round(1).to_string())
        if label == "с 2026-01-01" and rng_names:
            j = simulate(ev, live + [rng_names[0]], np.random.default_rng(0), frm, log=True)[5]
            pd.DataFrame(j).to_csv("results/combo4_journal.csv", index=False)
    out.close()


if __name__ == "__main__":
    main()
