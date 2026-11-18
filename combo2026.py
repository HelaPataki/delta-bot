"""
Все стратегии на одном депозите с 2026-01-01: $100, 3 общих слота, кто первый пришёл при свободном слоте — тот в сделке.
Считается локально из уже готовых сделок (новых загрузок нет). Живых ботов не трогает.

  ШОРТ (живой бот): фандинг 24ч ≤ −0,2% + OI 24ч ≥ +60%, вход по рынку через 15 мин, 1:1,8 (стоп 4,3%), 24 ч — results/oi_sim2.
  ЛОНГ дельта (живой бот): x5, дельта ≤ −17%, POC внизу, условный ордер POC + 5 тиков, 1:4, до 10 дней — results/long_sim.
  L3 (живой бот): карта ликвидаций + толпа в шортах, капа $10M–$2B, тейк 20 / стоп 10, до 7 дней — results/l3_hourly.
  ЛИСТИНГ (новый): шорт через 30 дней после листинга, оборот 24ч ≥ $50M, стоп 30 / тейк 20, до 30 дней — results/listing_sim
    (основной вариант и «стоп с проскальзыванием 3%»).
Плечо: шорт, лонг, L3 — x2; листинг — x1. Позиция = текущий депозит / 3 × плечо. Комиссия учтена, фандинг — где был в тесте
(шорт, лонг, листинг). Если по стратегии в один момент несколько монет — берём случайную; 500 повторов.
Данные заканчиваются там, где заканчиваются исходные прогоны (конец октября — начало ноября 2026).
"""

import numpy as np
import pandas as pd

FROM = pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000
SLOTS, START, RUNS = 3, 100.0, 500
H = 3600 * 1000
LEV = {"шорт": 2, "лонг": 2, "L3": 2, "листинг": 1}


def load(slip):
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
    ev["L3"] = [(t, (g["вход"] + g["часов"] * H).to_numpy(), g["pnl %"].to_numpy(), g["монета"].to_numpy())
                for t, g in q.groupby("вход")]
    a = pd.read_csv("results/listing_sim/trades.csv.gz")
    a = a[(a["вход через, д"] == 30) & (a["оборот 24ч до входа $"] >= 50e6)].dropna(subset=["стоп30/тейк20"])
    col = "стоп30/тейк20 проск." if slip else "стоп30/тейк20"
    ev["листинг"] = [(t, (g.t + g["стоп30/тейк20 часов"] * H).to_numpy(), g[col].to_numpy(), g.symbol.to_numpy())
                     for t, g in a.groupby("t")]
    return ev


def simulate(ev, use, rng, log=False):
    events = sorted((e[0], name, e) for name in use for e in ev[name] if e[0] >= FROM)
    eq, peak, mdd, opened = START, START, 0.0, []
    cnt = {n: 0 for n in use}
    pnl_by = {n: 0.0 for n in use}
    journal = []
    monthly = {}

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
            continue
        i = rng.integers(len(nets))
        p = eq / SLOTS * LEV[name] * nets[i] / 100
        opened.append((exits[i], p))
        cnt[name] += 1
        pnl_by[name] += p
        if log:
            journal.append({"вход (Киев)": pd.Timestamp(t, unit="ms", tz="UTC").tz_convert("Europe/Kyiv")
                            .strftime("%Y-%m-%d %H:%M"), "стратегия": name, "монета": syms[i],
                            "результат, % цены": round(nets[i], 2), "плечо": LEV[name], "$": round(p, 2)})
    settle(float("inf"))
    return eq, mdd, cnt, pnl_by, monthly, journal


def main():
    rows, out = [], []
    combos = [["шорт", "лонг", "L3"], ["шорт", "лонг", "L3", "листинг"], ["листинг"], ["шорт"], ["лонг"], ["L3"]]
    for slip in (False, True):
        ev = load(slip)
        for use in combos:
            if slip and "листинг" not in use:
                continue
            res = [simulate(ev, use, np.random.default_rng(i)) for i in range(RUNS)]
            fin = np.array([r[0] for r in res])
            dd = np.array([r[1] for r in res])
            cnt = {n: np.median([r[2][n] for r in res]) for n in use}
            pb = {n: np.median([r[3][n] for r in res]) for n in use}
            mon = pd.DataFrame([r[4] for r in res]).median()
            rows.append({"стратегии": " + ".join(use) + (" (стоп с проскальз.)" if slip else ""),
                         "сделок": " / ".join(f"{n} {cnt[n]:.0f}" for n in use),
                         "итог: средний $": np.median(fin), "плохой (5%) $": np.percentile(fin, 5),
                         "хороший (95%) $": np.percentile(fin, 95), "просадка: обычная, %": np.median(dd),
                         "тяжёлая (5%), %": np.percentile(dd, 95),
                         "вклад, $": " / ".join(f"{n} {pb[n]:+.0f}" for n in use),
                         "худший месяц, $": mon.min() if len(mon) else np.nan})
            if use == ["шорт", "лонг", "L3", "листинг"] and not slip:
                pd.DataFrame(simulate(ev, use, np.random.default_rng(0), log=True)[5]).to_csv(
                    "results/combo2026_journal.csv", index=False)
                out.append(mon)
    df = pd.DataFrame(rows)
    with open("results/combo2026_summary.md", "w") as f:
        f.write("# Все стратегии с 2026-01-01: $100, 3 общих слота\n\n" + __doc__.split("\n", 2)[2] + "\n")
        f.write(df.round(1).to_markdown(index=False) + "\n\n")
        if out:
            f.write("Помесячно (все 4 стратегии, медиана по повторам, $):\n\n" +
                    out[0].round(1).to_frame("$").to_markdown() + "\n")
    print(df.round(1).to_string())
    if out:
        print(out[0].round(1).to_string())


if __name__ == "__main__":
    main()
