"""
Проверка двух идей для алерта (перед созданием бота). Старые скрипты не меняются.

1) Шорт: «не входить, если за 15 минут после сигнала цена уже ушла вниз на X%+».
   Сравнивается цена закрытия сигнальной свечи и цена входа (открытие минуты через 15 мин).
2) Выбор монеты из пачки: случайно (как в симуляции) против правил
   (шорт — самый большой рост OI / самый отрицательный фандинг; лонг — самый большой объём / самая сильная дельта).

Итог — средний % на сделку и симуляция $100, 2 слота, шорт + лонг 1:4 / 10 дн., x1 и x2.
"""

import glob
import os

import numpy as np
import pandas as pd

OUT_DIR = "results/alert_rules"
N_RUNS, FEE, SLOTS, START = 1000, 0.1, 2, 100.0


def load():
    sh = pd.read_csv("results/oi_sim2/oi_sim_trades.csv.gz")
    sh = sh[sh.G1 & (sh.fund24 <= -0.2) & (sh.oi_chg24 >= 60) & (sh.rr == 1.8) & (sh.hold_h == 24)].copy()
    sig = pd.concat(pd.read_csv(f) for f in glob.glob("results/oi_funding_2025/parts/*.csv.gz"))
    sig = sig[["symbol", "open_time", "close"]].drop_duplicates(["symbol", "open_time"])
    sh = sh.merge(sig, on=["symbol", "open_time"], how="left")
    sh["drift"] = (sh.entry / sh["close"] - 1) * 100          # + цена выросла, − упала за 15 мин
    sh["net"] = sh.pnl_pct + sh.funding_pct - FEE
    lg = pd.read_csv("results/long_sim/oi_sim_trades.csv.gz")
    lg = lg[(lg.rr == 4.0) & (lg.hold_h == 240.0)].copy()
    lg["net"] = lg.pnl_pct + lg.funding_pct - FEE
    return sh, lg


def year(df):
    return pd.to_datetime(df.open_time, unit="ms").dt.year


def events(df, side, pick):
    ev = []
    for _, c in df.groupby("open_time", sort=True):
        c = c.sort_values("entry_time")
        ev.append((c.entry_time.iloc[0], side, c, pick))
    return ev


def choose(c, pick, rng):
    if pick == "random":
        return rng.integers(len(c))
    col, asc = pick
    return int(np.argsort(c[col].to_numpy() * (1 if asc else -1), kind="stable")[0])


def simulate(ev, lev, rng):
    bal, busy = np.full(SLOTS, START / SLOTS), np.zeros(SLOTS)
    peak, dd = START, 0.0
    for t_in, side, c, pick in ev:
        free = np.where(busy <= t_in)[0]
        if len(free) == 0:
            continue
        i, k = choose(c, pick, rng), free[0]
        bal[k] = max(bal[k] * (1 + lev * c.net.iloc[i] / 100), 0.0)
        busy[k] = c.exit_time.iloc[i]
        eq = bal.sum()
        peak, dd = max(peak, eq), min(dd, eq / max(peak, eq) - 1)
    return bal.sum(), dd


def sim_row(name, sh, lg, sp="random", lp="random"):
    ev = sorted(events(sh, "шорт", sp) + events(lg, "лонг", lp), key=lambda e: e[0])
    row = {"вариант": name, "шортов в истории": len(sh)}
    for lev in (1, 2):
        res = [simulate(ev, lev, np.random.default_rng(i)) for i in range(N_RUNS)]
        fin, dd = np.array([r[0] for r in res]), np.array([r[1] for r in res])
        row[f"x{lev} средний"] = f"${np.median(fin):,.0f}"
        row[f"x{lev} плохой (5%)"] = f"${np.percentile(fin, 5):,.0f}"
        row[f"x{lev} просадка тяжёлая"] = f"{np.percentile(dd, 5) * 100:.0f}%"
    print(row, flush=True)
    return row


def pack_pick(df, rules):
    """Средний результат монеты, выбранной правилом, в пачках из 2+ монет, по годам."""
    packs = [c for _, c in df.groupby("open_time") if len(c) > 1]
    rows = []
    for name, pick in rules:
        vals, yrs = [], []
        for c in packs:
            v = c.net.mean() if pick == "random" else c.net.iloc[choose(c, pick, None)]
            vals.append(v); yrs.append(pd.to_datetime(c.open_time.iloc[0], unit="ms").year)
        s = pd.Series(vals, index=yrs)
        r = {"правило": name, "пачек": len(packs), "% на сделку": round(s.mean(), 2)}
        for y in sorted(set(yrs)):
            r[f"{y}"] = round(s[s.index == y].mean(), 2)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sh, lg = load()
    out = ["# Проверка правил для алерта\n"]

    # 1. Куда ушла цена за 15 минут и что было со сделкой
    bins = [-100, -2, -1, -0.5, 0, 0.5, 1, 2, 100]
    labels = ["упала >2%", "упала 1–2%", "упала 0,5–1%", "упала 0–0,5%", "выросла 0–0,5%", "выросла 0,5–1%", "выросла 1–2%", "выросла >2%"]
    sh["корзина"] = pd.cut(sh.drift, bins, labels=labels)
    t = sh.groupby("корзина", observed=False).agg(сделок=("net", "size"), винрейт=("result", lambda r: (r == "тейк").mean() * 100),
                                                   **{"% на сделку": ("net", "mean")}).round(2).reset_index()
    out.append("## 1. Шорт: куда ушла цена за 15 минут от сигнала до входа\n")
    out.append(f"Всего шортов: {len(sh)}, нет цены сигнала: {sh['close'].isna().sum()}. "
               f"Средний сдвиг {sh.drift.mean():+.2f}%, медиана {sh.drift.median():+.2f}%.\n")
    out.append(t.to_markdown(index=False) + "\n")

    rows = [sim_row("как было (входим всегда)", sh, lg)]
    for x in (0.5, 1.0, 2.0):
        f = sh[~(sh.drift <= -x)]
        r = sim_row(f"не входить, если упала на {x:g}%+", f, lg)
        r["% на сделку пропущенных"] = round(sh[sh.drift <= -x].net.mean(), 2)
        rows.append(r)
    f = sh[sh.drift.abs() < 0.5]
    rows.append(sim_row("не входить, если ушла на 0,5%+ в любую сторону", f, lg))
    out.append("### Симуляция $100, 2 слота, шорт + лонг 1:4 / 10 дн.\n")
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    # 2. Выбор монеты из пачки
    out.append("## 2. Какую монету брать из пачки (только пачки из 2+ монет)\n")
    out.append("«% на сделку» — средний результат выбранной монеты; «случайно» = среднее по всей пачке. По годам — для проверки устойчивости.\n")
    out.append("### Шорт\n" + pack_pick(sh, [("случайно", "random"), ("самый большой рост OI", ("oi_chg24", False)),
                                             ("самый маленький рост OI", ("oi_chg24", True)),
                                             ("самый отрицательный фандинг", ("fund24", True)),
                                             ("самый слабый фандинг", ("fund24", False))]).to_markdown(index=False) + "\n")
    out.append("### Лонг\n" + pack_pick(lg, [("случайно", "random"), ("самый большой объём (x)", ("vol_ratio", False)),
                                            ("самый маленький объём (x)", ("vol_ratio", True)),
                                            ("самая сильная дельта", ("delta_pct", True)),
                                            ("самая слабая дельта", ("delta_pct", False))]).to_markdown(index=False) + "\n")
    rows = [sim_row("случайно (как было)", sh, lg),
            sim_row("шорт: max OI, лонг: max объём", sh, lg, ("oi_chg24", False), ("vol_ratio", False)),
            sim_row("шорт: min фандинг, лонг: min дельта", sh, lg, ("fund24", True), ("delta_pct", True))]
    out.append("### Симуляция с выбором по правилу\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")

    open(f"{OUT_DIR}/alert_rules_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
