"""
Сделки по признакам «до пампа» (1h-свечи) и честная проверка: 2025 — подбор, 2026 — проверка.

Признаки — те же, что в pre_pump.py (считаются той же функцией), только на момент закрытия свечи.
Сигнал — первая свеча, где условие стало выполняться (на прошлой свече не выполнялось),
и не раньше чем через 72 ч после прошлого сигнала этой же монеты по этому же условию.

Входы (только лонг):
  A «сразу»          — по закрытию сигнальной свечи;
  B «подтверждение»  — в течение 72 ч после сигнала первая 1h-свеча, закрывшаяся на +6% выше цены сигнала
                       с объёмом >= x3 к норме; вход по её закрытию. Нет такой свечи — нет сделки.
Выходы (три варианта): стоп 4.3% / тейк 7.74%, стоп 8% / тейк 16%, стоп 12% / тейк 30%;
  если за 72 ч ни то ни другое — выход по закрытию через 72 ч. Стоп и тейк в одной свече — считается стоп.
  Комиссия 0.1% за сделку (вход + выход) вычитается. Результат — % движения цены (при x2 доход ×2).
Сравнение: «случайный вход» — каждая монета раз в сутки в 00:00 UTC; «подтверждение без фильтра» —
  любая свеча +6% за 6 ч с объёмом >= x3, сразу вход.
Пересекающиеся сделки не ограничиваются (это проверка признаков, а не депозита).

Режимы: MODE=prep, SHARD=0..N-1, MODE=combine.
"""

import itertools
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("INTERVAL", "1h")

import numpy as np
import pandas as pd

import history as Hst
import pre_pump as PP

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "results/pre_pump_trades"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
HOLD = PP.C(72)
CONF_UP, CONF_VOL = 0.06, 3
FEE = 0.1
EXITS = [("стоп 4.3 / тейк 7.74", 4.3, 7.74), ("стоп 8 / тейк 16", 8.0, 16.0), ("стоп 12 / тейк 30", 12.0, 30.0)]


def fixed(f):
    return {
        "толпа в шортах + OI 3д ≥ +10% + дельта 7д ≥ 0": (f.ls_acc <= 1) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0),
        "крупные сделки (≥ 1.22) + дельта 7д ≥ 0": (f.trade_size_x >= 1.22) & (f.cvd7d >= 0),
        "аномалия объёма ≥ x10 за 3д + дельта 7д ≥ 0": (f.volx_max3d >= 10) & (f.cvd7d >= 0),
        "толпа в шортах + дельта 7д ≥ 0": (f.ls_acc <= 1) & (f.cvd7d >= 0),
        "вынос лонгов: OI за 1д ≤ −6%": f.oi_chg1d <= -6,
    }


GRID = list(itertools.product([0.8, 1.0, 1.2], [5, 10, 20], [-2, 0, 2]))   # ls_acc ≤, OI 3д ≥, дельта 7д ≥


def grid(f):
    return {f"сетка: л/ш ≤ {a}, OI 3д ≥ +{b}%, дельта ≥ {c}": (f.ls_acc <= a) & (f.oi_chg3d >= b) & (f.cvd7d >= c)
            for a, b, c in GRID}


def signals(cond, elig):
    """Индексы: условие стало выполняться, пауза 72 ч между сигналами."""
    cond = cond.fillna(False).to_numpy() & elig
    prev = np.concatenate([[False], cond[:-1]])
    out, last = [], -10 ** 9
    for i in np.flatnonzero(cond & ~prev):
        if i - last > HOLD:
            out.append(i)
            last = i
    return out


def trade(h, l, c, e):
    """Результаты трёх вариантов выхода для входа по закрытию свечи e (None, если не хватает данных)."""
    if e + HOLD >= len(c):
        return None
    px, hw, lw = c[e], h[e + 1:e + 1 + HOLD], l[e + 1:e + 1 + HOLD]
    res = {}
    for name, st, tk in EXITS:
        s_hit, t_hit = lw <= px * (1 - st / 100), hw >= px * (1 + tk / 100)
        si = int(np.argmax(s_hit)) if s_hit.any() else 10 ** 6
        ti = int(np.argmax(t_hit)) if t_hit.any() else 10 ** 6
        if si == ti == 10 ** 6:
            pct = (c[e + HOLD] / px - 1) * 100
        elif si <= ti:
            pct = -st
        else:
            pct = tk
        res[name] = pct - FEE
    return res


def coin(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return None
    h, l, c = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    t, vx, elig = f.open_time.to_numpy(), f.vol_x.to_numpy(), f.elig.to_numpy()
    n = len(f)
    rows = []

    def add(strat, entry, s, e):
        r = trade(h, l, c, e)
        if r is None:
            return
        end = min(s + HOLD, n - 1)
        pumped = h[s + 1:end + 1].max() / c[s] - 1 >= PP.PUMP
        left = (h[e + 1:end + 1].max() / c[e] - 1) * 100 if e < end else 0.0
        rows.append({"symbol": sym, "strat": strat, "entry": entry, "signal_time": t[s], "entry_time": t[e],
                     "year": pd.Timestamp(t[s], unit="ms").year, "pumped": pumped, "left_to_peak": left, **r})

    def confirm(s):
        for j in range(s + 1, min(s + 1 + HOLD, n)):
            if c[j] >= c[s] * (1 + CONF_UP) and vx[j] >= CONF_VOL:
                return j
        return None

    for strat, cond in {**fixed(f), **grid(f)}.items():
        for s in signals(cond, elig):
            add(strat, "A сразу", s, s)
            j = confirm(s)
            if j is not None:
                add(strat, "B подтверждение", s, j)
    hours = pd.to_datetime(t, unit="ms").hour
    for s in np.flatnonzero(elig & (hours == 0)):
        add("случайный вход", "A сразу", s, s)
    jump = pd.Series((c / pd.Series(c).shift(PP.C(6)).to_numpy() >= 1 + CONF_UP) & (vx >= CONF_VOL))
    for s in signals(jump, elig):
        add("подтверждение без фильтра", "B подтверждение", s, s)
    return pd.DataFrame(rows)


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps[Hst.base_asset(s)] for s in syms if Hst.base_asset(s) in caps}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]
    if LIMIT:
        mine = mine[:LIMIT]
    b = Hst.load_klines("BTCUSDT", PP.INTERVAL)
    btc = pd.DataFrame({"open_time": b.open_time, "btc7d": (b.close / b.close.shift(PP.C(168)) - 1) * 100})
    print(f"часть {SHARD}: монет {len(mine)}", flush=True)

    def work(sym):
        try:
            return coin(sym, uni[sym], btc)
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(3) as ex:
        for i, p in enumerate(ex.map(work, mine), 1):
            if p is not None and len(p):
                parts.append(p)
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.concat(parts).round(3).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)


def stats(g, ex, months):
    x = g[ex]
    if len(x) == 0:
        return None
    return {"сделок": len(x), "в месяц": round(len(x) / months, 1),
            "винрейт": f"{(x > 0).mean() * 100:.0f}%", "в среднем %": round(x.mean(), 2),
            "медиана %": round(x.median(), 2), "сумма %": round(x.sum(), 0)}


def combine():
    files = sorted(f for f in os.listdir(f"{OUT}/parts") if f.endswith(".csv.gz"))
    a = pd.concat([pd.read_csv(f"{OUT}/parts/{f}") for f in files], ignore_index=True)
    span = {y: max((g.signal_time.max() - g.signal_time.min()) / 86_400_000 / 30.4, 1) for y, g in a.groupby("year")}
    exn = [e[0] for e in EXITS]
    w = open(f"{OUT}/pre_pump_trades_summary.md", "w").write
    w("# Сделки по признакам «до пампа» (1h)\n\n"
      "Лонг. A — вход сразу по закрытию сигнальной свечи; B — вход после подтверждения (+6% от цены сигнала, "
      "объём ≥ x3, в течение 72 ч). Выход: стоп / тейк или по времени через 72 ч. Комиссия 0.1% вычтена. "
      "% — движение цены на сделку (при x2 доход и убыток вдвое больше). Пересекающиеся сделки не ограничены.\n\n"
      "**2025 — год подбора, 2026 — проверка.** Если признак работает в 2025 и не работает в 2026 — это подгонка.\n\n")

    def table(sub):
        rows = []
        for (strat, entry), g in sub.groupby(["strat", "entry"], sort=False):
            for ex in exn:
                for y in sorted(g.year.unique()):
                    r = stats(g[g.year == y], ex, span[y])
                    if r:
                        rows.append({"условие": strat, "вход": entry, "выход": ex, "год": y, **r})
        return pd.DataFrame(rows)

    order = list(fixed(pd.DataFrame(columns=["ls_acc", "oi_chg3d", "cvd7d", "trade_size_x", "volx_max3d", "oi_chg1d"])))
    main = a[a.strat.isin(order + ["случайный вход", "подтверждение без фильтра"])]
    w("## Готовые условия и сравнение\n\n" + table(main).to_markdown(index=False) + "\n\n")

    # подбор по сетке на 2025 -> проверка на 2026
    g = a[a.strat.str.startswith("сетка")]
    best = []
    for (strat, entry), gg in g.groupby(["strat", "entry"]):
        tr = gg[gg.year == 2025]
        if len(tr) < 100:
            continue
        for ex in exn:
            best.append((tr[ex].mean(), strat, entry, ex))
    best.sort(reverse=True)
    rows = []
    for _, strat, entry, ex in best[:10]:
        gg = g[(g.strat == strat) & (g.entry == entry)]
        r25, r26 = stats(gg[gg.year == 2025], ex, span[2025]), stats(gg[gg.year == 2026], ex, span.get(2026, 1))
        rows.append({"условие": strat, "вход": entry, "выход": ex,
                     "2025 сделок": r25["сделок"], "2025 в среднем %": r25["в среднем %"], "2025 винрейт": r25["винрейт"],
                     "2026 сделок": r26["сделок"] if r26 else 0, "2026 в среднем %": r26["в среднем %"] if r26 else "-",
                     "2026 винрейт": r26["винрейт"] if r26 else "-"})
    w("## Подбор на 2025 → проверка на 2026\n\n"
      f"Сетка: {len(GRID)} вариантов порогов «толпа в шортах + OI + дельта» × 2 входа × 3 выхода. "
      "Лучшие 10 по 2025 (не меньше 100 сделок) и что с ними в 2026.\n\n"
      + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")

    # сколько пампа остаётся после подтверждения
    rows = []
    for strat, gg in main[main.entry == "B подтверждение"].groupby("strat", sort=False):
        p = gg[gg.pumped]
        rows.append({"условие": strat, "подтверждений": len(gg), "из них был памп +30%": f"{gg.pumped.mean() * 100:.0f}%",
                     "медиана остатка до пика (все)": round(gg.left_to_peak.median(), 1),
                     "медиана остатка до пика (пампы)": round(p.left_to_peak.median(), 1) if len(p) else "-"})
    w("## Подтверждение: сколько пампа остаётся\n\n"
      "Остаток до пика — максимум цены за 72 ч от сигнала после входа, % от цены входа.\n\n"
      + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    a["signal_time"] = pd.to_datetime(a.signal_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    a["entry_time"] = pd.to_datetime(a.entry_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    a[a.strat.isin(order)].to_csv(f"{OUT}/pre_pump_trades.csv", index=False)
    print(open(f"{OUT}/pre_pump_trades_summary.md").read()[:5000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
