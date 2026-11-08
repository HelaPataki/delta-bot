"""
Вход и выход для рабочего условия «толпа в шортах + OI + дельта» (1h-свечи). Цель — винрейт от 55%.

Условия (сигнал — свеча, где условие стало выполняться; пауза 72 ч между сигналами монеты):
  «л/ш ≤ 1»   — лонг/шорт аккаунтов ≤ 1, OI за 3д ≥ +10%, накопленная дельта 7д ≥ 0;
  «л/ш ≤ 0.8» — то же с л/ш ≤ 0.8 (лучшее по подбору на 2025).
Входы (лонг):
  A сразу          — закрытие сигнальной свечи;
  B подтверждение  — в течение 72 ч первая свеча, закрывшаяся на +6% выше цены сигнала с объёмом >= x3;
  C откат −3%      — лимитка на 3% ниже цены сигнала, действует 24 ч (исполнилась, если low коснулся);
  D откат −6%      — то же на 6% ниже;
  E пробой         — стоп-ордер на 0.5% выше максимума 72 ч до сигнала, действует 72 ч
                     (если цена откроется выше уровня — вход по открытию).
Вход внутри свечи (C, D, E): если low этой же свечи ниже стопа — считается стоп (порядок внутри часа неизвестен).
Выходы: стоп 4/6/8/12% × тейк 4/6/8/10/15/20/30% × удержание не дольше 24/48/72 ч (потом по закрытию).
  Стоп и тейк в одной свече — стоп. Комиссия 0.1% за сделку.
Для каждого варианта: винрейт и средний % по годам (2025 / 2026), самая длинная серия убытков
и максимальная просадка депозита при риске 2% депозита на сделку (сделки по порядку входа).
"""

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
import pre_pump_trades as TR

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "results/pre_pump_exits"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
WAIT = PP.C(72)
STOPS = [4, 6, 8, 12]
TAKES = [4, 6, 8, 10, 15, 20, 30]
HOLDS = [24, 48, 72]
FEE = 0.1
RISK = 2.0       # % депозита на сделку
WIN_MIN = 55
MIN_N = int(os.environ.get("MIN_N", "50"))   # минимум сделок в каждом году


def exits(o, h, l, c, e, px, inbar_low=None):
    """Все варианты выхода для входа по цене px на свече e; путь — со свечи e+1.
    inbar_low — если вход внутри свечи e (лимитка/стоп-ордер): порядок внутри свечи неизвестен,
    поэтому консервативно: если low этой свечи ниже стопа — считается стоп, тейк в этой свече не засчитывается."""
    maxh = PP.C(max(HOLDS))
    if e + maxh >= len(c):
        return None
    hw, lw = h[e + 1:e + 1 + maxh], l[e + 1:e + 1 + maxh]
    res = {}
    for st in STOPS:
        s_hit = lw <= px * (1 - st / 100)
        si = int(np.argmax(s_hit)) if s_hit.any() else 10 ** 6
        if inbar_low is not None and inbar_low <= px * (1 - st / 100):
            si = -1
        for tk in TAKES:
            t_hit = hw >= px * (1 + tk / 100)
            ti = int(np.argmax(t_hit)) if t_hit.any() else 10 ** 6
            for hd in HOLDS:
                n = PP.C(hd)
                if si < n and si <= ti:
                    pct = -st
                elif ti < n:
                    pct = tk
                else:
                    pct = (c[e + n] / px - 1) * 100
                res[f"{st}|{tk}|{hd}"] = round(pct - FEE, 3)
    return res


def coin(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return None
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    t, vx, elig = f.open_time.to_numpy(), f.vol_x.to_numpy(), f.elig.to_numpy()
    n = len(f)
    hi72 = pd.Series(h).rolling(WAIT).max().to_numpy()
    conds = {"л/ш ≤ 1": (f.ls_acc <= 1) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0),
             "л/ш ≤ 0.8": (f.ls_acc <= 0.8) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0)}
    rows = []

    def add(cond, entry, s, e, px, inbar=False):
        r = exits(o, h, l, c, e, px, l[e] if inbar else None)
        if r is not None:
            rows.append({"symbol": sym, "cond": cond, "entry": entry, "signal_time": t[s], "entry_time": t[e],
                         "year": pd.Timestamp(t[s], unit="ms").year, **r})

    for cname, cond in conds.items():
        for s in TR.signals(cond, elig):
            if s + WAIT + PP.C(max(HOLDS)) >= n:
                continue
            ps = c[s]
            add(cname, "A сразу", s, s, ps)
            for j in range(s + 1, s + 1 + WAIT):                       # B
                if c[j] >= ps * 1.06 and vx[j] >= 3:
                    add(cname, "B подтверждение", s, j, c[j])
                    break
            for name, dip in (("C откат −3%", 0.03), ("D откат −6%", 0.06)):
                lvl = ps * (1 - dip)
                for j in range(s + 1, s + 1 + PP.C(24)):
                    if l[j] <= lvl:
                        add(cname, name, s, j, min(lvl, o[j]), inbar=True)
                        break
            lvl = hi72[s] * 1.005                                       # E
            for j in range(s + 1, s + 1 + WAIT):
                if h[j] >= lvl:
                    add(cname, "E пробой", s, j, max(lvl, o[j]), inbar=True)
                    break
    return pd.DataFrame(rows)


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
    pd.concat(parts).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps[Hst.base_asset(s)] for s in syms if Hst.base_asset(s) in caps}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni))


def streak(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def max_dd(r):
    eq = np.cumprod(1 + RISK / 100 * r)
    peak = np.maximum.accumulate(np.concatenate([[1], eq]))[1:]
    return (eq / peak - 1).min() * 100


def combine():
    files = sorted(f for f in os.listdir(f"{OUT}/parts") if f.endswith(".csv.gz"))
    a = pd.concat([pd.read_csv(f"{OUT}/parts/{f}") for f in files], ignore_index=True).sort_values("entry_time")
    months = {y: max((g.signal_time.max() - g.signal_time.min()) / 86_400_000 / 30.4, 1) for y, g in a.groupby("year")}
    cols = [c for c in a.columns if c.count("|") == 2]
    rows = []
    for (cond, entry), g in a.groupby(["cond", "entry"]):
        g25, g26 = g[g.year == 2025], g[g.year == 2026]
        for col in cols:
            st, tk, hd = (float(x) for x in col.split("|"))
            x25, x26 = g25[col], g26[col]
            if len(x25) < MIN_N or len(x26) < MIN_N:
                continue
            r = (g[col] / st).to_numpy()
            rows.append({"условие": cond, "вход": entry, "стоп %": st, "тейк %": tk, "держать до, ч": int(hd),
                         "сделок/мес 25": round(len(x25) / months.get(2025, 1), 1), "сделок/мес 26": round(len(x26) / months.get(2026, 1), 1),
                         "винрейт 25": (x25 > 0).mean() * 100, "винрейт 26": (x26 > 0).mean() * 100,
                         "ср. % 25": x25.mean(), "ср. % 26": x26.mean(),
                         "серия убытков": streak(g[col].to_numpy()),
                         f"просадка при риске {RISK:g}%": max_dd(r),
                         f"итог при риске {RISK:g}%": (np.prod(1 + RISK / 100 * r) - 1) * 100})
    t = pd.DataFrame(rows)
    t["хуже из двух лет, %"] = t[["ср. % 25", "ср. % 26"]].min(axis=1)
    fmt = t.copy()
    for c in ("винрейт 25", "винрейт 26"):
        fmt[c] = fmt[c].round(0).astype(int).astype(str) + "%"
    for c in ("ср. % 25", "ср. % 26", "хуже из двух лет, %"):
        fmt[c] = fmt[c].round(2)
    for c in (f"просадка при риске {RISK:g}%", f"итог при риске {RISK:g}%"):
        fmt[c] = fmt[c].round(0).astype(int).astype(str) + "%"

    w = open(f"{OUT}/pre_pump_exits_summary.md", "w").write
    w("# Вход и выход для «толпа в шортах + OI + дельта» (1h)\n\n"
      "Лонг. Комиссия 0.1% вычтена; % — движение цены на сделку (при x2 вдвое больше). "
      f"Серия убытков — самая длинная подряд за оба года (сделки по порядку входа). "
      f"Просадка и итог — если на каждую сделку рисковать {RISK:g}% депозита (позиция = {RISK:g}% / стоп). "
      "Сделки, открытые одновременно, не ограничены.\n\n")
    ok = (t["винрейт 25"] >= WIN_MIN) & (t["винрейт 26"] >= WIN_MIN) & (t["хуже из двух лет, %"] > 0)
    w(f"## Винрейт ≥ {WIN_MIN}% в обоих годах и плюс в обоих годах ({ok.sum()} вариантов)\n\n")
    if ok.any():
        idx = t[ok].sort_values("хуже из двух лет, %", ascending=False).index[:30]
        w(fmt.loc[idx].drop(columns=["хуже из двух лет, %"]).to_markdown(index=False) + "\n\n")
    else:
        w("Таких нет.\n\n")
    w("## Лучшие по каждому входу (по худшему из двух лет)\n\n")
    best = t.loc[t.groupby(["условие", "вход"])["хуже из двух лет, %"].idxmax()]
    w(fmt.loc[best.index].drop(columns=["хуже из двух лет, %"]).to_markdown(index=False) + "\n\n")
    w(f"## Лучшие с винрейтом ≥ {WIN_MIN}% по каждому входу (даже если в минусе)\n\n")
    hw = t[(t["винрейт 25"] >= WIN_MIN) & (t["винрейт 26"] >= WIN_MIN)]
    if len(hw):
        b2 = hw.loc[hw.groupby(["условие", "вход"])["хуже из двух лет, %"].idxmax()]
        w(fmt.loc[b2.index].drop(columns=["хуже из двух лет, %"]).to_markdown(index=False) + "\n\n")
    fmt.drop(columns=["хуже из двух лет, %"]).to_csv(f"{OUT}/pre_pump_exits_all.csv", index=False)
    print(open(f"{OUT}/pre_pump_exits_summary.md").read()[:5000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
