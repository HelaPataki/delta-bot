"""
Эксперимент (не бот): порог фильтра «вход не выше X% от максимума 30 дней» + примеры сделок.

Рабочее условие: л/ш аккаунтов ≤ 0.8, OI 3д ≥ +10%, накопл. дельта 7д ≥ 0 (1h-свечи, с 2025-01-01);
вход на подтверждении (+6% к цене сигнала, объём ≥ x3, в течение 72 ч); стоп 12%, тейк 10% или 20%, до 48 ч.
Пороги: без фильтра, 80 / 85 / 90 / 95% от максимума 30 дней (по high, на момент входа).
Депозит: риск 2%, 3 слота, без ночи, стоп с проскальзыванием 1% (как в pre_pump_sim).
Примеры: с сентября 2026, порог 90%, тейк 20% — случайно 2 сделки в плюсе и 2 в минусе, подробно.
"""

import json
import os
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("INTERVAL", "1h")
os.environ.setdefault("START", "2024-07-01")
os.environ.setdefault("SIG_FROM", "2025-01-01")

import numpy as np
import pandas as pd

import history as Hst
import pre_pump as PP
import pre_pump_ake as AK
import pre_pump_examples as PE
import pre_pump_sim as PS
import pre_pump_trades as TR

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = "results/pre_pump_filter"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
C, H = PP.C, 3600 * 1000
TH = [None, 0.80, 0.85, 0.90, 0.95]
EX_TH, EX_TK = 0.90, 20
EX_FROM = pd.Timestamp("2026-09-01", tz=Hst.KYIV).timestamp() * 1000


def trades(f):
    """[(s, e, {10: (pct, k, kind), 20: ...}, to_hi30, px7d)] — сделки рабочего условия."""
    h, l, c = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    vx, elig, n = f.vol_x.to_numpy(), f.elig.to_numpy(), len(f)
    hi30 = pd.Series(h).rolling(C(720), min_periods=C(168)).max().to_numpy()
    cond = (f.ls_acc <= 0.8) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0)
    out = []
    for s in TR.signals(cond, elig):
        e = next((j for j in range(s + 1, min(s + 1 + C(72), n)) if c[j] >= c[s] * 1.06 and vx[j] >= 3), None)
        if e is None:
            continue
        res = {tk: AK.path_result(h, l, c, e, c[e], 12.0, tk, C(48)) for tk in (10, 20)}
        if None in res.values():
            continue
        out.append((s, e, res, c[e] / hi30[e], f.px7d.iloc[e]))
    return out


def coin(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return []
    t = f.open_time.to_numpy()
    rows = []
    for s, e, res, th, px7 in trades(f):
        row = {"symbol": sym, "signal_time": t[s], "entry_time": t[e] + AK.IH_MS,
               "year": pd.Timestamp(t[s], unit="ms").year, "to_hi30": th, "px7d_entry": px7}
        for tk, (pct, k, kind) in res.items():
            row[f"pct{tk}"], row[f"kind{tk}"], row[f"exit{tk}"] = pct - AK.FEE, kind, t[k] + AK.IH_MS
        rows.append(row)
    return rows


def btc_frame():
    b = Hst.load_klines("BTCUSDT", PP.INTERVAL)
    return pd.DataFrame({"open_time": b.open_time, "btc7d": (b.close / b.close.shift(C(168)) - 1) * 100})


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
    btc = btc_frame()
    print(f"часть {SHARD}: монет {len(mine)}", flush=True)

    def work(sym):
        try:
            return coin(sym, uni[sym], btc)
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(work, mine), 1):
            rows += r
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True).sort_values("entry_time")
    hr = pd.to_datetime(a.entry_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.hour
    a["night"] = (hr >= 23) | (hr <= 5)
    d = a[~a.night]
    w = open(f"{OUT}/pre_pump_filter_summary.md", "w").write
    w("# Порог фильтра «вход не выше X% от максимума 30 дней» (эксперимент, не бот)\n\n"
      "л/ш ≤ 0.8, OI 3д ≥ +10%, дельта 7д ≥ 0; вход на подтверждении; стоп 12%, до 48 ч. "
      "Депозит: старт 100, риск 2%, 3 слота, без ночи, стоп с проскальзыванием 1%. Комиссия 0.1%.\n\n")
    rows = []
    for th in TH:
        g = d if th is None else d[d.to_hi30 <= th]
        for tk in (10, 20):
            s, _ = PS.simulate(g, tk, 3, True)
            if not s:
                continue
            x25, x26 = g[g.year == 2025][f"pct{tk}"], g[g.year == 2026][f"pct{tk}"]
            rows.append({"порог": "без фильтра" if th is None else f"≤ {th * 100:.0f}%", "тейк %": tk,
                         "сделок 2025": len(x25), "сделок 2026": len(x26),
                         "винрейт 2025": f"{(x25 > 0).mean() * 100:.0f}%", "винрейт 2026": f"{(x26 > 0).mean() * 100:.0f}%",
                         "ср. % 2025": round(x25.mean(), 2), "ср. % 2026": round(x26.mean(), 2), **s})
    w(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    w("Где был вход относительно максимума 30 дней (все сделки, без ночи): " + ", ".join(
        f"≤ {x}%: {(d.to_hi30 <= x / 100).mean() * 100:.0f}%" for x in (70, 80, 85, 90, 95, 100)) + "\n\n")
    # результат по корзинам, чтобы видеть форму зависимости
    bins = [0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    lab = ["< 70%", "70–80%", "80–85%", "85–90%", "90–95%", "95–100%"]
    d2 = d.assign(корзина=pd.cut(d.to_hi30, bins, labels=lab))
    tb = d2.groupby("корзина", observed=True).agg(сделок=("pct20", "size"),
                                                  ср_тейк10=("pct10", "mean"), ср_тейк20=("pct20", "mean"),
                                                  винрейт_тейк20=("pct20", lambda x: f"{(x > 0).mean() * 100:.0f}%"))
    w("## Результат по тому, насколько вход ниже максимума 30 дней\n\n" + tb.round(2).to_markdown() + "\n\n")

    # примеры
    ex = a[(a.signal_time >= EX_FROM) & (a.to_hi30 <= EX_TH)]
    rnd = random.Random(int(os.environ.get("SEED", "11")))
    pick = []
    for want in (True, False):
        pool = ex[(ex[f"pct{EX_TK}"] > 0) == want]
        pick += [r for _, r in pool.sample(min(2, len(pool)), random_state=rnd.randint(0, 10 ** 6)).iterrows()]
    text = [f"# Примеры: порог {EX_TH * 100:.0f}%, тейк {EX_TK}%, с сентября 2026\n",
            f"Подходящих сделок с сентября: {len(ex)} (в плюсе: {(ex[f'pct{EX_TK}'] > 0).sum()}, "
            f"в минусе: {(ex[f'pct{EX_TK}'] <= 0).sum()}). Время — по Киеву, время свечи = её закрытие.\n"]
    uni = json.load(open(f"{OUT}/universe.json"))
    btc = btc_frame()
    for r in pick:
        f = PP.coin(r.symbol, uni[r.symbol], btc, keep_all=True)
        t = f.open_time.to_numpy()
        s = int(np.flatnonzero(t == r.signal_time)[0])
        for s2, e, res, th, px7 in trades(f):
            if s2 == s:
                p = f.iloc[s - 1]
                prev = [x for x, ok in ((f"лонг/шорт был {p.ls_acc:.3f}", p.ls_acc <= 0.8),
                                        (f"OI за 3д был {p.oi_chg3d:+.1f}%", p.oi_chg3d >= 10),
                                        (f"дельта 7д была {p.cvd7d:+.2f}%", p.cvd7d >= 0)) if not ok]
                extra = (f"Фильтр «не на вершине»: цена входа = {th * 100:.0f}% от максимума 30 дней (порог ≤ 90%), "
                         f"рост за 7 дней до входа {px7:+.1f}%.")
                text.append(PE.describe(r.symbol, f, s, e, res, ", ".join(prev) or "—", main_tk=EX_TK, extra=extra))
                break
    tab = ex.copy()
    for col in ("signal_time",):
        tab[col] = pd.to_datetime(tab[col], unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    tab["вход"] = pd.to_datetime(tab.entry_time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    tab["к макс. 30д"] = (tab.to_hi30 * 100).round(0).astype(int).astype(str) + "%"
    tab["рост 7д"] = tab.px7d_entry.round(0)
    tab["итог тейк 20"] = tab.kind20 + " " + tab.pct20.round(1).astype(str) + "%"
    text.append("## Все подходящие сделки с сентября\n\n" + tab[["symbol", "signal_time", "вход", "к макс. 30д", "рост 7д",
                                                                  "итог тейк 20"]].to_markdown(index=False) + "\n")
    open(f"{OUT}/examples.md", "w").write("\n".join(text))
    out = a.copy()
    for col in ("signal_time", "entry_time", "exit10", "exit20"):
        out[col] = pd.to_datetime(out[col], unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    out.round(4).to_csv(f"{OUT}/pre_pump_filter_trades.csv", index=False)
    print(open(f"{OUT}/pre_pump_filter_summary.md").read()[:4000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
