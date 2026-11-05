"""
Шорт по нисходящему тренду: пробой минимума за N дней и ретест пробитого уровня снизу. 4h-свечи.

Пробой: закрытие 4h-свечи ниже минимума предыдущих N дней (N = 20 / 30). Уровень = этот минимум.
  Фильтры на свече пробоя: капа $10M–$400M, оборот >= $50K. Один пробой на монету, пока он не отработан.
Направление «сразу»: от закрытия свечи пробоя, 10 дней (что раньше −4% / +4%, −10% / +10%).
Сделка «ретест»: условный ордер в шорт на уровне пробоя, ждём возврата цены до него W дней (3 / 7).
  Стоп +3% / +4,3% / +6% от входа, тейк = 1,8 x риск. По 4h-свечам: в свече входа проверяется только стоп
  (если её максимум выше стопа — стоп); дальше, если стоп и тейк в одной свече — считается стоп. Держим 10 дней.
Для сравнения — направление от всех свечей.
"""

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.environ.get("OUT_DIR", "results/trend_short")
NS = [20, 30]
WS = [3, 7]
STOPS = [3.0, 4.3, 6.0]
RR = 1.8
HOR = 60


def first(x):
    return int(np.argmax(x)) if x.any() else 10 ** 6


def direction(h, l, c, i):
    hw, lw = h[i + 1:i + 1 + HOR], l[i + 1:i + 1 + HOR]
    up, dn = hw / c[i] - 1, lw / c[i] - 1
    u4, d4 = first(up >= 0.04), first(dn <= -0.04)
    return {"complete": len(hw) == HOR, "first_dn": d4 < u4, "first_up": u4 < d4,
            "dn10": bool((dn <= -0.1).any()), "up10": bool((up >= 0.1).any())}


def trade(h, l, c, j, entry, stop_pct):
    stop, take = entry * (1 + stop_pct / 100), entry * (1 - stop_pct * RR / 100)
    if h[j] >= stop:
        return "стоп", -1.0
    hw, lw = h[j + 1:j + 1 + HOR], l[j + 1:j + 1 + HOR]
    if len(hw) < HOR:
        return "не завершена", np.nan
    s, t = first(hw >= stop), first(lw <= take)
    if s == t == 10 ** 6:
        return "по времени", (entry - c[j + HOR]) / (stop - entry)
    return ("стоп", -1.0) if s <= t else ("тейк", RR)


def coin(sym, mcap_now):
    k = Hst.load_klines(sym)
    if k is None or len(k) < 200:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    qv, t = k.quote_volume.to_numpy(), k.open_time.to_numpy()
    mcap = mcap_now * c / c[-1]
    start = pd.Timestamp(Hst.START, tz="UTC").timestamp() * 1000
    n = len(k)
    rows, base = [], []
    for i in range(0, n - 1, 3):                                   # базовая выборка свечей для сравнения
        if t[i] >= start and Hst.MCAP_MIN <= mcap[i] <= Hst.MCAP_MAX and qv[i] >= Hst.MIN_TURNOVER:
            base.append({"symbol": sym, **direction(h, l, c, i)})
    for N in NS:
        w = N * 6
        prev_low = pd.Series(l).shift(1).rolling(w, min_periods=w).min().to_numpy()
        busy = -1
        for i in range(w, n - 1):
            if i <= busy or t[i] < start or not (c[i] < prev_low[i]):
                continue
            if not (Hst.MCAP_MIN <= mcap[i] <= Hst.MCAP_MAX) or qv[i] < Hst.MIN_TURNOVER:
                continue
            lvl = prev_low[i]
            r = {"symbol": sym, "N": N, "time": t[i], "level": lvl, "close": c[i],
                 "below_pct": (c[i] / lvl - 1) * 100, **direction(h, l, c, i)}
            for W in WS:
                j_hit = None
                for j in range(i + 1, min(i + 1 + W * 6, n)):
                    if h[j] >= lvl:
                        j_hit = j
                        break
                r[f"retest{W}"] = j_hit is not None
                r[f"retest{W}_h"] = (j_hit - i) * 4 if j_hit is not None else np.nan
                for sp in STOPS:
                    if j_hit is None:
                        r[f"res_{W}_{sp}"], r[f"R_{W}_{sp}"] = "не было ретеста", np.nan
                    else:
                        res, R = trade(h, l, c, j_hit, lvl, sp)
                        r[f"res_{W}_{sp}"], r[f"R_{W}_{sp}"] = res, R
            rows.append(r)
            busy = i + HOR                                         # следующий пробой по монете — не раньше чем через 10 дней
    return pd.DataFrame(rows), pd.DataFrame(base)


def dstats(g):
    g = g[g.complete]
    return {"сигналов": len(g), "сначала −4%": f"{g.first_dn.mean() * 100:.0f}%", "сначала +4%": f"{g.first_up.mean() * 100:.0f}%",
            "−10% за 10д": f"{g.dn10.mean() * 100:.0f}%", "+10% за 10д": f"{g.up10.mean() * 100:.0f}%"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    with_cap = [s for s in syms if Hst.base_asset(s) in caps]

    def work(sym):
        try:
            return coin(sym, caps[Hst.base_asset(sym)])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(8) as ex:
        res = [r for r in ex.map(work, with_cap) if r is not None]
    a = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    b = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    months = (a.time.max() - a.time.min()) / 86_400_000 / 30.4

    with open(f"{OUT_DIR}/trend_short_summary.md", "w") as f:
        f.write("# Шорт по нисходящему тренду: пробой минимума за N дней и ретест уровня\n\n")
        f.write(f"Сделка: шорт на уровне пробоя при возврате цены к нему снизу; тейк = {RR:g} × риск "
                f"(безубыточный винрейт {1 / (1 + RR) * 100:.1f}%, безубыточно = 0R). По 4h-свечам, комиссии не учтены.\n\n")
        f.write("## Направление сразу после пробоя (от закрытия свечи пробоя)\n\n")
        rows = [{"группа": "все свечи (для сравнения)", **dstats(b)}]
        for N in NS:
            rows.append({"группа": f"пробой минимума {N} дн.", "в месяц": f"{(a.N == N).sum() / months:.1f}", **dstats(a[a.N == N])})
        f.write(pd.DataFrame(rows).fillna("").to_markdown(index=False) + "\n\n")
        f.write("## Сделка на ретесте уровня\n\n")
        rows = []
        for N in NS:
            g = a[a.N == N]
            for W in WS:
                for sp in STOPS:
                    res, R = g[f"res_{W}_{sp}"], g[f"R_{W}_{sp}"]
                    closed = res.isin(["тейк", "стоп", "по времени"])
                    tk, st = (res == "тейк").sum(), (res == "стоп").sum()
                    rows.append({"пробой": f"{N} дн.", "ждём ретест": f"{W} дн.",
                                 "был ретест": f"{g[f'retest{W}'].mean() * 100:.0f}%", "стоп": f"+{sp:g}%",
                                 "сделок": int(closed.sum()), "в месяц": f"{closed.sum() / months:.1f}",
                                 "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
                                 "в среднем R": f"{R[closed].mean():+.2f}" if closed.any() else "-"})
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out = a.copy()
    out["time"] = pd.to_datetime(out.time, unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M")
    out.round(4).to_csv(f"{OUT_DIR}/trend_short_signals.csv", index=False)
    print(open(f"{OUT_DIR}/trend_short_summary.md").read())


if __name__ == "__main__":
    main()
