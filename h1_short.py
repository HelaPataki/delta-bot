"""
Шортовая теория на 1h-свечах: «запертые покупатели».

Сигнал (1h-свеча USDT-перпетуала Binance, с 2025-01-01, капа $10M–$400M, оборот свечи >= $12.5K):
  объём >= x3 / x5 / x10 к медиане 48 предыдущих 1h-свечей (те же 2 суток, что и на 4h),
  дельта >= +17%,
  «теория»: POC (минутка с наибольшим объёмом) в верхних 30% свечи и закрытие ниже POC.
Для сравнения — остальные сигналы (объём + дельта >= +17%, но без условия по POC).

1) Направление (по 1h high/low, 10 дней): что раньше −4% или +4%, как часто −10% / +10%.
2) Сделки по минуткам: шорт условным ордером, стоп +4,3%, тейк −7,74% (1:1,8),
   вход E1 — POC − 5 тиков, E2 — максимум свечи (с тенью); ожидание ордера 1 час и 4 часа.
   Если в одной минутке задеты и стоп, и тейк — считается стоп. Комиссии не учтены.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as B
import poc as P
import history as Hst
import short_tests as ST
from poc import kyiv_to_utc

STOP = float(os.environ.get("STOP_PCT", "4.3"))
TAKE = float(os.environ.get("TAKE_PCT", "7.74"))
OUT_DIR = os.environ.get("OUT_DIR", "results/h1_short" + ("" if (STOP, TAKE) == (4.3, 7.74) else f"_stop{STOP:g}_take{TAKE:g}"))
H1 = 3600 * 1000
HOR = 240                       # 10 дней по 1h
MIN_TURN = 12_500
TICKS = 5
WAITS = {"1 час": H1, "4 часа": 4 * H1}
THREADS = 16


class NoCache(dict):
    def __setitem__(self, k, v):
        pass


P._cache = NoCache()
ST.RR = TAKE / STOP          # тейк в R для расчёта сделок


def coin(sym, mcap_now):
    df = Hst.load_klines(sym, "1h")
    if df is None or len(df) < 500:
        return None
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    v, qv, tbv = df.volume.to_numpy(), df.quote_volume.to_numpy(), df.taker_buy_volume.to_numpy()
    delta = (2 * tbv - v) / v * 100
    norm = pd.Series(qv).shift(1).rolling(48, min_periods=48).median().to_numpy()
    vr = qv / norm
    mcap = mcap_now * c / c[-1]
    t = df.open_time.to_numpy()
    m = ((vr >= 3) & (delta >= 17) & (qv >= MIN_TURN) & (mcap >= Hst.MCAP_MIN) & (mcap <= Hst.MCAP_MAX)
         & (t >= pd.Timestamp(Hst.START, tz="UTC").timestamp() * 1000))
    rows = []
    for i in np.where(m)[0]:
        hw, lw = h[i + 1:i + 1 + HOR], l[i + 1:i + 1 + HOR]
        up, dn = hw / c[i] - 1, lw / c[i] - 1

        def first(x):
            return int(np.argmax(x)) if x.any() else 10 ** 6

        u4, d4 = first(up >= 0.04), first(dn <= -0.04)
        rows.append({"symbol": sym, "open_time": t[i], "close": c[i], "high": h[i], "change_pct": (c[i] / o[i] - 1) * 100,
                     "delta_pct": delta[i], "vol_ratio": vr[i], "turnover": qv[i], "mcap": mcap[i],
                     "complete": len(hw) == HOR, "first_dn": d4 < u4, "first_up": u4 < d4,
                     "dn10": bool((dn <= -0.10).any()), "up10": bool((up >= 0.10).any())})
    return pd.DataFrame(rows)


def poc_and_trades(r):
    t_open = r["open_time"]
    mins = ST.Mins(r["symbol"], t_open)
    sig = np.array(list(mins.rows(t_open, t_open + H1)))
    out = {"poc": np.nan, "poc_pos": np.nan}
    if len(sig) == 0:
        return out
    # для POC нужен объём минуток: берём их отдельно (с колонкой объёма)
    day = pd.Timestamp(t_open, unit="ms", tz="UTC").date()
    full = P.minutes(r["symbol"], day)
    if full is None:
        return out
    w = full[(full.open_time >= t_open) & (full.open_time < t_open + H1)]
    if len(w) == 0:
        return out
    i = w.volume.idxmax()
    poc = (w.at[i, "high"] + w.at[i, "low"]) / 2
    hi, lo = w.high.max(), w.low.min()
    out["poc"] = poc
    out["poc_pos"] = (poc - lo) / (hi - lo) * 100 if hi > lo else 50.0
    tick = B.infer_tick(mins.parts[0])
    stop_of = lambda e: e * (1 + STOP / 100)
    take_of = lambda e: e * (1 - TAKE / 100)
    t0 = t_open + H1
    for code, lvl in (("E1", poc - TICKS * tick), ("E2", hi)):
        for wn, wv in WAITS.items():
            res, R, wh = ST.trade(mins, t0, r["close"], lvl, wv, stop_of, take_of)
            out[f"{code}|{wn}|result"] = res
            out[f"{code}|{wn}|R"] = R
    return out


def tstats(g, code, wn):
    res, R = g[f"{code}|{wn}|result"], g[f"{code}|{wn}|R"]
    closed = res.isin(["тейк", "стоп", "по времени"])
    tk, st = (res == "тейк").sum(), (res == "стоп").sum()
    return {"сделок": int(closed.sum()), "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
            "в среднем R": f"{R[closed].mean():+.2f}" if closed.any() else "-"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    with_cap = [s for s in syms if Hst.base_asset(s) in caps]
    print(f"монет: {len(with_cap)}")

    def work(sym):
        try:
            return coin(sym, caps[Hst.base_asset(sym)])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(8) as ex:
        a = pd.concat([p for p in ex.map(work, with_cap) if p is not None and len(p)], ignore_index=True)
    print(f"сигналов (объём ≥ x3, дельта ≥ +17%): {len(a)}")
    with ThreadPoolExecutor(THREADS) as ex:
        extra = pd.DataFrame(list(ex.map(poc_and_trades, a.to_dict("records"))))
    a = pd.concat([a, extra], axis=1)
    a["теория"] = (a.poc_pos >= 70) & (a.close < a.poc)
    a.insert(1, "candle_open_kyiv", pd.to_datetime(a.open_time, unit="ms", utc=True)
             .dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M"))
    a.drop(columns=["open_time"]).to_csv(f"{OUT_DIR}/h1_short_signals.csv", index=False)

    days = 22 * 30.4
    rows = []
    for x in (3, 5, 10):
        for name, m in (("теория", a["теория"]), ("остальные", ~a["теория"] & a.poc.notna())):
            g = a[m & (a.vol_ratio >= x)]
            gc = g[g.complete]
            r = {"объём": f"≥ x{x}", "группа": name, "сигналов": len(g), "в месяц": f"{len(g) / days * 30.4:.1f}",
                 "сначала −4%": f"{gc.first_dn.mean() * 100:.0f}%" if len(gc) else "-",
                 "сначала +4%": f"{gc.first_up.mean() * 100:.0f}%" if len(gc) else "-",
                 "−10% за 10д": f"{gc.dn10.mean() * 100:.0f}%" if len(gc) else "-",
                 "+10% за 10д": f"{gc.up10.mean() * 100:.0f}%" if len(gc) else "-"}
            rows.append(r)
    trows = []
    for x in (3, 5, 10):
        for name, m in (("теория", a["теория"]), ("остальные", ~a["теория"] & a.poc.notna())):
            g = a[m & (a.vol_ratio >= x)]
            for code, cname in (("E1", "вход POC − 5 тиков"), ("E2", "вход на максимуме")):
                for wn in WAITS:
                    trows.append({"объём": f"≥ x{x}", "группа": name, "вход": cname, "ожидание": wn,
                                  **tstats(g, code, wn)})
    with open(f"{OUT_DIR}/h1_short_summary.md", "w") as f:
        f.write("# Шортовая теория на 1h-свечах\n\n")
        f.write("Сигнал: объём ≥ x3/x5/x10 к медиане 48 предыдущих 1h-свечей, дельта ≥ +17%. "
                "«Теория»: POC в верхних 30% + закрытие ниже POC. «Остальные» — те же сигналы без условия по POC.\n\n")
        f.write("## Направление (10 дней, от закрытия свечи)\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
        f.write(f"## Сделки: стоп +{STOP:g}%, тейк −{TAKE:g}%. Безубыточный винрейт {STOP / (STOP + TAKE) * 100:.1f}%, "
                f"безубыточно = 0R (тейк = +{TAKE / STOP:.2f}R, стоп = −1R)\n\n")
        f.write(pd.DataFrame(trows).to_markdown(index=False) + "\n")
    print(open(f"{OUT_DIR}/h1_short_summary.md").read())


if __name__ == "__main__":
    main()
