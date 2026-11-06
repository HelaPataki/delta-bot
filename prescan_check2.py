"""
Проверка НОВОГО предварительного отбора шортов живого бота на 379 исторических сигналах.

За 45 мин до закрытия бот смотрит прогнозную ставку фандинга (её Binance считает из поминутного индекса премии)
и уже выплаченный фандинг за 24 ч. Кандидат: прогноз < 0 или выплаченный 24ч ≤ −0,1%; затем OI за 24ч ≥ +30%.
Прогноз на истории восстанавливается по формуле Binance из архива premiumIndexKlines (1m):
  P = взвешенное среднее индекса премии с начала периода (веса 1, 2, ..., n), ставка = P + clamp(I − P, ±0,05%).
Сначала формула сверяется с настоящими выплатами, потом считается охват.
Интервал фандинга — как у бота: из прошлого месяца, нет файла — 4 ч.
"""

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
from prescan_check import zip_csv, funding_file, oi_day, month_of, prev_month, asof

OUT_DIR = "results/prescan_check2"
H = 3600 * 1000
H4 = 4 * H
BEFORE = 45 * 60 * 1000


def premium(sym, day):
    df = zip_csv(f"{Hst.ARCHIVE}/data/futures/um/daily/premiumIndexKlines/{sym}/1m/{sym}-1m-{day:%Y-%m-%d}.zip")
    if df is None:
        return None
    if isinstance(df.iloc[0, 0], str):
        df = df.iloc[1:]
    df = df.iloc[:, :5].astype(float)
    df.columns = ["t", "o", "h", "l", "c"]
    return df


def predicted(prem, start, upto, hours):
    w = prem[(prem.t >= start) & (prem.t < upto)].sort_values("t")
    if len(w) < 5:
        return np.nan
    wts = np.arange(1, len(w) + 1)
    P = float(np.sum(w.c.to_numpy() * wts) / wts.sum())
    I = 0.0001 * hours / 8
    return P + float(np.clip(I - P, -0.0005, 0.0005))


def days_between(a, b):
    d, out = dt.datetime.fromtimestamp(a / 1000, dt.timezone.utc).date(), []
    while d <= dt.datetime.fromtimestamp(b / 1000, dt.timezone.utc).date():
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def one(sym, T):
    X = T - BEFORE
    m = month_of(T)
    pm, cur, prev2 = funding_file(sym, prev_month(m)), funding_file(sym, m), funding_file(sym, prev_month(prev_month(m)))
    f = pd.concat([p for p in (prev2, pm, cur) if p is not None]) if any(p is not None for p in (prev2, pm, cur)) else None
    r = {"symbol": sym, "T": T}
    if f is None:
        r["нет данных"] = True
        return r
    f = f.sort_values("t")
    real_h = f[f.t <= T].hours.iloc[-1]
    bot_h = pm.hours.iloc[-1] if pm is not None and len(pm) else 4.0
    step = int(real_h * H)
    s0 = (X // step) * step                                         # начало текущего периода фандинга
    prem = [premium(sym, d) for d in days_between(s0 - step, T)]
    prem = pd.concat([p for p in prem if p is not None]) if any(p is not None for p in prem) else None
    oi = [oi_day(sym, d) for d in days_between(X - 25 * H, X)]
    oi = pd.concat([o for o in oi if o is not None]).sort_values("t") if any(o is not None for o in oi) else None
    r.update({"интервал настоящий": real_h, "интервал у бота": bot_h})
    if prem is None:
        r["нет премии"] = True
        return r
    # сверка формулы: прошлый полный период против настоящей выплаты в его конце
    paid = f[(f.t >= s0 - 60_000) & (f.t <= s0 + 60_000)]
    if len(paid):
        r["формула прошлый период %"] = predicted(prem, s0 - step, s0, real_h) * 100
        r["выплата прошлый период %"] = paid.rate.iloc[0] * 100
    pred = predicted(prem, s0, X, real_h)
    r["прогноз за 45 мин (на 8ч) %"] = pred * 8 / bot_h * 100
    win = f[(f.t > X - 24 * H) & (f.t <= X)]
    r["выплачено 24ч (на 8ч) %"] = (win.rate * 8 / bot_h).mean() * 100 if len(win) else np.nan
    r["OI 24ч за 45 мин %"] = (asof(oi, "oi", X, 30 * 60 * 1000) / asof(oi, "oi", X - 24 * H) - 1) * 100 if oi is not None else np.nan
    fund_ok = r["прогноз за 45 мин (на 8ч) %"] < 0 or r["выплачено 24ч (на 8ч) %"] <= -0.1
    oi_ok = not (r["OI 24ч за 45 мин %"] < 30)
    r["по фандингу"] = bool(fund_ok)
    r["попал в отбор"] = bool(fund_ok and oi_ok)
    return r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sh = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz")[["symbol", "open_time"]].drop_duplicates()
    with ThreadPoolExecutor(12) as ex:
        S = pd.DataFrame(list(ex.map(lambda r: one(r.symbol, int(r.open_time) + H4), sh.itertuples())))
    S["сигнал (Киев)"] = pd.to_datetime(S["T"], unit="ms", utc=True).dt.tz_convert("Europe/Kyiv").dt.strftime("%Y-%m-%d %H:%M")
    S.to_csv(f"{OUT_DIR}/short_check2.csv", index=False)

    bad = S.get("нет данных", pd.Series(np.nan, index=S.index)).eq(True) | S.get("нет премии", pd.Series(np.nan, index=S.index)).eq(True)
    ok = S[~bad]
    v = ok.dropna(subset=["формула прошлый период %", "выплата прошлый период %"])
    sign = (np.sign(v["формула прошлый период %"]) == np.sign(v["выплата прошлый период %"])).mean() * 100
    corr = v["формула прошлый период %"].corr(v["выплата прошлый период %"])
    out = ["# Проверка нового отбора шортов (за 45 мин до закрытия)\n",
           f"Сигналов: {len(S)}, с данными: {len(ok)}.\n",
           "## Насколько формула прогноза совпадает с настоящими выплатами\n",
           f"- сверок: {len(v)}, знак совпал: **{sign:.0f}%**, корреляция: {corr:.2f}",
           f"- медианная разница: {(v['формула прошлый период %'] - v['выплата прошлый период %']).abs().median():.4f}%\n",
           "## Охват\n",
           f"- прошли по фандингу: **{int(ok['по фандингу'].sum())} из {len(ok)}**",
           f"- прошли целиком (фандинг + OI ≥ +30%): **{int(ok['попал в отбор'].sum())} из {len(ok)}**\n",
           "Подробно — short_check2.csv.\n"]
    open(f"{OUT_DIR}/prescan2_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
