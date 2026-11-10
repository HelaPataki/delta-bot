"""
Разбор одной монеты (не бот): что происходило за 30 дней до движения и во время него.

Данные:
  Coinalyze (до текущего часа): 1h-свечи с объёмом покупок (дельта), OI в $, фандинг (факт и прогноз),
    лонг/шорт по аккаунтам (толпа).
  Архив Binance metrics (до вчера, 5 мин): лонг/шорт топ-трейдеров по аккаунтам и по позициям,
    толпа по аккаунтам, соотношение тейкер-покупок к продажам.
  Архив Binance спот (до вчера, 1h): объём и дельта на споте.
Движение находится само: максимум за последние 21 день и минимум перед ним (в 10 днях до пика) = старт.
Окно разбора: 30 дней до старта + само движение до сейчас.
Проверка наших сигналов на этой монете: коробка 4h (+ фандинг), толпа в шортах + OI + дельта,
аномалия объёма 4h, лонг дельта-бота (x5 + дельта ≤ −17%, без POC), шорт дельта-бота.
Время — по Киеву, свеча подписана временем открытия (как на TradingView).
"""

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

import history as Hst
import oi_funding as OF

SYM = os.environ.get("SYM", "INITUSDT").upper()
CZS = f"{SYM}_PERP.A"
KEY = os.environ.get("COINALYZE_KEY", "")
OUT = "results/dissect"
H = 3600
KY = "Europe/Kyiv"
NOW = int(time.time()) // H * H
FROM = NOW - 75 * 86400


def cz(endpoint, **p):
    for _ in range(6):
        r = requests.get(f"https://api.coinalyze.net/v1/{endpoint}", headers={"api_key": KEY}, timeout=40,
                         params={"symbols": CZS, "interval": "1hour", "from": FROM, "to": NOW, **p})
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 30)) + 1)
            continue
        r.raise_for_status()
        d = r.json()
        return pd.DataFrame(d[0]["history"]) if d and d[0].get("history") else pd.DataFrame()
    raise RuntimeError(endpoint)


def archive_days(path_fmt, first):
    rows, d = [], first
    today = dt.datetime.now(dt.timezone.utc).date()
    while d < today:
        df = OF.get_zip_csv(path_fmt.format(d=d))
        if df is not None:
            rows.append(df)
        d += dt.timedelta(days=1)
    return pd.concat(rows) if rows else None


def kyiv(sec):
    t = pd.to_datetime(sec, unit="s", utc=True)
    return t.dt.tz_convert(KY) if isinstance(t, pd.Series) else t.tz_convert(KY)


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---------- Coinalyze, 1h ----------
    k = cz("ohlcv-history").rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    oi = cz("open-interest-history", convert_to_usd="true")[["t", "c"]].rename(columns={"c": "oi_usd"})
    fu = cz("funding-rate-history")[["t", "c"]].rename(columns={"c": "fund"})
    pf = cz("predicted-funding-rate-history")[["t", "c"]].rename(columns={"c": "pfund"})
    ls = cz("long-short-ratio-history")[["t", "r"]].rename(columns={"r": "ls_cz"})
    a = k.merge(oi, on="t", how="left").merge(fu, on="t", how="left").merge(pf, on="t", how="left").merge(ls, on="t", how="left")
    a = a.sort_values("t").reset_index(drop=True)
    a["usd"] = a.v * a.close
    a["dq"] = (2 * a.bv - a.v) * a.close            # дельта в $
    # интервал фандинга -> всё в % за 8 ч
    hours = 8.0
    m0 = (dt.date.today().replace(day=1) - dt.timedelta(days=1)).replace(day=1)
    ff = OF.get_zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{m0:%Y-%m}.zip")
    if ff is not None and ff.shape[1] >= 2:
        try:
            hours = float(ff.iloc[-1, 1])
        except ValueError:
            pass
    a["fund8"] = a.fund * 8 / hours
    a["pfund8"] = a.pfund * 8 / hours

    # ---------- архив metrics (топы) ----------
    first = (kyiv(FROM).date())
    met = archive_days(f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{{d:%Y-%m-%d}}.zip", first)
    if met is not None:
        met["t"] = pd.to_datetime(met.create_time, utc=True).astype("int64") // 10 ** 9 // H * H
        met = met.groupby("t").last().reset_index()
        cols = {"count_toptrader_long_short_ratio": "top_acc", "sum_toptrader_long_short_ratio": "top_pos",
                "count_long_short_ratio": "crowd_acc", "sum_taker_long_short_vol_ratio": "taker"}
        met = met.rename(columns=cols)[["t"] + [c for c in cols.values() if c in met.rename(columns=cols)]]
        a = a.merge(met, on="t", how="left")
    # ---------- спот ----------
    sp = archive_days(f"{Hst.ARCHIVE}/data/spot/daily/klines/{SYM}/1h/{SYM}-1h-{{d:%Y-%m-%d}}.zip", first)
    if sp is not None:
        sp = sp.iloc[:, [0, 7, 10]]
        sp.columns = ["t", "s_usd", "s_buy"]
        sp["t"] = sp.t.astype(float)
        sp["t"] = (np.where(sp.t > 1e14, sp.t / 1e6, sp.t / 1e3)).astype("int64")   # мкс или мс -> с
        sp["s_dq"] = 2 * sp.s_buy - sp.s_usd
        a = a.merge(sp[["t", "s_usd", "s_dq"]], on="t", how="left")
    a["kt"] = kyiv(a.t)

    # ---------- где движение ----------
    recent = a[a.t >= NOW - 21 * 86400]
    p = recent.high.idxmax()
    pre = a.loc[max(0, p - 240):p]
    s = pre.low.idxmin()
    start_t, peak_t = a.t[s], a.t[p]
    w = a[a.t >= start_t - 30 * 86400].copy()

    # ---------- дневная таблица ----------
    w["day"] = w.kt.dt.strftime("%m-%d")
    g = w.groupby("day", sort=False)
    med30 = w[w.t < start_t].groupby("day").usd.sum().median()
    d = pd.DataFrame({
        "закрытие": g.close.last(),
        "цена за день, %": (g.close.last() / g.open.first() - 1) * 100,
        "объём фьюч., $M": g.usd.sum() / 1e6,
        "объём к норме": g.usd.sum() / med30,
        "дельта фьюч., %": g.dq.sum() / g.usd.sum() * 100,
        "OI, $M": g.oi_usd.last() / 1e6,
        "фандинг (факт), % за 8ч": g.fund8.mean(),
        "фандинг (прогноз)": g.pfund8.last(),
        "толпа л/ш (Coinalyze)": g.ls_cz.last(),
    })
    d.insert(6, "OI за день, %", d["OI, $M"].pct_change() * 100)
    for col, name in (("crowd_acc", "толпа л/ш (Binance)"), ("top_acc", "топы л/ш (аккаунты)"),
                      ("top_pos", "топы л/ш (позиции)"), ("taker", "тейкеры покупки/продажи")):
        if col in w:
            d[name] = g[col].last() if col != "taker" else g[col].mean()
    if "s_usd" in w:
        d["объём спот, $M"] = g.s_usd.sum() / 1e6
        d["дельта спот, %"] = g.s_dq.sum() / g.s_usd.sum() * 100
        d["спот к фьюч."] = g.s_usd.sum() / g.usd.sum()
    start_day = kyiv(start_t).strftime("%m-%d")

    # ---------- 4h и наши сигналы ----------
    h4 = w.set_index("kt").resample("4h", origin="epoch", offset="0h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "usd": "sum", "dq": "sum",
         "oi_usd": "last", "fund8": "mean", "ls_cz": "last"}).dropna(subset=["close"])
    bt, bb = np.maximum(h4.open, h4.close), np.minimum(h4.open, h4.close)
    top, bot = bt.shift(1).rolling(60).max(), bb.shift(1).rolling(60).min()
    body = h4.close - h4.open
    part = (h4.close - np.maximum(h4.open, top)) / body.where(body > 0)
    fbox = h4.fund8.shift(1).rolling(60).mean()
    vx = h4.usd / h4.usd.shift(1).rolling(12).median()
    dpc = h4.dq / h4.usd * 100
    oi3 = (h4.oi_usd / h4.close) / (h4.oi_usd / h4.close).shift(18) - 1
    oi24 = (h4.oi_usd / h4.close) / (h4.oi_usd / h4.close).shift(6) - 1
    cvd7 = h4.dq.rolling(42).sum() / h4.usd.rolling(42).sum() * 100
    f24 = h4.fund8.rolling(6).mean()
    sig = []
    for ts in h4.index:
        r = []
        if part.get(ts, 0) >= 0.4 and body[ts] > 0:
            r.append(f"коробка: пробой (коробка {(top[ts] / bot[ts] - 1) * 100:.0f}%, фандинг за коробку "
                     f"{fbox[ts]:.4f}% — {'проходит' if fbox[ts] > 0.0116 else 'не проходит'} фильтр)")
        if h4.ls_cz[ts] <= 1 and oi3[ts] >= 0.10 and cvd7[ts] >= 0:
            r.append(f"толпа в шортах + OI + дельта (л/ш {h4.ls_cz[ts]:.2f}, OI 3д {oi3[ts] * 100:+.0f}%)")
        if vx[ts] >= 3:
            r.append(f"аномалия объёма x{vx[ts]:.1f}, дельта {dpc[ts]:+.0f}%")
        if vx[ts] >= 5 and dpc[ts] <= -17:
            r.append("лонг дельта-бота: x5 + дельта ≤ −17% (POC не проверялся)")
        if f24[ts] <= -0.2 and oi24[ts] >= 0.6:
            r.append("шорт дельта-бота: фандинг ≤ −0.2% + OI 24ч ≥ +60%")
        if r:
            sig.append((ts.strftime("%Y-%m-%d %H:%M"), "; ".join(r)))

    # ---------- 4h-таблица: 3 дня до старта и движение ----------
    win = h4[h4.index >= kyiv(start_t - 3 * 86400)]
    t4 = pd.DataFrame({"закрытие": win.close, "свеча, %": (win.close / win.open - 1) * 100,
                       "объём к норме": vx[win.index], "дельта, %": dpc[win.index],
                       "OI, $M": win.oi_usd / 1e6, "OI за свечу, %": win.oi_usd.pct_change() * 100,
                       "фандинг % за 8ч": win.fund8, "толпа л/ш": win.ls_cz})
    t4.index = t4.index.strftime("%m-%d %H:%M")

    # ---------- сводка ----------
    b30 = w[w.t < start_t]
    first_d, last_d = b30.iloc[: 24 * 7], b30.iloc[-24 * 7:]

    def ch(a_, b_):
        return f"{(b_ / a_ - 1) * 100:+.0f}%"

    lines = [
        f"# Разбор {SYM}: 30 дней до движения и само движение",
        "",
        f"Данные до {kyiv(NOW).strftime('%d.%m %H:%M')} (Киев). Топ-трейдеры и спот — из архива Binance, до вчерашнего дня.",
        "",
        f"**Движение:** старт {kyiv(start_t).strftime('%d.%m %H:%M')} от {a.low[s]:.5g}, "
        f"пик {kyiv(peak_t).strftime('%d.%m %H:%M')} — {a.high[p]:.5g} (**{(a.high[p] / a.low[s] - 1) * 100:+.0f}%**), "
        f"сейчас {a.close.iloc[-1]:.5g} ({(a.close.iloc[-1] / a.low[s] - 1) * 100:+.0f}% от старта).",
        "",
        "## 30 дней до старта: первая неделя против последней",
        "",
        "| показатель | первая неделя | последняя неделя перед стартом | изменение |",
        "|---|---|---|---|",
        f"| цена (закрытие) | {first_d.close.iloc[-1]:.5g} | {last_d.close.iloc[-1]:.5g} | {ch(first_d.close.iloc[-1], last_d.close.iloc[-1])} |",
        f"| диапазон цены за неделю | {(first_d.high.max() / first_d.low.min() - 1) * 100:.0f}% | {(last_d.high.max() / last_d.low.min() - 1) * 100:.0f}% | |",
        f"| объём фьюч. в день, $M | {first_d.usd.sum() / 7e6:.2f} | {last_d.usd.sum() / 7e6:.2f} | {ch(first_d.usd.sum(), last_d.usd.sum())} |",
        f"| дельта фьюч., % | {first_d.dq.sum() / first_d.usd.sum() * 100:+.1f} | {last_d.dq.sum() / last_d.usd.sum() * 100:+.1f} | |",
        f"| OI, $M (конец недели) | {first_d.oi_usd.iloc[-1] / 1e6:.2f} | {last_d.oi_usd.iloc[-1] / 1e6:.2f} | {ch(first_d.oi_usd.iloc[-1], last_d.oi_usd.iloc[-1])} |",
        f"| фандинг средний, % за 8ч | {first_d.fund8.mean():.4f} | {last_d.fund8.mean():.4f} | |",
        f"| толпа л/ш | {first_d.ls_cz.iloc[-1]:.2f} | {last_d.ls_cz.iloc[-1]:.2f} | |",
    ]
    for col, name in (("top_acc", "топы л/ш (аккаунты)"), ("top_pos", "топы л/ш (позиции)")):
        if col in b30 and b30[col].notna().any():
            lines.append(f"| {name} | {first_d[col].dropna().iloc[-1]:.2f} | {last_d[col].dropna().iloc[-1]:.2f} | |")
    if "s_usd" in b30:
        lines.append(f"| спот: объём в день, $M | {first_d.s_usd.sum() / 7e6:.2f} | {last_d.s_usd.sum() / 7e6:.2f} | "
                     f"{ch(first_d.s_usd.sum(), last_d.s_usd.sum())} |")
        lines.append(f"| спот: дельта, % | {first_d.s_dq.sum() / first_d.s_usd.sum() * 100:+.1f} | "
                     f"{last_d.s_dq.sum() / last_d.s_usd.sum() * 100:+.1f} | |")
    lines += ["", f"## По дням (старт движения — {start_day}, строка выделена)", "",
              d.round(3).rename(index=lambda x: f"**{x}**" if x == start_day else x).to_markdown(), "",
              "## 4h-свечи: 3 дня до старта и движение", "", t4.round(3).to_markdown(), "",
              "## Наши сигналы на этой монете (4h, весь период разбора)", ""]
    lines += [f"- {t}: {r}" for t, r in sig] or ["Ни один сигнал не сработал."]
    open(f"{OUT}/{SYM}.md", "w").write("\n".join(lines) + "\n")
    a.drop(columns=["kt"]).to_csv(f"{OUT}/{SYM}_1h.csv.gz", index=False)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
