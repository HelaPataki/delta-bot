"""
ЗОЛОТО и ИНДЕКСЫ — четыре идеи, только история. Живых ботов не трогает.

1. НОВОСТИ (золото, фьючерс CME GC=F, 1h-свечи за ~2 года, время Нью-Йорка):
   NFP — первая пятница месяца, свеча 08:00–09:00 NY (отчёт в 08:30), берём только если её объём
     ≥ x2 от медианы этого часа за 20 дней (так отсеиваются перенесённые отчёты);
   «другие новости 08:30» — любой будний день, где свеча 08:00 NY ≥ x3 от нормы (это CPI, PPI, продажи и т. п.);
   ФРС — дни решений, свеча 14:00–15:00 NY.
   Смотрим: продолжает ли цена движение новостной свечи (через 1ч, 3ч, к 16:00 NY, через 1 и 3 дня),
   и сделки «по ходу новости» / «против новости»: вход на закрытии новостной свечи, стоп/тейк фиксированные.
2. ПРОБОЙ СЕССИИ (золото, 1h): диапазон ночи, затем первое часовое закрытие за его границей → вход.
   A «Лондон»: диапазон 18:00–03:00 NY (Киев 01:00–10:00 зимой), вход до 11:00 NY;
   B «Нью-Йорк»: диапазон 18:00–08:00 NY (Киев 01:00–15:00 зимой), вход до 12:00 NY.
   Стоп — противоположная граница диапазона, тейк — 1 / 1,5 / 2 ширины диапазона, выход в 16:00 NY.
   Дни с очень узким (<0,25%) или широким (>2%) диапазоном пропускаем.
3. ТРЕНД (дневные свечи): GLD (фонд на золото, без перекатов контрактов) с 2005 года, для сравнения SPY и QQQ.
   «выше 200-дневной средней — лонг», «пробой 20-дневного максимума/минимума (выход по 10-дневному)»,
   «пересечение средних 50/200». Сравниваем с «купил и держу». Без плеча.
4. ВСПЛЕСКИ ОБЪЁМА НА ИНДЕКСАХ: фьючерсы ES=F (S&P 500) и NQ=F (Nasdaq 100), как gold_study:
   4h за ~2 года и дневные с 2010. Перекаты контрактов индексов — март/июнь/сентябрь/декабрь.

Комиссия+спред: внутри дня 0,05% на сделку (золото), в трендовых — 0,1% на каждый вход/выход.
Если стоп и тейк задеты в одной часовой свече — считаем стоп (худший вариант).
"""

import os

import numpy as np
import pandas as pd

import gold_study as G

OUT = os.environ.get("OUT_DIR", "results/gold_ideas")
NY = "America/New_York"
KYIV = "Europe/Kyiv"
COST_IN = 0.05
COST_TR = 0.10

FOMC = """2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09""".split()


def table(rows):
    return G.table(rows) if rows else "(нет данных)"


# ---------------- общие ----------------
def sim(h, i, side, stop, take, exit_i):
    """Вход на закрытии свечи i. side=+1 лонг, −1 шорт. Возвращает (результат %, исход, свечей)."""
    c0 = h.close.iat[i]
    hi, lo = h.high.to_numpy(), h.low.to_numpy()
    for j in range(i + 1, min(exit_i, len(h) - 1) + 1):
        if side > 0:
            if lo[j] <= stop:
                return (stop / c0 - 1) * 100 - COST_IN, "стоп", j - i
            if hi[j] >= take:
                return (take / c0 - 1) * 100 - COST_IN, "тейк", j - i
        else:
            if hi[j] >= stop:
                return (1 - stop / c0) * 100 - COST_IN, "стоп", j - i
            if lo[j] <= take:
                return (1 - take / c0) * 100 - COST_IN, "тейк", j - i
    j = min(exit_i, len(h) - 1)
    return side * (h.close.iat[j] / c0 - 1) * 100 - COST_IN, "время", j - i


def trade_stats(tr, name):
    if not len(tr):
        return {"вариант": name, "сделок": 0}
    r = tr.res.to_numpy()
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    streak = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        streak = max(streak, cur)
    half = len(tr) // 2
    return {"вариант": name, "сделок": len(tr),
            "винрейт %": round((r > 0).mean() * 100),
            "средняя %": round(r.mean(), 3),
            "сумма %": round(r.sum(), 1),
            "1-я половина, сумма %": round(r[:half].sum(), 1),
            "2-я половина, сумма %": round(r[half:].sum(), 1),
            "макс. просадка (сумма %)": round(dd, 1),
            "серия минусов": streak,
            "тейк/стоп/время": f"{(tr.how == 'тейк').sum()}/{(tr.how == 'стоп').sum()}/{(tr.how == 'время').sum()}"}


def hour_norm(h, days=20):
    """Медиана объёма в тот же час NY за `days` предыдущих торговых дней."""
    hr = h.index.hour
    out = pd.Series(np.nan, index=h.index)
    for x in np.unique(hr):
        m = hr == x
        out[m] = h.volume[m].shift(1).rolling(days, min_periods=10).median()
    return out


def idx_at(h, ts):
    k = h.index.searchsorted(ts, side="right") - 1
    return max(k, 0)


# ---------------- 1. новости ----------------
def news(h):
    h = h.copy()
    h.index = h.index.tz_convert(NY)
    ratio = h.volume / hour_norm(h)
    events = []
    dates = pd.Series(h.index.normalize().unique())
    for d in dates:
        day = d.date()
        is_nfp = d.dayofweek == 4 and d.day <= 7
        is_fomc = str(day) in FOMC
        for hour, kind in ((8, "nfp" if is_nfp else "other"), (14, "fomc" if is_fomc else None)):
            if kind is None:
                continue
            t = d + pd.Timedelta(hours=hour)
            if t not in h.index:
                continue
            r = ratio.loc[t]
            if r != r:
                continue
            if kind == "nfp" and r < 2:
                continue
            if kind == "other" and r < 3:
                continue
            events.append((t, kind, r))
    rows, trades = [], []
    names = {"nfp": "NFP (рынок труда США)", "other": "другие новости 08:30 NY (CPI и т. п.)", "fomc": "решение ФРС"}
    for t, kind, r in events:
        i = h.index.get_loc(t)
        o, c = h.open.iat[i], h.close.iat[i]
        mv = (c / o - 1) * 100
        sgn = 1 if mv >= 0 else -1
        end_day = idx_at(h, t.normalize() + pd.Timedelta(hours=16))
        row = {"kind": kind, "t": t, "mv": mv, "ratio": r}
        for lab, k in (("1ч", i + 1), ("3ч", i + 3), ("16:00", end_day),
                       ("1д", idx_at(h, t + pd.Timedelta(days=1))), ("3д", idx_at(h, t + pd.Timedelta(days=3)))):
            k = min(k, len(h) - 1)
            row[lab] = sgn * (h.close.iat[k] / c - 1) * 100 if k > i else np.nan
        rows.append(row)
        # сделки: по ходу / против, выход в конце следующего дня
        ex = idx_at(h, t.normalize() + pd.Timedelta(days=1, hours=16))
        for st, tk in ((0.5, 1.0), (0.7, 1.4), (1.0, 2.0)):
            for mode, side in (("по ходу", sgn), ("против", -sgn)):
                stop = c * (1 - side * st / 100)
                take = c * (1 + side * tk / 100)
                res, how, n = sim(h, i, side, stop, take, ex)
                trades.append({"kind": kind, "вариант": f"{names[kind]}: {mode}, стоп {st}% / тейк {tk}%",
                               "t": t, "res": res, "how": how})
    ev = pd.DataFrame(rows)
    tr = pd.DataFrame(trades)
    out = ["## 1. Новости — золото (GC=F, 1h)",
           "«Продолжение» — изменение цены ПО НАПРАВЛЕНИЮ новостной свечи от её закрытия "
           "(плюс = цена пошла дальше туда же, минус = развернулась).", ""]
    srows = []
    for kind in ("nfp", "other", "fomc"):
        e = ev[ev.kind == kind] if len(ev) else ev
        if not len(e):
            continue
        r = {"новость": names[kind], "случаев": len(e),
             "ход новостной свечи, медиана |%|": round(e.mv.abs().median(), 2)}
        for lab in ("1ч", "3ч", "16:00", "1д", "3д"):
            r[f"{lab}: сред. %"] = round(e[lab].mean(), 2)
            r[f"{lab}: продолжение, % случаев"] = round((e[lab] > 0).mean() * 100)
        srows.append(r)
    out.append(table(srows))
    out.append("")
    out.append("### Сделки после новостей (вход на закрытии новостной свечи, выход не позже 16:00 NY следующего дня)")
    out.append(table([trade_stats(g.sort_values("t"), v) for v, g in tr.groupby("вариант", sort=False)] if len(tr) else []))
    out.append("")
    if len(ev):
        ev = ev.copy()
        ev["свеча (Киев)"] = ev.t.dt.tz_convert(KYIV).dt.strftime("%Y-%m-%d %H:%M")
    return "\n".join(out), ev, tr


# ---------------- 2. пробой сессии ----------------
def breakout(h):
    h = h.copy()
    h.index = h.index.tz_convert(NY)
    trades = []
    days = h.index.normalize().unique()
    for d in days:
        if d.dayofweek >= 5:
            continue
        start = d - pd.Timedelta(hours=6)  # 18:00 NY предыдущего дня
        if d.dayofweek == 0:
            start = d - pd.Timedelta(days=2, hours=6)  # воскресенье 18:00
        for name, r_end, last_entry in (("A «Лондон»", 3, 11), ("B «Нью-Йорк»", 8, 12)):
            rng = h[(h.index >= start) & (h.index < d + pd.Timedelta(hours=r_end))]
            if len(rng) < 5:
                continue
            top, bot = rng.high.max(), rng.low.min()
            width = (top / bot - 1) * 100
            if width < 0.25 or width > 2:
                continue
            win = h[(h.index >= d + pd.Timedelta(hours=r_end)) & (h.index < d + pd.Timedelta(hours=last_entry))]
            side = 0
            for t, bar in win.iterrows():
                if bar.close > top:
                    side = 1
                elif bar.close < bot:
                    side = -1
                if side:
                    break
            if not side:
                continue
            i = h.index.get_loc(t)
            ex = idx_at(h, d + pd.Timedelta(hours=16))
            if ex <= i:
                continue
            c = h.close.iat[i]
            for mult in (1, 1.5, 2):
                stop = bot if side > 0 else top
                take = c + side * mult * (top - bot)
                res, how, n = sim(h, i, side, stop, take, ex)
                trades.append({"вариант": f"{name}, тейк {mult}× диапазона", "t": t, "side": side,
                               "width": width, "res": res, "how": how})
    tr = pd.DataFrame(trades)
    out = ["## 2. Пробой сессии — золото (GC=F, 1h)",
           "Стоп — противоположная граница ночного диапазона, выход не позже 16:00 NY (Киев 23:00 зимой).", ""]
    rows = []
    if len(tr):
        for v, g in tr.groupby("вариант", sort=False):
            g = g.sort_values("t")
            rows.append(trade_stats(g, v))
            rows.append(trade_stats(g[g.side > 0], "   ↳ только лонги"))
            rows.append(trade_stats(g[g.side < 0], "   ↳ только шорты"))
        out.append(f"Средняя ширина диапазона: A {tr[tr.вариант.str.startswith('A')].width.median():.2f}%, "
                   f"B {tr[tr.вариант.str.startswith('B')].width.median():.2f}%.")
    out.append(table(rows))
    out.append("")
    return "\n".join(out), tr


# ---------------- 3. тренд ----------------
def trend_one(d, ticker):
    c = d.close
    ret = c.pct_change().fillna(0)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi20, lo20 = c.rolling(20).max().shift(1), c.rolling(20).min().shift(1)
    hi10, lo10 = c.rolling(10).max().shift(1), c.rolling(10).min().shift(1)
    don = np.zeros(len(c))
    pos = 0
    for k in range(len(c)):
        x = c.iat[k]
        if pos == 0:
            if x > hi20.iat[k]:
                pos = 1
            elif x < lo20.iat[k]:
                pos = -1
        elif pos == 1 and x < lo10.iat[k]:
            pos = -1 if x < lo20.iat[k] else 0
        elif pos == -1 and x > hi10.iat[k]:
            pos = 1 if x > hi20.iat[k] else 0
        don[k] = pos
    don = pd.Series(don, index=c.index)
    rules = {
        "купил и держу": pd.Series(1.0, index=c.index),
        "выше 200-дн. средней → лонг, иначе вне рынка": (c > sma200).astype(float).where(sma200.notna(), 0),
        "200-дн. средняя: лонг выше / шорт ниже": np.sign(c - sma200).where(sma200.notna(), 0),
        "пробой 20 дней (лонг и шорт)": don,
        "пробой 20 дней (только лонг)": don.clip(lower=0),
        "средние 50/200: лонг выше / шорт ниже": np.sign(sma50 - sma200).where(sma200.notna(), 0),
    }
    rows = []
    for name, p in rules.items():
        p = p.shift(1).fillna(0)  # решение на закрытии, позиция со следующего дня
        cost = p.diff().abs().fillna(0) * COST_TR / 100
        r = p * ret - cost
        for per, m in (("весь период", slice(None)), ("до 2015", slice(None, "2014-12-31")),
                       ("2015–2026", slice("2015-01-01", None))):
            rr = r.loc[m]
            if len(rr) < 250:
                continue
            eq = (1 + rr).cumprod()
            yrs = len(rr) / 252
            cagr = (eq.iat[-1] ** (1 / yrs) - 1) * 100
            dd = (eq / eq.cummax() - 1).min() * 100
            pp = p.loc[m]
            rows.append({"инструмент": ticker, "правило": name, "период": per,
                         "доходность в год %": round(cagr, 1), "макс. просадка %": round(dd, 1),
                         "в рынке, % дней": round((pp != 0).mean() * 100),
                         "входов/выходов в год": round((pp.diff().abs() > 0).sum() / yrs, 1)})
    return rows


def trend():
    rows = []
    for tk in ("GLD", "SPY", "QQQ"):
        d = G.yahoo("1d", "max", ticker=tk, adjust=True)  # с учётом дивидендов
        if d is None or len(d) < 500:
            rows.append({"инструмент": tk, "правило": "данные не загрузились"})
            continue
        d = d[d.index >= "2005-01-01"]
        rows += trend_one(d, tk)
    out = ["## 3. Тренд — дневные свечи (GLD = золото без перекатов контрактов; SPY, QQQ — для сравнения)",
           "Доходность в год — сложная, без плеча, с комиссией. Просадка — от пика капитала.", "",
           table(rows), ""]
    return "\n".join(out), pd.DataFrame(rows)


# ---------------- 4. индексы: всплески объёма ----------------
def roll_index(idx):
    """Перекат фьючерсов на индексы: 5–20 число марта/июня/сентября/декабря."""
    return np.isin(idx.month, [3, 6, 9, 12]) & (idx.day >= 5) & (idx.day <= 20)


def indices():
    parts, sig = [], []
    for tk, nm in (("ES=F", "S&P 500 (ES)"), ("NQ=F", "Nasdaq 100 (NQ)")):
        h1 = G.yahoo("1h", "730d", ticker=tk)
        if h1 is not None and len(h1) > 500:
            h4 = G.to_4h(h1)
            r12 = h4.volume / G.norm_prev(h4.volume, 12)
            rsl = h4.volume / G.norm_slot(h4)
            rules = [("12 свечей", r12, m) for m in G.MULTS_4H] + [("то же время", rsl, m) for m in G.MULTS_4H]
            txt, sg = G.analyse(h4, 4 * G.H, f"4. {nm}, 4h (≈2 года)", rules, roll=True, roll_fn=roll_index)
            parts.append(txt)
            sig += sg
        d1 = G.yahoo("1d", "max", ticker=tk)
        if d1 is not None and len(d1) > 300:
            d1 = d1[d1.index >= "2010-01-01"].copy()
            rd = d1.volume / G.norm_prev(d1.volume, 20)
            rules = [("20 дней", rd, m) for m in G.MULTS_1D]
            txt, sg = G.analyse(d1, 24 * G.H, f"4. {nm}, дневные с 2010", rules, roll=True, roll_fn=roll_index)
            parts.append(txt)
            sig += sg
    return "\n".join(parts), pd.DataFrame(sig)


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = ["# Золото и индексы — четыре идеи", "",
             "Время в таблицах сделок — Киев. «Сумма %» — простая сумма результатов сделок без плеча "
             "(с плечом x2 умножай на 2). Половины — первая и вторая половина сделок по времени: "
             "если результат держится в обеих, это хороший знак.", ""]
    h1 = G.yahoo("1h", "730d")
    if h1 is not None and len(h1) > 500:
        txt, ev, tr = news(h1)
        parts.append(txt)
        txt2, tr2 = breakout(h1)
        parts.append(txt2)
        for df, nm in ((ev, "news_events"), (tr, "news_trades"), (tr2, "breakout_trades")):
            if len(df):
                df = df.copy()
                if "t" in df:
                    df["t"] = pd.to_datetime(df.t, utc=True).dt.tz_convert(KYIV).dt.strftime("%Y-%m-%d %H:%M")
                df.round(3).to_csv(f"{OUT}/{nm}.csv.gz", index=False)
    else:
        parts.append("## 1–2. Часовые данные золота не загрузились\n")
    txt, tdf = trend()
    parts.append(txt)
    txt, sdf = indices()
    parts.append(txt)
    if len(sdf):
        sdf.to_csv(f"{OUT}/index_signals.csv.gz", index=False)
    open(f"{OUT}/ideas_summary.md", "w").write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
