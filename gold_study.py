"""
ЗОЛОТО: есть ли смысл в аномалиях объёма (как в volume-bot / delta-bot), но на золоте.
Только история. Живых ботов не трогает.

Данные:
  A) Фьючерс на золото CME (GC=F, Yahoo Finance) — «настоящий» рынок золота с реальным объёмом.
     1h-свечи за ~2 последних года, собранные в 4h (UTC 00/04/08/12/16/20 = Киев 02/06/10/14/18/22 зимой),
     плюс дневные свечи с 2010 года.
  B) Перпетуал Binance XAUUSDT (архив data.binance.vision) — там есть дельта (taker buy/sell),
     как в твоих крипто-ботах. История короче — сколько есть.

СИГНАЛ: объём свечи ≥ x3 / x5 / x10 от нормы. Две нормы:
  «12 свечей» — среднее 12 предыдущих 4h-свечей (как в delta-bot);
  «то же время» — медиана объёма в тот же 4h-слот за 20 предыдущих торговых дней.
     Нужна, потому что у золота объём сильно зависит от сессии: открытие Нью-Йорка всегда
     в разы больше азиатской ночи, и правило «12 свечей» будет срабатывать почти каждый день.
  Дневные свечи: объём ≥ x2 / x3 от среднего 20 предыдущих дней.
  Повторы: после сигнала следующий по тому же правилу — не раньше чем через 24 часа.

ИСХОД (вперёд от закрытия сигнальной свечи, 10 календарных дней):
  макс. рост и макс. падение (по хаям/лоям), что из ±2% / ±3% / ±5% случилось раньше и на какой день,
  изменение цены через 1 / 3 / 10 дней.
БАЗА для сравнения — те же цифры для ВСЕХ свечей подряд. Если после сигналов движения
  не больше, чем после любой свечи, — в сигнале ничего нет.
Направление: отдельно для зелёной и красной сигнальной свечи (продолжение или разворот),
  а на Binance ещё и по дельте ≥ +17% / ≤ −17%.

Переходы контрактов CME (конец янв/мар/май/июл/сен/ноя) дают ложные всплески объёма
  у непрерывного GC=F — такие сигналы помечены и показаны отдельно.
"""

import datetime as dt
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

OUT = os.environ.get("OUT_DIR", "results/gold_study")
FWD_DAYS = 10
LEVELS = [2, 3, 5]
MULTS_4H = [3, 5, 10]
MULTS_1D = [2, 3]
COOLDOWN_H = 24
DELTA_TH = 17
KYIV = "Europe/Kyiv"
H = pd.Timedelta(hours=1)


# ---------------- данные ----------------
def yahoo(interval, period):
    import yfinance as yf
    for i in range(4):
        try:
            df = yf.Ticker("GC=F").history(period=period, interval=interval, auto_adjust=False)
            if len(df):
                break
        except Exception as e:  # noqa: BLE001
            print("yahoo error", e, file=sys.stderr)
        time.sleep(10 * (i + 1))
    else:
        return None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[(df.volume > 0) & df.close.notna()]
    return df


def to_4h(h1):
    """1h → 4h по UTC (как свечи Binance). Неполные слоты (перерыв, выходные) остаются как есть."""
    g = h1.resample("4h", origin="epoch", label="left", closed="left")
    df = pd.DataFrame({
        "open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
        "close": g.close.last(), "volume": g.volume.sum(), "hours": g.close.count(),
    })
    return df[df.hours > 0]


def binance_xau():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "delta-bot-gold"
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    frames = []
    today = dt.datetime.now(dt.timezone.utc).date()
    m = dt.date(2024, 1, 1)
    base = "https://data.binance.vision/data/futures/um"

    def read(content):
        with zipfile.ZipFile(io.BytesIO(content)) as z, z.open(z.namelist()[0]) as f:
            d = pd.read_csv(f, header=None)
        if isinstance(d.iloc[0, 0], str):
            d = d.iloc[1:]
        d = d.iloc[:, :12]
        d.columns = cols
        return d.astype(float)

    def fetch(url):
        for i in range(3):
            try:
                r = s.get(url, timeout=60)
                return r.content if r.status_code == 200 else None
            except Exception:  # noqa: BLE001
                time.sleep(3 * (i + 1))
        return None

    while m <= today:
        cur = (m.year, m.month) == (today.year, today.month)
        if not cur:
            c = fetch(f"{base}/monthly/klines/XAUUSDT/4h/XAUUSDT-4h-{m:%Y-%m}.zip")
            if c:
                frames.append(read(c))
        else:
            d = m
            while d < today:
                c = fetch(f"{base}/daily/klines/XAUUSDT/4h/XAUUSDT-4h-{d:%Y-%m-%d}.zip")
                if c:
                    frames.append(read(c))
                d += dt.timedelta(days=1)
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    if not frames:
        return None
    k = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
    k.index = pd.to_datetime(k.open_time.astype("int64"), unit="ms", utc=True)
    df = k[["open", "high", "low", "close", "quote_volume"]].rename(columns={"quote_volume": "volume"})
    v = k.volume.replace(0, np.nan)
    df["delta_pct"] = ((2 * k.taker_buy_volume - k.volume) / v * 100).round(1)
    df["hours"] = 4
    return df[df.volume > 0]


# ---------------- сигналы ----------------
def norm_prev(vol, n):
    return vol.shift(1).rolling(n, min_periods=n).mean()


def norm_slot(df, days=20):
    """Медиана объёма в том же 4h-слоте (час UTC) за `days` предыдущих торговых дней."""
    slot = df.index.hour
    out = pd.Series(np.nan, index=df.index)
    for hr in np.unique(slot):
        m = slot == hr
        v = df.volume[m]
        out[m] = v.shift(1).rolling(days, min_periods=days).median()
    return out


def roll_flag(idx):
    """Окно перехода контракта CME: 18-е число и позже в янв/мар/май/июл/сен/ноя + первые 3 дня следующего месяца."""
    mo, d = idx.month, idx.day
    odd = np.isin(mo, [1, 3, 5, 7, 9, 11]) & (d >= 18)
    nxt = np.isin(mo, [2, 4, 6, 8, 10, 12]) & (d <= 3)
    return odd | nxt


def pick(df, ratio, mult):
    """Индексы сигналов с антиповтором 24 часа."""
    cand = df.index[(ratio >= mult).to_numpy()]
    out, last = [], None
    for t in cand:
        if last is None or t - last >= pd.Timedelta(hours=COOLDOWN_H):
            out.append(t)
            last = t
    return out


# ---------------- исходы ----------------
def outcomes(df, bar):
    """Для каждой свечи: что было за FWD_DAYS дней после её закрытия."""
    t = df.index.tz_convert(None).to_numpy()
    close_t = t + np.timedelta64(int(bar / H), "h")
    hi, lo, cl = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    n = len(df)
    end = np.searchsorted(t, close_t + np.timedelta64(FWD_DAYS, "D"), side="left")
    full = t[-1] >= close_t + np.timedelta64(FWD_DAYS, "D") - np.timedelta64(4, "D")
    res = {k: np.full(n, np.nan) for k in
           ["up_max", "dn_max", "r1", "r3", "r10"] +
           [f"first{l}" for l in LEVELS] + [f"day{l}" for l in LEVELS]}
    for i in range(n - 1):
        j0, j1 = i + 1, end[i]
        if j1 <= j0 or not full[i]:
            continue
        c0 = cl[i]
        h = hi[j0:j1] / c0 * 100 - 100
        l_ = lo[j0:j1] / c0 * 100 - 100
        res["up_max"][i] = h.max()
        res["dn_max"][i] = l_.min()
        for nd, key in ((1, "r1"), (3, "r3"), (FWD_DAYS, "r10")):
            k = np.searchsorted(t, close_t[i] + np.timedelta64(nd, "D"), side="left") - 1
            if k > i:
                res[key][i] = cl[k] / c0 * 100 - 100
        tt = t[j0:j1]
        for lv in LEVELS:
            iu = np.argmax(h >= lv) if (h >= lv).any() else None
            idn = np.argmax(l_ <= -lv) if (l_ <= -lv).any() else None
            if iu is None and idn is None:
                res[f"first{lv}"][i] = 0
                continue
            if idn is None or (iu is not None and iu < idn):
                res[f"first{lv}"][i], k = 1, iu
            elif iu is None or idn < iu:
                res[f"first{lv}"][i], k = -1, idn
            else:  # в одной свече оба — неясно, считаем «обе»
                res[f"first{lv}"][i], k = 2, iu
            res[f"day{lv}"][i] = (tt[k] - close_t[i]) / np.timedelta64(1, "D")
    o = pd.DataFrame(res, index=df.index)
    o["green"] = (df.close >= df.open).to_numpy()
    o["body_pct"] = (df.close / df.open * 100 - 100).round(2).to_numpy()
    return o


def stats(o, name):
    o = o.dropna(subset=["up_max"])
    if not len(o):
        return {"группа": name, "n": 0}
    row = {"группа": name, "n": len(o),
           "макс.рост, медиана %": round(o.up_max.median(), 2),
           "макс.падение, медиана %": round(o.dn_max.median(), 2)}
    for lv in LEVELS:
        f = o[f"first{lv}"]
        row[f"±{lv}% за 10д, %"] = round((f != 0).mean() * 100)
        hit = o[f != 0]
        row[f"±{lv}%: вверх первым, %"] = round(((hit[f"first{lv}"] == 1).sum() / max(len(hit), 1)) * 100)
    row["день до ±3%, медиана"] = round(o.loc[o.first3 != 0, "day3"].median(), 1) if (o.first3 != 0).any() else None
    for key, nm in (("r1", "1д"), ("r3", "3д"), ("r10", "10д")):
        row[f"через {nm}, сред. %"] = round(o[key].mean(), 2)
    return row


def table(rows):
    df = pd.DataFrame(rows)
    try:
        return df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return df.to_string(index=False)


def analyse(df, bar, label, rules, roll=False, delta=False):
    """rules: список (название, ratio, mult). Возвращает текст раздела и список сигналов."""
    o = outcomes(df, bar)
    lines = [f"## {label}",
             f"Свечей: {len(df)}, период {df.index[0]:%Y-%m-%d} — {df.index[-1]:%Y-%m-%d}. "
             f"Обычный дневной размах цены (хай−лоу дня, медиана): "
             f"{(df.high.resample('1D').max() / df.low.resample('1D').min() * 100 - 100).dropna().median():.2f}%.",
             ""]
    rows = [stats(o, "ВСЕ свечи (база)")]
    rows.append(stats(o[o.green], "  все зелёные"))
    rows.append(stats(o[~o.green], "  все красные"))
    sig_rows = []
    for name, ratio, mult in rules:
        idx = pick(df, ratio, mult)
        s = o.loc[idx]
        if roll:
            rf = roll_flag(s.index)
            rows.append(stats(s[~rf], f"{name} x{mult} (без перехода контракта)"))
            rows.append(stats(s[rf], f"{name} x{mult} — окно перехода контракта"))
            s_use = s[~rf]
        else:
            rows.append(stats(s, f"{name} x{mult}"))
            s_use = s
        rows.append(stats(s_use[s_use.green], f"   ↳ зелёная свеча"))
        rows.append(stats(s_use[~s_use.green], f"   ↳ красная свеча"))
        if delta and "delta_pct" in df:
            dp = df.delta_pct.reindex(s_use.index)
            rows.append(stats(s_use[dp >= DELTA_TH], f"   ↳ дельта ≥ +{DELTA_TH}%"))
            rows.append(stats(s_use[dp <= -DELTA_TH], f"   ↳ дельта ≤ −{DELTA_TH}%"))
        for t in idx:
            r = o.loc[t]
            sig_rows.append({
                "источник": label, "правило": f"{name} x{mult}",
                "свеча (Киев)": t.tz_convert(KYIV).strftime("%Y-%m-%d %H:%M"),
                "объём/норма": round(float(ratio.loc[t]), 1),
                "тело %": r.body_pct,
                "дельта %": float(df.delta_pct.loc[t]) if "delta_pct" in df else None,
                "переход контракта": bool(roll_flag(pd.DatetimeIndex([t]))[0]) if roll else None,
                "макс.рост 10д %": round(r.up_max, 2), "макс.падение 10д %": round(r.dn_max, 2),
                "±3% первым": {1: "вверх", -1: "вниз", 0: "нет", 2: "обе в одной свече"}.get(r.first3),
                "день ±3%": round(r.day3, 1) if r.day3 == r.day3 else None,
                "через 1д %": round(r.r1, 2), "через 3д %": round(r.r3, 2), "через 10д %": round(r.r10, 2),
            })
    lines.append(table(rows))
    lines.append("")
    return "\n".join(lines), sig_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = ["# Золото: аномалии объёма — что было потом",
             "",
             f"Исход считаем {FWD_DAYS} календарных дней от закрытия сигнальной свечи. "
             "«±X% за 10д» — доля случаев, когда цена хоть раз ушла на X% в любую сторону; "
             "«вверх первым» — из них доля, где +X% случилось раньше −X%. "
             "Сравнивай сигналы с базой «ВСЕ свечи»: смысл есть, только если после сигналов заметно чаще "
             "бывают большие движения или направление заметно перекошено.",
             ""]
    all_sig = []

    h1 = yahoo("1h", "730d")
    if h1 is not None and len(h1) > 500:
        h4 = to_4h(h1)
        r12 = h4.volume / norm_prev(h4.volume, 12)
        rsl = h4.volume / norm_slot(h4)
        rules = [("12 свечей", r12, m) for m in MULTS_4H] + [("то же время", rsl, m) for m in MULTS_4H]
        txt, sg = analyse(h4, 4 * H, "A1. Фьючерс CME GC=F, 4h (≈2 года)", rules, roll=True)
        parts.append(txt)
        all_sig += sg
        # насколько объём зависит от времени суток
        prof = h4.groupby(h4.index.hour).volume.median()
        prof = (prof / prof.median()).round(2)
        parts.append("Объём по времени свечи (начало 4h-свечи по UTC; Киев = UTC+2 зимой, +3 летом), "
                     "медиана относительно типичной свечи: " +
                     ", ".join(f"{k:02d}:00 UTC → x{v}" for k, v in prof.items()))
        parts.append("")
    else:
        parts.append("## A1. GC=F 4h — данные Yahoo не загрузились\n")

    d1 = yahoo("1d", "max")
    if d1 is not None and len(d1) > 300:
        d1 = d1[d1.index >= "2010-01-01"].copy()
        d1["hours"] = 24
        rd = d1.volume / norm_prev(d1.volume, 20)
        rules = [("20 дней", rd, m) for m in MULTS_1D]
        txt, sg = analyse(d1, 24 * H, "A2. Фьючерс CME GC=F, дневные свечи с 2010", rules, roll=True)
        parts.append(txt)
        all_sig += sg
        # по годам: устойчиво ли
        o = outcomes(d1, 24 * H)
        idx = pick(d1, rd, 2)
        s = o.loc[idx]
        s = s[~roll_flag(s.index)].dropna(subset=["up_max"])
        base = o.dropna(subset=["up_max"])
        yr = []
        for y in sorted(set(base.index.year)):
            b, x = base[base.index.year == y], s[s.index.year == y]
            yr.append({"год": y, "сигналов x2": len(x),
                       "±3% за 10д: сигналы %": round((x.first3 != 0).mean() * 100) if len(x) else None,
                       "±3% за 10д: база %": round((b.first3 != 0).mean() * 100),
                       "через 10д после сигнала, сред. %": round(x.r10.mean(), 2) if len(x) else None,
                       "через 10д база, сред. %": round(b.r10.mean(), 2)})
        parts.append("### A2 по годам (дневной сигнал x2, без перехода контракта)")
        parts.append(table(yr))
        parts.append("")
    else:
        parts.append("## A2. GC=F 1d — данные Yahoo не загрузились\n")

    bx = binance_xau()
    if bx is not None and len(bx) > 200:
        r12 = bx.volume / norm_prev(bx.volume, 12)
        rsl = bx.volume / norm_slot(bx)
        rules = [("12 свечей", r12, m) for m in MULTS_4H] + [("то же время", rsl, m) for m in MULTS_4H]
        txt, sg = analyse(bx, 4 * H, "B. Binance XAUUSDT перпетуал, 4h (с дельтой)", rules, delta=True)
        parts.append(txt)
        all_sig += sg
    else:
        parts.append("## B. Binance XAUUSDT — архив не найден или слишком короткий\n")

    open(f"{OUT}/gold_summary.md", "w").write("\n".join(parts))
    pd.DataFrame(all_sig).to_csv(f"{OUT}/gold_signals.csv", index=False)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
