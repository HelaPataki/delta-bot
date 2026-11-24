"""
Боковик ПОСЛЕ ПАМПА: когда монета «выдохлась» и пойдёт вниз, а когда это база для нового роста.
Только история. Живых ботов не трогает.

Данные: архив Binance (1h-свечи фьючерсов, фандинг, metrics: OI и соотношения лонгов/шортов).
Дни — по UTC (00:00 UTC = 02:00 Киев зимой).

ПАМП: дневной хай на 80%+ выше минимума последних 30 дней (L0 = этот минимум).
  ВЕРШИНА T — самый высокий хай после старта. Если цена перебивает T, вершина переносится.
  Памп «умер», когда закрытие опустилось до L0 × 1.1 (откатил почти весь рост) или 150 дней без нового хая.
МОМЕНТЫ НАБЛЮДЕНИЯ: через 14, 21, 30, 45, 60, 90, 120 дней после вершины (на закрытии дня),
  если вершина за это время не перебита. Признаки — только по данным до этого момента.
ИСХОД (вперёд от момента): «вниз» — цена дошла до −25% раньше, чем до +25% (за 30 дней); «вверх» — наоборот;
  «ничего» — ни то ни другое. Плюс: перебила ли цена вершину T за 60 дней.
СДЕЛКИ (шорт, вход по открытию следующего часа после закрытия дня; комиссия 0,1%, фандинг учтён; держим до 30 дней):
  A — сразу, стоп +15%, тейк −25%;
  B — сразу, стоп чуть выше вершины T (+1%), тейк −25%; считаем только моменты, где такой стоп ≤ 40%;
  C — ждём пробоя: первое дневное закрытие ниже минимума последних 21 дня (−1%) в следующие 30 дней,
      шорт со стопом +15% и тейком −25%.
СЕТАП «выдох» (то, что видно на PROM/BERA/GIGGLE): с вершины прошло ≥ 21 дня, хай последних 10 дней
  на 5%+ ниже хая предыдущих 20 дней (нижний максимум) и этот отскок шёл на меньшем объёме, чем прошлый.
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL
import short_struct as SS

OUT = os.environ.get("OUT_DIR", "results/post_pump")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
DMS = 24 * H
D0 = dt.date(2020, 11, 1)
FROM = pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000
RISE = 0.80
CHECKS = [14, 21, 30, 45, 60, 90, 120]
MOVE = 25
FWD = 30
MIN_TURN = 2e6
EXAMPLES = os.environ.get("EXAMPLES", "PROMUSDT,BERAUSDT,GIGGLEUSDT").split(",")


def grid(k):
    """Часовые свечи на полной сетке (дыры = NaN) и дневные массивы по UTC."""
    t0 = int(k.open_time.iloc[0] // DMS * DMS)
    t1 = int(k.open_time.iloc[-1])
    full = pd.DataFrame({"open_time": np.arange(t0, t1 + 1, H, dtype="int64")})
    k = k.copy()
    k["open_time"] = k.open_time.astype("int64")
    g = full.merge(k[["open_time", "open", "high", "low", "close", "quote_volume"]], on="open_time", how="left")
    hr = {c: g[c].to_numpy(float) for c in ("open", "high", "low", "close", "quote_volume")}
    hr["t"] = g.open_time.to_numpy()
    n = len(g) // 24
    o = hr["open"][:n * 24].reshape(n, 24)
    hh = hr["high"][:n * 24].reshape(n, 24)
    ll = hr["low"][:n * 24].reshape(n, 24)
    cc = hr["close"][:n * 24].reshape(n, 24)
    qq = hr["quote_volume"][:n * 24].reshape(n, 24)
    ok = np.isfinite(cc).all(axis=1)
    d = {"t": t0 + np.arange(n) * DMS, "o": o[:, 0], "h": np.nanmax(np.where(np.isfinite(hh), hh, -np.inf), axis=1),
         "l": np.nanmin(np.where(np.isfinite(ll), ll, np.inf), axis=1), "c": cc[:, -1],
         "qv": np.nansum(qq, axis=1), "ok": ok}
    d["h"] = np.where(ok, d["h"], np.nan)
    d["l"] = np.where(ok, d["l"], np.nan)
    return hr, d


def short_at(hr, j, stop, take, fund, hours=FWD * 24):
    if j >= len(hr["t"]) or not np.isfinite(hr["open"][j]):
        return np.nan
    n = min(hours, len(hr["t"]) - j)
    if n < hours * 0.9:
        return np.nan
    r, _ = SS.short_trade(hr["open"], hr["high"], hr["low"], hr["close"], hr["t"], j, n, stop, take, fund)
    return r


def snapshot(d, ev, i):
    """Признаки на закрытии дня i (только прошлое)."""
    h, l, c, qv = d["h"], d["l"], d["c"], d["qv"]
    T, L0, P = ev["T"], ev["L0"], ev["P"]
    e = c[i]
    a10 = slice(max(0, i - 9), i + 1)
    a20 = slice(max(0, i - 29), max(0, i - 9))
    h10, h20 = np.nanmax(h[a10]), np.nanmax(h[a20]) if i >= 29 else np.nan
    l10, l20 = np.nanmin(l[a10]), np.nanmin(l[a20]) if i >= 29 else np.nan
    j10 = a10.start + int(np.nanargmax(h[a10]))
    j20 = a20.start + int(np.nanargmax(h[a20])) if i >= 29 else None
    r21 = slice(max(0, i - 20), i + 1)
    hi21, lo21 = np.nanmax(h[r21]), np.nanmin(l[r21])
    pz = slice(ev["l0day"], min(P + 4, i + 1))
    vpk_day = pz.start + int(np.nanargmax(qv[pz]))
    vpk = qv[vpk_day]
    v10 = np.nanmean(qv[a10])
    v20 = np.nanmean(qv[a20]) if i >= 29 else np.nan
    return {
        "дней от вершины": i - P,
        "рост пампа, %": (T / L0 - 1) * 100,
        "от вершины, %": (e / T - 1) * 100,
        "откат доли роста, %": (T - e) / (T - L0) * 100,
        "нижний хай, %": (h10 / h20 - 1) * 100 if h20 and np.isfinite(h20) else np.nan,
        "нижний лой, %": (l10 / l20 - 1) * 100 if l20 and np.isfinite(l20) else np.nan,
        "объём отскока к прошлому": qv[j10] / qv[j20] if j20 is not None and qv[j20] > 0 else np.nan,
        "объём 10д к 20д до": v10 / v20 if v20 and v20 > 0 else np.nan,
        "объём 10д к пиковому дню, %": v10 / vpk * 100 if vpk > 0 else np.nan,
        "пик объёма у вершины": int(abs(vpk_day - P) <= 2),
        "цена к хаю дня пикового объёма, %": (e / h[vpk_day] - 1) * 100,
        "ширина 21д, %": (hi21 / lo21 - 1) * 100,
        "позиция в коридоре 21д, %": (e - lo21) / (hi21 - lo21) * 100 if hi21 > lo21 else np.nan,
        "оборот 24ч $": qv[i],
        "_lo21": lo21, "_h10": h10, "_h20": h20, "_j10": j10, "_j20": j20,
    }


def coin(sym, btc):
    today = dt.datetime.now(dt.timezone.utc).date()
    k = PL.klines_any("um", sym, D0, today)
    if k is None or len(k) < 24 * 60:
        return [], [], []
    hr, d = grid(k)
    t, h, l, c, qv = d["t"], d["h"], d["l"], d["c"], d["qv"]
    nd = len(t)
    fund = SS.Fund(*SS.funding_raw(sym))
    snaps, events, now = [], [], []
    ev = None
    last_ok = int(np.where(d["ok"])[0][-1])

    def close_ev(i, how):
        T, P0 = ev["Tmax"], ev["P0"]
        after = l[ev["Pmax"]:min(nd, ev["Pmax"] + 180)]
        mn = np.nanmin(after) if len(after) else np.nan
        k50 = np.where(after <= T * 0.5)[0]
        events.append({"symbol": sym, "старт": PL.kyiv(int(t[ev["l0day"]]))[:10], "вершина": PL.kyiv(int(t[ev["Pmax"]]))[:10],
                       "рост, %": (T / ev["L0"] - 1) * 100, "перебивов вершины": ev["retakes"],
                       "мин. за 180д от вершины, %": (mn / T - 1) * 100,
                       "дней до −50%": int(k50[0]) if len(k50) else np.nan,
                       "конец": how, "t": int(t[ev["Pmax"]])})

    for i in range(30, nd):
        if not d["ok"][i]:
            continue
        if ev is None:
            if t[i] < FROM:
                continue
            w = slice(i - 30, i + 1)
            lo = np.nanmin(l[w])
            if np.isfinite(lo) and lo > 0 and h[i] >= lo * (1 + RISE) and np.nansum(qv[i - 6:i + 1]) / 7 >= MIN_TURN:
                l0 = w.start + int(np.nanargmin(l[w]))
                ev = {"L0": lo, "l0day": l0, "T": h[i], "P": i, "P0": i, "Tmax": h[i], "Pmax": i, "retakes": 0}
            continue
        if h[i] > ev["T"]:
            if i - ev["P"] >= CHECKS[0]:
                ev["retakes"] += 1
            ev["T"], ev["P"] = h[i], i
            if h[i] > ev["Tmax"]:
                ev["Tmax"], ev["Pmax"] = h[i], i
            continue
        if c[i] <= ev["L0"] * 1.1:
            close_ev(i, "умер (откат до старта)")
            ev = None
            continue
        if i - ev["P"] > 150:
            close_ev(i, "150 дней без хая")
            ev = None
            continue
        k_ = i - ev["P"]
        last = i == last_ok
        if k_ not in CHECKS and not last:
            continue
        if i < 29 or qv[i] < MIN_TURN * 0.5:
            if not last:
                continue
        s = snapshot(d, ev, i)
        s.update({"symbol": sym, "t": int(t[i]), "день": PL.kyiv(int(t[i]))[:10], "_P": ev["P"], "_T": ev["T"]})
        if last:
            now.append(s)
            continue
        j = (i + 1) * 24                                   # первый час следующего дня
        e = c[i]
        fw = slice(i + 1, min(nd, i + 1 + FWD))
        hh, ll = hr["high"][j:j + FWD * 24], hr["low"][j:j + FWD * 24]
        up, dn = hh >= e * (1 + MOVE / 100), ll <= e * (1 - MOVE / 100)
        ju = int(np.argmax(up)) if up.any() else 10 ** 9
        jd = int(np.argmax(dn)) if dn.any() else 10 ** 9
        s["исход"] = "ничего" if ju == jd == 10 ** 9 else ("вниз" if jd < ju else "вверх")
        f60 = slice(i + 1, min(nd, i + 61))
        s["перебила вершину за 60д"] = int(np.nanmax(h[f60]) > ev["T"]) if f60.stop > f60.start else np.nan
        s["мин. за 60д, %"] = (np.nanmin(l[f60]) / e - 1) * 100 if f60.stop > f60.start else np.nan
        s["через 30д, %"] = (c[i + FWD] / e - 1) * 100 if i + FWD < nd else np.nan
        s["через 60д, %"] = (c[i + 60] / e - 1) * 100 if i + 60 < nd else np.nan
        s["шорт A, %"] = short_at(hr, j, 15, 25, fund)
        stopB = (ev["T"] * 1.01 / e - 1) * 100
        s["стоп B, %"] = stopB
        s["шорт B, %"] = short_at(hr, j, stopB, 25, fund)
        rC = np.nan
        for q in range(i + 1, min(nd - 1, i + 1 + FWD)):
            if c[q] < s["_lo21"] * 0.99:
                rC = short_at(hr, (q + 1) * 24, 15, 25, fund)
                s["пробой C через, дней"] = q - i
                break
        s["шорт C, %"] = rC
        snaps.append(s)
    if ev is not None:
        close_ev(nd - 1, "ещё жив")
    # метрики (OI, топы, толпа) — только для моментов наблюдения и «сейчас»
    allp = snaps + now
    if allp:
        days = set()
        for s in allp:
            dd = pd.Timestamp(s["t"], unit="ms", tz="UTC").date()
            days |= {dd, dd - dt.timedelta(days=7), pd.Timestamp(int(t[s["_P"]]), unit="ms", tz="UTC").date()}
        met = PL.metrics(sym, days)
        oi, glob, top = PL.AsOf(met, "oi_usd"), PL.AsOf(met, "glob"), PL.AsOf(met, "top_pos")
        for s in allp:
            q = s["t"] + DMS - 1                               # конец дня
            o_now, o_7, o_p = oi.at(q, 6 * H), oi.at(q - 7 * DMS, 6 * H), oi.at(int(t[s["_P"]]) + DMS - 1, 6 * H)
            s["OI 7д, %"] = (o_now / o_7 - 1) * 100 if o_7 else np.nan
            s["OI от вершины, %"] = (o_now / o_p - 1) * 100 if o_p else np.nan
            g, g7 = glob.at(q, 6 * H), glob.at(q - 7 * DMS, 6 * H)
            p, p7 = top.at(q, 6 * H), top.at(q - 7 * DMS, 6 * H)
            s["толпа L/S"], s["толпа 7д"] = g, g - g7
            s["топы L/S"], s["топы 7д"] = p, p - p7
            s["фандинг 7д, %"] = fund.short_gain(q - 7 * DMS, q)
            bi = np.searchsorted(btc[0], q, side="right") - 1
            b30 = np.searchsorted(btc[0], q - 30 * DMS, side="right") - 1
            s["BTC 30д, %"] = (btc[1][bi] / btc[1][b30] - 1) * 100 if b30 >= 0 else np.nan
    for s in allp:
        s["сетап"] = int(s["дней от вершины"] >= 21 and np.isfinite(s["нижний хай, %"]) and s["нижний хай, %"] <= -5
                         and np.isfinite(s["объём отскока к прошлому"]) and s["объём отскока к прошлому"] < 1)
        for key in [x for x in s if x.startswith("_")]:
            s.pop(key)
    return snaps, events, now


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(s for s in Hst.list_symbols() if s.endswith("USDT") and s != "BTCUSDT")
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    b = PL.klines_any("um", "BTCUSDT", D0, dt.datetime.now(dt.timezone.utc).date())
    btc = (b.open_time.to_numpy(), b.close.to_numpy())

    def work(s):
        try:
            return coin(s, btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return [], [], []

    S, E, N = [], [], []
    with ThreadPoolExecutor(3) as ex:
        for a, b_, n in ex.map(work, mine):
            S += a
            E += b_
            N += n
    pd.DataFrame(S).to_csv(f"{OUT}/parts/s{SHARD}.csv.gz", index=False)
    pd.DataFrame(E).to_csv(f"{OUT}/parts/e{SHARD}.csv.gz", index=False)
    pd.DataFrame(N).to_csv(f"{OUT}/parts/n{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: моментов {len(S)}, пампов {len(E)}, живых сейчас {len(N)}")


def read(prefix):
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        if fn.startswith(prefix):
            try:
                parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def trades(g):
    r = {"моментов": len(g)}
    for x in "ABC":
        z = g[f"шорт {x}, %"]
        if x == "B":
            z = z[g["стоп B, %"] <= 40]                    # стоп дальше 40% для депозита нереален
        z = z.dropna()
        r[f"{x}: сделок"] = len(z)
        r[f"{x}: плюс, %"] = (z > 0).mean() * 100 if len(z) else np.nan
        r[f"{x}: средняя, %"] = z.mean() if len(z) else np.nan
    r["вниз, %"] = (g.исход == "вниз").mean() * 100
    r["вверх, %"] = (g.исход == "вверх").mean() * 100
    r["перебила вершину за 60д, %"] = g["перебила вершину за 60д"].mean() * 100
    r["медиана стопа B, %"] = g["стоп B, %"].median()
    return r


SKIP = {"symbol", "t", "день", "исход", "перебила вершину за 60д", "мин. за 60д, %", "через 30д, %", "через 60д, %",
        "шорт A, %", "шорт B, %", "шорт C, %", "стоп B, %", "пробой C через, дней", "год", "сетап"}


def combine():
    ev = read("s").sort_values("t").reset_index(drop=True)
    pumps = read("e")
    now = read("n")
    ev["год"] = pd.to_datetime(ev.t, unit="ms").dt.year
    ev["период"] = np.where(ev.год <= 2024, "2021–24", "2025–26")
    ev.to_csv(f"{OUT}/post_pump.csv.gz", index=False)
    pumps.to_csv(f"{OUT}/pumps.csv.gz", index=False)
    w = open(f"{OUT}/post_pump_summary.md", "w").write
    w("# Боковик после пампа: выдох или база\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Пампов: {len(pumps)}, моментов наблюдения: {len(ev)}, монет: {ev.symbol.nunique()}.\n\n")

    # 1. судьба пампов
    pp = pumps[pumps.конец != "ещё жив"].copy()
    pp["год"] = pd.to_datetime(pp.t, unit="ms").dt.year
    fate = []
    for per, g in (("2021–24", pp[pp.год <= 2024]), ("2025–26", pp[pp.год >= 2025]), ("все", pp)):
        fate.append({"период": per, "пампов": len(g),
                     "откатили почти всё, %": (g.конец == "умер (откат до старта)").mean() * 100,
                     "хоть раз перебили вершину после 14д, %": (g["перебивов вершины"] > 0).mean() * 100,
                     "дошли до −50% от вершины, %": g["дней до −50%"].notna().mean() * 100,
                     "медиана дней до −50%": g["дней до −50%"].median(),
                     "медиана мин. за 180д, %": g["мин. за 180д от вершины, %"].median()})
    w("## 1. Чем заканчиваются пампы +80%\n\n" + pd.DataFrame(fate).round(1).to_markdown(index=False) + "\n\n")

    # 2. по дням от вершины
    rows = []
    for (k, per), g in ev.groupby(["дней от вершины", "период"]):
        r = {"дней от вершины": k, "период": per}
        r.update(trades(g))
        rows.append(r)
    w("## 2. Базовый уровень по дням от вершины (все моменты, без фильтров)\n\n"
      + pd.DataFrame(rows).round(1).to_markdown(index=False) + "\n\n")

    # 3. сетап против остальных
    rows = []
    for (st, per), g in ev.groupby(["сетап", "период"]):
        r = {"сетап «выдох»": "да" if st else "нет", "период": per}
        r.update(trades(g))
        rows.append(r)
    w("## 3. Сетап «выдох» (≥21д от вершины + нижний хай −5% + отскок на меньшем объёме) против остальных\n\n"
      + pd.DataFrame(rows).round(1).to_markdown(index=False) + "\n\n")

    # 4. признаки по группам
    feats = [c for c in ev.columns if c not in SKIP and c not in ("период",) and pd.api.types.is_numeric_dtype(ev[c])]
    summ = []
    w("## 4. Признаки по группам (5 групп; «вниз−вверх» в п.п., шорт A — средняя, %)\n\n")
    for f in feats:
        x = ev.dropna(subset=[f]).copy()
        if x[f].nunique() < 2:
            continue
        x["группа"] = pd.qcut(x[f], 5, duplicates="drop") if x[f].nunique() >= 10 else x[f].astype(str)
        rows = []
        for gn, y in x.groupby("группа", observed=True):
            r = {"группа": str(gn), "случаев": len(y)}
            for per in ("2021–24", "2025–26"):
                z = y[y.период == per]
                dn, up = (z.исход == "вниз").mean() * 100, (z.исход == "вверх").mean() * 100
                r[f"{per} вниз−вверх"] = dn - up
                r[f"{per} шорт A"] = z["шорт A, %"].mean()
                r[f"{per} перебила"] = z["перебила вершину за 60д"].mean() * 100
            rows.append(r)
        tb = pd.DataFrame(rows)
        if len(tb) < 2:
            continue
        a, b = tb["2021–24 вниз−вверх"].to_numpy(), tb["2025–26 вниз−вверх"].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        corr = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= 3 else np.nan
        best = int(np.nanargmax(a)) if np.isfinite(a).any() else 0
        summ.append({"признак": f, "размах 21–24": np.nanmax(a) - np.nanmin(a), "согласие лет": corr,
                     "лучшая группа для шорта": tb.группа[best], "её вниз−вверх 21–24": a[best],
                     "её вниз−вверх 25–26": b[best], "её шорт A 25–26": tb["2025–26 шорт A"][best]})
        w(f"### {f}\n\n" + tb.round(1).to_markdown(index=False) + "\n\n")
    sm = pd.DataFrame(summ).sort_values("размах 21–24", ascending=False)
    w("## 5. Сводка признаков (доверяем тем, где согласие лет ближе к 1)\n\n" + sm.round(2).to_markdown(index=False) + "\n\n")

    # 6. примеры и «сейчас»
    cols = ["symbol", "день", "дней от вершины", "рост пампа, %", "от вершины, %", "нижний хай, %",
            "объём отскока к прошлому", "объём 10д к пиковому дню, %", "OI от вершины, %", "топы L/S", "толпа L/S", "сетап"]
    out = ["исход", "перебила вершину за 60д", "через 30д, %", "шорт A, %", "шорт B, %", "стоп B, %", "шорт C, %"]
    ex = ev[ev.symbol.isin(EXAMPLES)]
    if len(ex):
        w("## 6. Примеры: PROM, BERA, GIGGLE\n\n" + ex[cols + out].round(2).to_markdown(index=False) + "\n\n")
    if len(now):
        now = now.sort_values(["сетап", "дней от вершины"], ascending=[False, True])
        w("## 7. Сейчас: живые боковики после пампа (последний полный день архива)\n\n"
          + now[[c for c in cols if c in now]].round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/post_pump_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
