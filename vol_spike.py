"""
Огромный всплеск объёма на 4h — предвестник пампа или дампа? Что происходит за 60 дней и чем отличаются эти случаи.
Все USDT-перпетуалы Binance (с делистнутыми, без акций), всплески с 2022 года (раньше нет данных о позициях).

Всплеск: объём 4h-свечи ≥ 10 × среднего за 30 дней (180 свечей), оборот свечи ≥ $1M, монете ≥ 30 дней на фьючерсах.
Повтор у монеты в течение 30 дней — не новый всплеск.

Итог за 60 дней от закрытия свечи всплеска:
  ПАМП  — цена сначала дошла до +50% (раньше, чем до −40%);
  ДАМП  — сначала −40%;
  НИЧЕГО — ни того, ни другого за 60 дней.
Плюс: цена через 7/14/30/60 дней, максимальный рост и падение.

Признаки в момент всплеска (только то, что видно к закрытию свечи):
  свеча — рост/падение, фитили, объём к 30 дням и к 3 дням, оборот;
  что было ДО — цена за 7/30/90 дней, где цена в диапазоне 90 дней, насколько «плоской» была база 30 дней;
  позиции — L/S толпы (все аккаунты) и топов, их изменение за сутки, доля агрессивных покупок (taker), фандинг;
  открытый интерес — изменение за свечу, за сутки, за 7 дней;
  ликвидности — шорты в пределах +10% и лонги в пределах −10% в % капы (та же карта, что у L3), капа;
  рынок — BTC выше/ниже MA50, возраст монеты.
Признаки «первых суток» (видны только через сутки) — отдельно: куда ушла цена и OI за первые 24 часа.

MODE=prep (капы CoinGecko) → MODE=shard (SHARD/NSHARDS) → MODE=combine.
"""

import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

os.environ["START"] = "2021-09-01"
import history as Hst  # noqa: E402
import pump_life as PL  # noqa: E402
import tradfi  # noqa: E402

DIR = "results/vol_spike"
H = 3600 * 1000
D = 24 * H
SPIKE_X, MIN_TURN, MIN_AGE, COOL = 10.0, 1_000_000, 180, 180       # 180 свечей 4h = 30 дней
FROM = pd.Timestamp("2022-01-01").value // 10**6
N60 = 360
UP, DOWN = 50.0, -40.0


def btc_regime():
    b = Hst.load_klines("BTCUSDT", "4h")
    ma = b.close.rolling(300).mean()
    return pd.Series((b.close > ma).to_numpy(), index=b.open_time.astype("int64").to_numpy())


def liq_map(sym, t_close, cap_now, px_now):
    """Шорты до +10% и лонги до −10% в % капы на момент закрытия свечи всплеска (модель L3)."""
    if not cap_now:
        return np.nan, np.nan
    import live_l3 as L3                     # импорт здесь: модуль тяжёлый
    d1 = dt.datetime.fromtimestamp(t_close / 1000, dt.timezone.utc).date()
    d0 = d1 - dt.timedelta(days=40)
    k = PL.klines_any("um", sym, d0, d1)
    met = PL.metrics(sym, {d0 + dt.timedelta(days=i) for i in range(41)})
    if k is None or met is None:
        return np.nan, np.nan
    k = k[k.open_time + H <= t_close].reset_index(drop=True)
    met = met[met.t <= t_close].reset_index(drop=True)
    if not len(k):
        return np.nan, np.nan
    cap_t = cap_now * float(k.close.iloc[-1]) / px_now
    ser = L3.condition_series(k, met, cap_t, hours=1)
    return (ser[-1]["up"], ser[-1]["dn"]) if ser else (np.nan, np.nan)


def analyse(sym, df, btc, fund, cap_now):
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    qv = df["quote_volume"].to_numpy()
    tk = df["taker_buy_quote_volume"].to_numpy() if "taker_buy_quote_volume" in df else np.full(len(df), np.nan)
    t = df.open_time.to_numpy().astype("int64")
    px_now = c[-1]
    rows, last = [], -10**9
    for i in range(MIN_AGE, len(df)):
        if t[i] < FROM or i - last <= COOL:
            continue
        base30 = v[i - 180:i].mean()
        if base30 <= 0 or v[i] < SPIKE_X * base30 or qv[i] < MIN_TURN:
            continue
        last = i
        tc = int(t[i] + 4 * H)                                 # закрытие свечи всплеска
        C = c[i]
        r = {"symbol": sym, "t": int(t[i]), "год": pd.Timestamp(int(t[i]), unit="ms").year,
             "свеча %": (C / o[i] - 1) * 100, "фитиль верх %": (h[i] / max(o[i], C) - 1) * 100,
             "фитиль низ %": (1 - l[i] / min(o[i], C)) * 100,
             "объём к 30д": v[i] / base30, "объём к 3д": v[i] / max(v[i - 18:i].mean(), 1e-12), "оборот $M": qv[i] / 1e6,
             "taker покупки %": tk[i] / qv[i] * 100 if qv[i] > 0 else np.nan,
             "цена 7д до %": (C / c[i - 42] - 1) * 100, "цена 30д до %": (C / c[i - 180] - 1) * 100,
             "цена 90д до %": (C / c[max(0, i - 540)] - 1) * 100 if i >= 540 else np.nan,
             "возраст дн": i / 6, "BTC выше MA50": bool(btc.asof(t[i])) if len(btc) else None}
        lo90, hi90 = l[max(0, i - 540):i].min(), h[max(0, i - 540):i].max()
        r["место в диапазоне 90д %"] = (C - lo90) / (hi90 - lo90) * 100 if hi90 > lo90 else np.nan
        r["ширина базы 30д %"] = (c[i - 180:i].max() / c[i - 180:i].min() - 1) * 100
        r["пробой макс 30д"] = bool(C > h[i - 180:i].max())
        # итог за 60 дней
        cc, hh, ll = c[i + 1:i + 1 + N60], h[i + 1:i + 1 + N60], l[i + 1:i + 1 + N60]
        r["полные 60д"] = len(cc) == N60
        up_i = np.flatnonzero(hh >= C * (1 + UP / 100))
        dn_i = np.flatnonzero(ll <= C * (1 + DOWN / 100))
        fu, fd = (up_i[0] if len(up_i) else 10**9), (dn_i[0] if len(dn_i) else 10**9)
        r["итог"] = "ПАМП" if fu < fd else ("ДАМП" if fd < fu else "НИЧЕГО")
        r["дней до итога"] = (min(fu, fd) + 1) / 6 if min(fu, fd) < 10**9 else np.nan
        for dd in (1, 7, 14, 30, 60):
            k_ = dd * 6 - 1
            r[f"{dd}д %"] = (cc[k_] / C - 1) * 100 if len(cc) > k_ else np.nan
        if len(hh):
            r["макс рост 60д %"] = (hh.max() / C - 1) * 100
            r["макс падение 60д %"] = (ll.min() / C - 1) * 100
        # первые сутки (подсказка, видна только через 24 ч)
        if len(cc) >= 6:
            r["первые сутки: закрылась выше макс всплеска"] = bool(cc[:6].max() > h[i])
            r["первые сутки: закрылась ниже мин всплеска"] = bool(cc[:6].min() < l[i])
        # фандинг
        if fund is not None and len(fund):
            f_ = fund[fund.t <= tc]
            r["фандинг %"] = float(f_.rate8.iloc[-1]) if len(f_) else np.nan
            r["фандинг 7д ср %"] = float(f_[f_.t > tc - 7 * D].rate8.mean()) if len(f_) else np.nan
        # позиции и OI
        d1 = dt.datetime.fromtimestamp(tc / 1000, dt.timezone.utc).date()
        met = PL.metrics(sym, {d1 - dt.timedelta(days=k_) for k_ in range(0, 9)} | {d1 + dt.timedelta(days=1)})
        if met is not None and len(met):
            A = {k_: PL.AsOf(met, k_) for k_ in ("oi_usd", "glob", "top_pos", "top_acc")}
            at = lambda k_, tt: A[k_].at(tt, 2 * H)
            r["L/S толпа"] = at("glob", tc)
            r["L/S толпа изм. 24ч"] = at("glob", tc) - at("glob", tc - D)
            r["L/S топы (позиции)"] = at("top_pos", tc)
            r["L/S топы изм. 24ч"] = at("top_pos", tc) - at("top_pos", tc - D)
            r["L/S топы (аккаунты)"] = at("top_acc", tc)
            oi = at("oi_usd", tc)
            r["OI за свечу %"] = (oi / at("oi_usd", tc - 4 * H) - 1) * 100
            r["OI 24ч до %"] = (at("oi_usd", tc - 4 * H) / at("oi_usd", tc - 4 * H - D) - 1) * 100
            r["OI 7д до %"] = (at("oi_usd", tc - 4 * H) / at("oi_usd", tc - 4 * H - 7 * D) - 1) * 100
            r["OI / оборот свечи"] = oi / qv[i] if qv[i] > 0 else np.nan
            r["первые сутки: OI %"] = (at("oi_usd", tc + D) / oi - 1) * 100
            r["первые сутки: L/S толпа изм."] = at("glob", tc + D) - at("glob", tc)
        try:
            up, dn = liq_map(sym, tc, cap_now, px_now)
        except Exception as e:  # noqa: BLE001
            print(sym, "карта:", repr(e), flush=True)
            up, dn = np.nan, np.nan
        r["шорты до +10% %капы"], r["лонги до −10% %капы"] = up, dn
        r["капа $M"] = cap_now * C / px_now / 1e6 if cap_now else np.nan
        rows.append(r)
    return rows


def prep():
    import liq_cap as LC
    os.makedirs(DIR, exist_ok=True)
    json.dump(LC.caps_with_fallback(), open(f"{DIR}/caps.json", "w"))


def shard():
    k, n = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
    caps = json.load(open(f"{DIR}/caps.json")) if os.path.exists(f"{DIR}/caps.json") else {}
    btc = btc_regime()
    syms = [s for s in Hst.list_symbols() if s not in tradfi.TRADFI and not s.startswith(("BTCDOM", "USDC"))][k::n]
    rows = []
    for m, s in enumerate(syms):
        try:
            df = Hst.load_klines(s, "4h")
            if df is None or len(df) < MIN_AGE + 10:
                continue
            fund = PL.funding(s, dt.date(2021, 12, 1))
            rows += analyse(s, df, btc, fund, caps.get(Hst.base_asset(s), 0))
        except Exception as e:  # noqa: BLE001
            print(s, "ошибка", repr(e), flush=True)
        if m % 10 == 0:
            print(f"{m}/{len(syms)} {s}: всплесков {len(rows)}", flush=True)
    os.makedirs(f"{DIR}/parts", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{DIR}/parts/sp{k}.csv.gz", index=False)


# ---------------- сводка ----------------
def outcome(g):
    return pd.Series({"всплесков": len(g), "ПАМП": (g["итог"] == "ПАМП").mean() * 100,
                      "ДАМП": (g["итог"] == "ДАМП").mean() * 100, "НИЧЕГО": (g["итог"] == "НИЧЕГО").mean() * 100,
                      "через 30д (медиана)": g["30д %"].median(), "через 60д (медиана)": g["60д %"].median()})


def fmt(t):
    t = t.copy()
    for c_ in ("ПАМП", "ДАМП", "НИЧЕГО"):
        t[c_] = t[c_].map(lambda x: f"{x:.0f}%")
    for c_ in ("через 30д (медиана)", "через 60д (медиана)"):
        t[c_] = t[c_].map(lambda x: f"{x:+.1f}%" if x == x else "—")
    t["всплесков"] = t["всплесков"].astype(int)
    return t


def feature_table(f, col, q=5):
    x = f[col]
    if x.dtype == bool or x.dropna().isin([True, False]).all():
        b = x.map({True: "да", False: "нет"})
    else:
        x = pd.to_numeric(x, errors="coerce")
        if x.notna().sum() < 50:
            return None
        try:
            b = pd.qcut(x, q, duplicates="drop")
        except ValueError:
            return None
        b = b.map(lambda iv: f"{iv.left:.3g} … {iv.right:.3g}" if iv == iv else "нет данных")
    t = f.assign(_b=b).groupby(["_b", "период"], observed=True).apply(outcome).unstack("период")
    pp = t[["ПАМП"]].copy()
    out = pd.DataFrame(index=t.index)
    for p in ("2022–24", "2025–26"):
        if ("всплесков", p) in t:
            out[f"всплесков {p}"] = t[("всплесков", p)].fillna(0).astype(int)
            out[f"ПАМП / ДАМП {p}"] = [f"{a:.0f}% / {b:.0f}%" if a == a else "—"
                                      for a, b in zip(t[("ПАМП", p)], t[("ДАМП", p)])]
            out[f"60д {p}"] = t[("через 60д (медиана)", p)].map(lambda x: f"{x:+.1f}%" if x == x else "—")
    out.index.name = col
    return out


def combine():
    parts = [pd.read_csv(f_) for f_ in glob.glob(f"{DIR}/parts/*.csv.gz")]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d.to_csv(f"{DIR}/spikes.csv.gz", index=False)
    f = d[d["полные 60д"]].copy()
    f["период"] = np.where(f["год"] <= 2024, "2022–24", "2025–26")
    f["цвет"] = np.where(f["свеча %"] >= 0, "зелёная", "красная")
    out = ["# Огромный всплеск объёма на 4h: предвестник пампа или дампа?", "",
           f"Всплеск: объём 4h ≥ ×{SPIKE_X:.0f} к среднему за 30 дней, оборот ≥ $1M, монете ≥ 30 дней; повтор у монеты в течение 30 дней "
           f"не считается. С 2022 года, все монеты с делистнутыми, без акций. Всплесков: {len(d)}, с полными 60 днями после: {len(f)}.", "",
           f"**ПАМП** — цена сначала дошла до +{UP:.0f}% (раньше, чем до {DOWN:.0f}%), **ДАМП** — сначала {DOWN:.0f}%, "
           "**НИЧЕГО** — ни того, ни другого за 60 дней. Все цены — к закрытию свечи всплеска.", "",
           "## В целом", "",
           fmt(f.groupby("период").apply(outcome)).to_markdown(), "",
           fmt(f.groupby("год").apply(outcome)).to_markdown(), "",
           "### По цвету свечи", "", fmt(f.groupby(["цвет", "период"]).apply(outcome)).to_markdown(), "",
           "## Признаки в момент всплеска", "",
           "Каждый признак разбит на 5 равных групп. Смотрим, где ПАМП заметно чаще или реже среднего — **в обоих периодах**. "
           "Если признак работает только в одном периоде, это, скорее всего, случайность.", ""]
    feats = ["свеча %", "фитиль верх %", "фитиль низ %", "объём к 30д", "объём к 3д", "оборот $M", "taker покупки %",
             "цена 7д до %", "цена 30д до %", "цена 90д до %", "место в диапазоне 90д %", "ширина базы 30д %", "пробой макс 30д",
             "L/S толпа", "L/S толпа изм. 24ч", "L/S топы (позиции)", "L/S топы изм. 24ч", "L/S топы (аккаунты)",
             "OI за свечу %", "OI 24ч до %", "OI 7д до %", "OI / оборот свечи", "фандинг %", "фандинг 7д ср %",
             "шорты до +10% %капы", "лонги до −10% %капы", "капа $M", "BTC выше MA50", "возраст дн"]
    for col in feats:
        if col not in f:
            continue
        t = feature_table(f, col)
        if t is not None:
            out += [f"### {col}", "", t.to_markdown(), ""]
    out += ["## Подсказки первых суток (видны только через 24 часа после всплеска)", ""]
    for col in ["первые сутки: закрылась выше макс всплеска", "первые сутки: закрылась ниже мин всплеска",
                "первые сутки: OI %", "первые сутки: L/S толпа изм.", "1д %"]:
        if col in f:
            t = feature_table(f, col)
            if t is not None:
                out += [f"### {col}", "", t.to_markdown(), ""]
    open(f"{DIR}/vol_spike_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    {"prep": prep, "combine": combine}.get(os.environ.get("MODE"), shard)()
