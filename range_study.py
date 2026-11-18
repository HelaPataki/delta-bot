"""
Боковики: что отличает РАСПРЕДЕЛЕНИЕ (после боковика падение) от НАКОПЛЕНИЯ (после боковика рост). Только история.
Живых ботов не трогает.

БОКОВИК: на момент 00:00 UTC (02:00 Киев зимой) последние 7 дней цена фьючерса ходила в коридоре
  шириной от 4% до 25% (макс. хай / мин. лоу). Оборот ≥ $1M в сутки. Один и тот же боковик не считаем дважды:
  после найденного случая по монете пропускаем 14 дней.
ИСХОД за 14 дней от закрытия (цена на момент проверки):
  «вниз» — цена дошла до −15% раньше, чем до +15%; «вверх» — наоборот; «ничего» — ни то ни другое.
  Дополнительно: результат шорта со стопом +10% и тейком −15% (комиссия 0,1%, без фандинга) и изменение за 14 дней.
ПРИЗНАКИ (только данные до момента проверки):
  форма боковика: ширина, где закрытие внутри коридора, наклон, нижние максимумы/минимумы (хаи последних 3 дней
    к хаям первых 4 дней), объём на зелёных часах / на красных;
  объём: последние 24ч к среднему в боковике, боковик к 30 дням до него;
  дельта фьючерса (покупки − продажи по рынку, % объёма) за весь боковик и за последние 24ч;
  спот: доля спота в общем объёме и её изменение, дельта спота;
  позиции: OI за боковик и за 24ч, толпа (long/short аккаунтов) и её изменение, топ-трейдеры и их изменение,
    фандинг в среднем за боковик;
  контекст: дней с листинга, падение от хая 90 дней, изменение за 30 дней до боковика, BTC за 7 и 30 дней.
ПРОВЕРКА: каждый признак → 5 групп; смотрим долю «вниз», долю «вверх» и их разницу, отдельно 2021–2024 и 2025–2026.
  Признак полезен, только если группы ведут себя одинаково в оба периода.
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

OUT = os.environ.get("OUT_DIR", "results/range_study")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, DAY = 3600 * 1000, 24
W = 7 * DAY
HOR = 14 * DAY
D0 = dt.date(2020, 11, 1)
FROM = pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000
WMIN, WMAX, MOVE, MIN_TURN = 4, 25, 15, 1e6


def slope(x):
    x = np.asarray(x, float)
    if len(x) < 3 or not np.isfinite(x).all() or x[0] <= 0:
        return np.nan
    i = np.arange(len(x))
    return np.polyfit(i, x / x[0] * 100, 1)[0] * DAY        # % в сутки


def coin(sym, btc):
    today = dt.datetime.now(dt.timezone.utc).date()
    k = PL.klines_any("um", sym, D0, today)
    if k is None or len(k) < 60 * DAY:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c, qv, tb = (k[x].to_numpy() for x in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"))
    listed = t[0] if t[0] > pd.Timestamp("2020-11-05", tz="UTC").timestamp() * 1000 else None
    sp = PL.klines_any("spot", sym, D0, today)
    if sp is not None and len(sp):
        sp = pd.DataFrame({"open_time": t}).merge(sp[["open_time", "quote_volume", "taker_buy_quote_volume"]],
                                                   on="open_time", how="left")
        sqv, stb = sp.quote_volume.to_numpy(), sp.taker_buy_quote_volume.to_numpy()
    else:
        sqv = stb = np.full(len(t), np.nan)
    fund = SS.Fund(*SS.funding_raw(sym))
    cand, last = [], -10 ** 9
    for j in np.where(t % (DAY * H) == 0)[0]:
        if t[j] < FROM or j < 30 * DAY + W or j + HOR > len(t) or j - last < 14 * DAY:
            continue
        if np.any(np.diff(t[j - W:j + HOR]) != H):
            continue
        hi, lo = h[j - W:j].max(), l[j - W:j].min()
        width = (hi / lo - 1) * 100
        if not (WMIN <= width <= WMAX) or qv[j - DAY:j].sum() < MIN_TURN:
            continue
        cand.append(j)
        last = j
    if not cand:
        return []
    days = set()
    for j in cand:
        d = pd.Timestamp(int(t[j]), unit="ms", tz="UTC").date()
        days |= {d - dt.timedelta(days=x) for x in (0, 1, 7, 8)}
    met = PL.metrics(sym, days)
    oi, glob, top = PL.AsOf(met, "oi_usd"), PL.AsOf(met, "glob"), PL.AsOf(met, "top_pos")
    rows = []
    for j in cand:
        ts = int(t[j])
        rng = slice(j - W, j)
        hi, lo, e = h[rng].max(), l[rng].min(), c[j - 1]
        # исход
        up, dn = h[j:j + HOR] >= e * (1 + MOVE / 100), l[j:j + HOR] <= e * (1 - MOVE / 100)
        ju = int(np.argmax(up)) if up.any() else HOR
        jd = int(np.argmax(dn)) if dn.any() else HOR
        out = "ничего" if ju == HOR and jd == HOR else ("вниз" if jd < ju else "вверх")
        s_up, s_dn = h[j:j + HOR] >= e * 1.10, l[j:j + HOR] <= e * 0.85
        js = int(np.argmax(s_up)) if s_up.any() else HOR
        jt = int(np.argmax(s_dn)) if s_dn.any() else HOR
        short = (-10 if js <= jt else 15) if (js < HOR or jt < HOR) else (1 - c[j + HOR - 1] / e) * 100
        vday = qv[rng].sum() / 7
        v30 = qv[j - W - 30 * DAY:j - W].sum() / 30
        green = c[rng] >= o[rng]
        sq = np.nansum(sqv[rng])
        sq30 = np.nansum(sqv[j - W - 30 * DAY:j - W])
        q30 = qv[j - W - 30 * DAY:j - W].sum()
        o_now, o_st, o_24 = oi.at(ts, 3 * H), oi.at(ts - W * H, 3 * H), oi.at(ts - DAY * H, 3 * H)
        g_now, g_st = glob.at(ts, 3 * H), glob.at(ts - W * H, 3 * H)
        p_now, p_st = top.at(ts, 3 * H), top.at(ts - W * H, 3 * H)
        bi = np.searchsorted(btc[0], ts, side="right") - 1
        b7 = np.searchsorted(btc[0], ts - W * H, side="right") - 1
        b30 = np.searchsorted(btc[0], ts - 30 * DAY * H, side="right") - 1
        h90 = h[max(0, j - 90 * DAY):j].max()
        rows.append({
            "symbol": sym, "t": ts, "исход": out, "шорт 10/15, %": short - 0.1,
            "через 14д, %": (c[j + HOR - 1] / e - 1) * 100,
            "ширина, %": (hi / lo - 1) * 100, "позиция в коридоре, %": (e - lo) / (hi - lo) * 100,
            "наклон, %/д": slope(c[rng]),
            "хаи: конец к началу, %": (h[j - 3 * DAY:j].max() / h[j - W:j - 3 * DAY].max() - 1) * 100,
            "лои: конец к началу, %": (l[j - 3 * DAY:j].min() / l[j - W:j - 3 * DAY].min() - 1) * 100,
            "объём зелёных/красных": qv[rng][green].sum() / max(qv[rng][~green].sum(), 1),
            "объём 24ч к боковику": qv[j - DAY:j].sum() / vday if vday > 0 else np.nan,
            "объём боковика к 30д до": vday / v30 if v30 > 0 else np.nan,
            "дельта боковика, %": (2 * tb[rng].sum() - qv[rng].sum()) / qv[rng].sum() * 100,
            "дельта 24ч, %": (2 * tb[j - DAY:j].sum() - qv[j - DAY:j].sum()) / max(qv[j - DAY:j].sum(), 1) * 100,
            "доля спота, %": sq / (sq + qv[rng].sum()) * 100 if sq > 0 else np.nan,
            "доля спота: изменение, п.п.": (sq / (sq + qv[rng].sum()) - sq30 / (sq30 + q30)) * 100
            if sq > 0 and sq30 > 0 else np.nan,
            "дельта спота, %": (2 * np.nansum(stb[rng]) - sq) / sq * 100 if sq > 0 else np.nan,
            "OI за боковик, %": (o_now / o_st - 1) * 100 if o_st else np.nan,
            "OI за 24ч, %": (o_now / o_24 - 1) * 100 if o_24 else np.nan,
            "толпа L/S": g_now, "толпа: изменение": g_now - g_st,
            "топы L/S": p_now, "топы: изменение": p_now - p_st,
            "фандинг за боковик, %": fund.short_gain(ts - W * H, ts) / 21,
            "дней с листинга": (ts - listed) / (DAY * H) if listed else np.nan,
            "от хая 90д, %": (e / h90 - 1) * 100,
            "30д до боковика, %": (c[j - W - 1] / c[j - W - 1 - 30 * DAY] - 1) * 100,
            "BTC 7д, %": (btc[1][bi] / btc[1][b7] - 1) * 100 if b7 >= 0 else np.nan,
            "BTC 30д, %": (btc[1][bi] / btc[1][b30] - 1) * 100 if b30 >= 0 else np.nan,
            "оборот 24ч $": qv[j - DAY:j].sum()})
    return rows


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
    btc = (b.open_time.to_numpy(), b.open.to_numpy())

    def work(s):
        try:
            return coin(s, btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/r{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} боковиков, монет {len(mine)}")


SKIP = {"symbol", "t", "исход", "шорт 10/15, %", "через 14д, %", "год", "время (Киев)"}


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["год"] = pd.to_datetime(ev.t, unit="ms").dt.year
    ev["время (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/ranges.csv.gz", index=False)
    w = open(f"{OUT}/range_study_summary.md", "w").write
    w("# Боковики: распределение против накопления\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Боковиков: {len(ev)}, монет: {ev.symbol.nunique()}.\n\n")
    base = []
    for per, g in (("2021–24", ev[ev.год <= 2024]), ("2025–26", ev[ev.год >= 2025]), ("все", ev)):
        vc = g.исход.value_counts(normalize=True) * 100
        base.append({"период": per, "боковиков": len(g), "вниз, %": vc.get("вниз", 0), "вверх, %": vc.get("вверх", 0),
                     "ничего, %": vc.get("ничего", 0), "шорт 10/15 средняя, %": g["шорт 10/15, %"].mean()})
    w("## Базовый уровень\n\n" + pd.DataFrame(base).round(1).to_markdown(index=False) + "\n\n")

    w("## Медианы признаков по исходу\n\n")
    feats = [c for c in ev.columns if c not in SKIP]
    med = ev.groupby("исход")[feats].median().T
    w(med.round(2).to_markdown() + "\n\n")

    summ = []
    for f in feats:
        x = ev.dropna(subset=[f]).copy()
        if x[f].nunique() < 10:
            continue
        x["группа"] = pd.qcut(x[f], 5, duplicates="drop")
        rows = []
        for gn, y in x.groupby("группа", observed=True):
            r = {"группа": str(gn), "случаев": len(y)}
            for per, z in (("21–24", y[y.год <= 2024]), ("25–26", y[y.год >= 2025])):
                d, u = (z.исход == "вниз").mean() * 100, (z.исход == "вверх").mean() * 100
                r[f"{per} вниз, %"], r[f"{per} вверх, %"], r[f"{per} вниз−вверх"] = d, u, d - u
                r[f"{per} шорт 10/15"] = z["шорт 10/15, %"].mean()
            rows.append(r)
        tb = pd.DataFrame(rows)
        if len(tb) < 3:
            continue
        a, b = tb["21–24 вниз−вверх"].to_numpy(), tb["25–26 вниз−вверх"].to_numpy()
        corr = np.corrcoef(a, b)[0, 1]
        best = int(np.argmax(a))
        summ.append({"признак": f, "размах вниз−вверх 21–24": a.max() - a.min(), "согласие лет (корр.)": corr,
                     "лучшая группа для шорта (по 21–24)": tb.группа[best],
                     "её вниз−вверх 21–24": a[best], "её вниз−вверх 25–26": b[best],
                     "её шорт 10/15 в 25–26": tb["25–26 шорт 10/15"][best]})
        w(f"### {f}\n\n" + tb.round(1).to_markdown(index=False) + "\n\n")
    sm = pd.DataFrame(summ).sort_values("размах вниз−вверх 21–24", ascending=False)
    head = ("## Сводка: какие признаки отличают распределение от накопления\n\n"
            "«вниз−вверх» — доля падений минус доля ростов в группе (п.п.). Сверху — самые сильные по 2021–2024. "
            "Доверяем признакам с высоким согласием лет (корреляция ближе к 1).\n\n")
    w(head + sm.round(2).to_markdown(index=False) + "\n\n")
    print(head + sm.round(2).to_markdown(index=False))


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
