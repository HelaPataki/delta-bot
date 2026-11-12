"""
Сделки по сигналу «ликвидность относительно капы × толпа» (из liq_cap.py). Живого бота не трогает.

Сигналы (заданы заранее по результатам liq_cap, карта «14д», сумма в пределах ±10% от цены, в % от капы):
  L1  ЛОНГ:  сверху ≥3% капы, сверху > снизу, топ-трейдеры L/S < 0,8 (крупные в шортах)
  L2  ЛОНГ:  то же по карте «45д»
  L3  ЛОНГ:  сверху ≥3% капы, сверху > снизу, все аккаунты L/S < 0,8
  S1  ШОРТ:  снизу ≥3% капы, снизу > сверху, топ-трейдеры L/S ≥ 1,8 (крупные в лонгах)
  S2  ШОРТ:  сверху ≥3% капы, сверху > снизу, топ-трейдеры L/S ≥ 2,5 («вытряхивание» вниз)
  S3  ШОРТ:  как S1, но топы L/S ≥ 2,5;  S4 — L/S ≥ 3,5 (редкий перекос, зеркало L1);  S5 — как S4 по карте «45д»
Один сигнал = первая 4h-свеча, когда условие стало верным (до этого 24 ч было неверным).
Пока по монете открыта сделка этого сигнала — новые сигналы этой монеты пропускаются.
Вход: по закрытию сигнальной 4h-свечи. Тейк/стоп (в % цены): 5/5, 10/5, 10/10, 15/7,5, 20/10. Не дошло за 7 дней — выход по рынку.
Стоп и тейк в одной часовой свече — считаем стоп. Комиссия 0,1% на круг. Фандинг не учтён.
Депозит: $100, до 3 сделок одновременно по ⅓ депозита; без плеча и с плечом 2x.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-12-01")

import numpy as np
import pandas as pd

import history as Hst
import liq_cap as LC
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_trade")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
HOR = 168
FEE = 0.1
TPSL = [(5, 5), (10, 5), (10, 10), (15, 7.5), (20, 10)]

_orig_load = Hst.load_klines
_cache = {}


def cached_load(sym, interval="4h"):
    key = (sym, interval)
    if key not in _cache:
        _cache[key] = _orig_load(sym, interval)
    return _cache[key]


Hst.load_klines = cached_load


def signals(o):
    up14, dn14 = o["14д вверх10 %капы"], o["14д вниз10 %капы"]
    up45, dn45 = o["45д вверх10 %капы"], o["45д вниз10 %капы"]
    top, allr = o["L/S топ"], o["L/S все"]
    return {
        "L1 лонг: сверху ≥3%, топы в шортах": (+1, (up14 >= 3) & (up14 > dn14) & (top < 0.8)),
        "L2 лонг: то же, карта 45д": (+1, (up45 >= 3) & (up45 > dn45) & (top < 0.8)),
        "L3 лонг: сверху ≥3%, все в шортах": (+1, (up14 >= 3) & (up14 > dn14) & (allr < 0.8)),
        "S1 шорт: снизу ≥3%, топы в лонгах": (-1, (dn14 >= 3) & (dn14 > up14) & (top >= 1.8)),
        "S2 шорт: сверху ≥3%, топы сильно в лонгах": (-1, (up14 >= 3) & (up14 > dn14) & (top >= 2.5)),
        "S3 шорт: снизу ≥3%, топы в лонгах 2,5+": (-1, (dn14 >= 3) & (dn14 > up14) & (top >= 2.5)),
        "S4 шорт: снизу ≥3%, топы в лонгах 3,5+": (-1, (dn14 >= 3) & (dn14 > up14) & (top >= 3.5)),
        "S5 шорт: снизу ≥3%, карта 45д, топы 3,5+": (-1, (dn45 >= 3) & (dn45 > up45) & (top >= 3.5)),
    } | grid(o)


def grid(o):
    """Сетка лонгов для частоты: порог сверху × порог толпы × карта × кто (топы / все)."""
    out = {}
    for mp in ("14д", "45д"):
        up, dn = o[f"{mp} вверх10 %капы"], o[f"{mp} вниз10 %капы"]
        for thr in (1, 2, 3):
            for who, col in (("топы", "L/S топ"), ("все", "L/S все")):
                for ls in (0.8, 0.9, 1.0, 1.2):
                    out[f"G лонг: {mp}, сверху ≥{thr}%, {who} L/S<{ls}"] = (+1, (up >= thr) & (up > dn) & (o[col] < ls))
    return out


# выходы: (имя, тейк — число % или "кластер", стоп %, безубыток после +X% или None, половина на +5%)
EXITS = [("10/10", 10, 10, None, False), ("10/5", 10, 5, None, False), ("15/7.5", 15, 7.5, None, False),
         ("20/10", 20, 10, None, False), ("5/5", 5, 5, None, False),
         ("10/10 + БУ после +5", 10, 10, 5, False), ("20/10 + БУ после +5", 20, 10, 5, False),
         ("кластер / стоп 10", "кластер", 10, None, False), ("кластер / стоп 7.5", "кластер", 7.5, None, False),
         ("кластер / стоп 10 + БУ после +5", "кластер", 10, 5, False),
         ("½ на +5, остаток до кластера, БУ", "кластер", 10, 5, True)]
GRID_EXITS = {"10/10", "кластер / стоп 10", "кластер / стоп 10 + БУ после +5"}
_btc = {}


def simulate(side, px, hh, ll, cc, tp, sl, be, half):
    """Сделка по часовым свечам. Внутри свечи сначала стоп (осторожно). Возвращает (pnl %, итог, часов)."""
    stop = px * (1 - side * sl / 100)
    take = px * (1 + side * tp / 100)
    be_px = px * (1 + side * be / 100) if be else None
    half_px = px * (1 + side * 0.05) if half else None
    realized, left, moved = 0.0, 1.0, False
    for j in range(len(hh)):
        hit_stop = ll[j] <= stop if side > 0 else hh[j] >= stop
        if hit_stop:
            realized += left * side * (stop / px - 1) * 100
            return realized, ("безубыток" if moved else "стоп"), j + 1
        if half_px is not None and left == 1.0 and (hh[j] >= half_px if side > 0 else ll[j] <= half_px):
            realized += 0.5 * 5.0
            left = 0.5
        hit_take = hh[j] >= take if side > 0 else ll[j] <= take
        if hit_take:
            realized += left * side * (take / px - 1) * 100
            return realized, "тейк", j + 1
        if be_px is not None and not moved and (hh[j] >= be_px if side > 0 else ll[j] <= be_px):
            stop, moved = px * (1 + side * 0.002), True      # стоп в безубыток (+комиссия)
    realized += left * side * (cc[-1] / px - 1) * 100
    return realized, "по времени", len(hh)


def btc_regime():
    if "r" not in _btc:
        k = _orig_load("BTCUSDT", "1d")
        k = k.assign(ma=k.close.rolling(50).mean())
        _btc["r"] = dict(zip((k.open_time + 24 * H).astype("int64"), k.close > k.ma))
    return _btc["r"]


def coin(sym, cap):
    try:
        if sym in LC.STOCKS:
            return []
        o = LC.coin(sym, cap)
        if o is None or not len(o):
            return []
        if "снимок" in o:
            o = o[o["снимок"].isna()]
        o = o.sort_values("t").reset_index(drop=True)
        k = cached_load(sym, "1h")
        t_open = k.open_time.to_numpy()
        h, l, c = k.high.to_numpy(), k.low.to_numpy(), k.close.to_numpy()
        idx = {int(x): j for j, x in enumerate(t_open)}
        # спот и фандинг для фильтров
        d0 = pd.Timestamp(o.t.min(), unit="s", tz="UTC").date()
        d1 = pd.Timestamp(o.t.max(), unit="s", tz="UTC").date()
        spot = PL.klines_any("spot", Hst.base_asset(sym) + "USDT", d0, d1)
        spot_t = set(spot.open_time.astype("int64")) if spot is not None else set()
        fund = PL.funding(sym, d0.replace(day=1))
        F = PL.AsOf(fund, "rate8")
        btc = btc_regime()
        rows = []
        for name, (side, cond) in signals(o).items():
            is_grid = name.startswith("G ")
            exits = [e for e in EXITS if (not is_grid or e[0] in GRID_EXITS)] if side > 0 else EXITS[:5]
            cond = cond.fillna(False).to_numpy()
            prev_true_t = -10 ** 18
            busy_until = {e[0]: -1 for e in exits}
            for r_i in np.nonzero(cond)[0]:
                tt = int(o.t.iloc[r_i]) * 1000
                fresh = tt - prev_true_t > 24 * H
                prev_true_t = tt
                if not fresh:
                    continue
                i = idx.get(tt - H)
                if i is None or i + HOR >= len(k):
                    continue
                px = c[i]
                hh, ll, cc = h[i + 1:i + 1 + HOR], l[i + 1:i + 1 + HOR], c[i + 1:i + 1 + HOR]
                cl = o["14д кластер вверх %" if side > 0 else "14д кластер вниз %"].iloc[r_i] if "14д кластер вверх %" in o else np.nan
                day = (tt // (24 * H)) * 24 * H
                tags = {"спот": "есть" if (tt - H) in spot_t or (tt - 2 * H) in spot_t else "нет",
                        "фандинг": F.at(tt, 9 * H), "BTC выше MA50": btc.get(day, None),
                        "OI 24ч %": o["OI 24ч %"].iloc[r_i] if "OI 24ч %" in o else np.nan,
                        "цена 24ч %": o["цена 24ч %"].iloc[r_i] if "цена 24ч %" in o else np.nan}
                for ex_name, tp, sl, be, half in exits:
                    if i <= busy_until[ex_name]:
                        continue
                    if tp == "кластер":
                        if not np.isfinite(cl):
                            continue
                        tp_v = max(3.0, float(cl))
                    else:
                        tp_v = tp
                    pnl, res, hrs = simulate(side, px, hh, ll, cc, tp_v, sl, be, half)
                    busy_until[ex_name] = i + hrs
                    rows.append({"сигнал": name, "монета": sym, "вход": tt, "тейк/стоп": ex_name, "итог": res,
                                 "pnl %": pnl - FEE, "часов": hrs, "тейк %": tp_v, "стоп %": sl,
                                 "капа $M": o["капа $M"].iloc[r_i], "L/S топ": o["L/S топ"].iloc[r_i],
                                 "L/S все": o["L/S все"].iloc[r_i],
                                 "сверху %капы": o["14д вверх10 %капы"].iloc[r_i], "снизу %капы": o["14д вниз10 %капы"].iloc[r_i],
                                 **tags})
        return rows
    finally:
        for key in [x for x in _cache if x[0] == sym]:
            _cache.pop(key, None)


def prep():
    os.makedirs(OUT, exist_ok=True)
    LC.OUT = OUT
    LC.prep()


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, uni[s])
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/trades{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, сделок {len(rows)}")


def deposit(g, lev, slots=3):
    import heapq
    g = g.sort_values("вход")
    E, peak, mdd, op = 100.0, 100.0, 0.0, []
    for t_in, hrs, pnl in zip(g["вход"], g["часов"], g["pnl %"]):
        while op and op[0][0] <= t_in:
            _, ch = heapq.heappop(op)
            E += ch
            peak = max(peak, E)
            mdd = min(mdd, E / peak - 1)
        if len(op) >= slots:
            continue
        heapq.heappush(op, (t_in + hrs * H, E / slots * lev * pnl / 100))
    while op:
        _, ch = heapq.heappop(op)
        E += ch
        peak = max(peak, E)
        mdd = min(mdd, E / peak - 1)
    return E, mdd * 100


def stats(g, months, years):
    sl = g["стоп %"].iloc[0]
    r = {"сделок": len(g), "в месяц": round(len(g) / months, 1), "монет": g["монета"].nunique(),
         "тейков": f"{(g['итог'] == 'тейк').mean() * 100:.0f}%", "стопов": f"{(g['итог'] == 'стоп').mean() * 100:.0f}%",
         "безубыток": f"{(g['итог'] == 'безубыток').mean() * 100:.0f}%", "ср. pnl %": round(g["pnl %"].mean(), 2),
         "ср. R": round((g["pnl %"] / sl).mean(), 2)}
    for y in years:
        gy = g[g["год"] == y]
        r[f"{y}: сделок / ср.pnl"] = f"{len(gy)} / {gy['pnl %'].mean():+.2f}" if len(gy) else "-"
    e1, d1 = deposit(g, 1)
    e2, d2 = deposit(g, 2)
    r["$100 → без плеча"] = f"{e1:.0f} ({d1:.0f}%)"
    r["$100 → 2x"] = f"{e2:.0f} ({d2:.0f}%)"
    return r


def worst_year(r, years):
    v = [float(r[f"{y}: сделок / ср.pnl"].split("/")[1]) for y in years if r[f"{y}: сделок / ср.pnl"] != "-"]
    return min(v) if v else np.nan


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True)
    a["дата"] = pd.to_datetime(a["вход"], unit="ms", utc=True).dt.tz_convert("Europe/Kyiv")
    a["год"] = a["дата"].dt.year
    a["месяц"] = a["дата"].dt.strftime("%Y-%m")
    years = sorted(a["год"].unique())
    a[~a["сигнал"].str.startswith("G ")].assign(дата=lambda x: x["дата"].dt.strftime("%Y-%m-%d %H:%M")).to_csv(
        f"{OUT}/trades.csv.gz", index=False)
    months = max(1.0, (a["дата"].max() - a["дата"].min()).days / 30.4)
    rows = []
    for (sig, ts), g in a.groupby(["сигнал", "тейк/стоп"]):
        rows.append({"сигнал": sig, "выход": ts, **stats(g, months, years)})
    d = pd.DataFrame(rows)
    d["худший год, ср.pnl"] = d.apply(lambda r: worst_year(r, years), axis=1)
    d.to_csv(f"{OUT}/liq_trade_table.csv", index=False)
    w = open(f"{OUT}/liq_trade_summary.md", "w").write
    w("# Сделки: ликвидность относительно капы × толпа\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
    w("Выходы: «кластер» = тейк на уровне самой толстой полосы шортовых ликвидаций сверху (не ближе 3%); "
      "«БУ после +5» = после +5% стоп переносится на вход; «½ на +5» = половина закрывается на +5%, остаток идёт "
      "до кластера со стопом в безубытке. В скобках у депозита — максимальная просадка (по закрытым сделкам).\n\n")
    main = d[~d["сигнал"].str.startswith("G ")]
    w("## Основные сигналы × все выходы\n\n" + main.to_markdown(index=False) + "\n\n")
    gd = d[d["сигнал"].str.startswith("G ") & (d["в месяц"] >= 8)]
    if len(gd):
        w("## Сетка лонгов: 8+ сделок в месяц (≈2+ в неделю), лучшие по худшему году\n\n"
          + gd.sort_values("худший год, ср.pnl", ascending=False).head(40).to_markdown(index=False) + "\n\n")
    # фильтры: для L1–L3 и 5 лучших сеточных, на трёх выходах
    best_grid = list(gd.sort_values("худший год, ср.pnl", ascending=False)["сигнал"].drop_duplicates().head(5)) if len(gd) else []
    pick = [x for x in a["сигнал"].unique() if x[:2] in ("L1", "L2", "L3")] + best_grid
    fl = {"спот есть": lambda x: x["спот"] == "есть", "спота нет": lambda x: x["спот"] == "нет",
          "фандинг < 0": lambda x: x["фандинг"] < -0.005, "фандинг ≈ 0 (−0,005…0,011)": lambda x: x["фандинг"].between(-0.005, 0.011),
          "фандинг > 0,011": lambda x: x["фандинг"] > 0.011,
          "BTC выше MA50": lambda x: x["BTC выше MA50"] == True, "BTC ниже MA50": lambda x: x["BTC выше MA50"] == False,
          "OI за 24ч вырос 5%+": lambda x: x["OI 24ч %"] >= 5, "OI за 24ч не рос": lambda x: x["OI 24ч %"] < 5,
          "цена за 24ч упала": lambda x: x["цена 24ч %"] < 0, "цена за 24ч выросла": lambda x: x["цена 24ч %"] >= 0}
    rows = []
    for sig in pick:
        for ex in ("10/10", "кластер / стоп 10", "кластер / стоп 10 + БУ после +5"):
            g0 = a[(a["сигнал"] == sig) & (a["тейк/стоп"] == ex)]
            if not len(g0):
                continue
            rows.append({"сигнал": sig, "выход": ex, "фильтр": "— без фильтра", **stats(g0, months, years)})
            for fn, f in fl.items():
                g = g0[f(g0).fillna(False).astype(bool)]
                if len(g) >= 10:
                    rows.append({"сигнал": sig, "выход": ex, "фильтр": fn, **stats(g, months, years)})
    if rows:
        fdf = pd.DataFrame(rows)
        fdf.to_csv(f"{OUT}/liq_trade_filters.csv", index=False)
        w("## Фильтры (спот, фандинг, BTC, OI, цена за сутки)\n\n" + fdf.to_markdown(index=False) + "\n\n")
    for sig, g in a[(a["тейк/стоп"] == "10/10") & ~a["сигнал"].str.startswith("G ")].groupby("сигнал"):
        w(f"## {sig}: монеты и месяцы\n\n")
        w("Монеты: " + ", ".join(f"{s} {v}" for s, v in g["монета"].value_counts().head(25).items()) + "\n\n")
        w("По месяцам: " + ", ".join(f"{m}: {v}" for m, v in g["месяц"].value_counts().sort_index().items()) + "\n\n")
    print(open(f"{OUT}/liq_trade_summary.md").read()[:20000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
