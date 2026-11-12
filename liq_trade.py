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
    }


def coin(sym, cap):
    try:
        o = LC.coin(sym, cap)
        if o is None or not len(o):
            return []
        if "снимок" in o:
            o = o[o["снимок"].isna()]
        if sym in LC.STOCKS:
            return []
        o = o.sort_values("t").reset_index(drop=True)
        k = cached_load(sym, "1h")
        t_open = k.open_time.to_numpy()
        h, l, c = k.high.to_numpy(), k.low.to_numpy(), k.close.to_numpy()
        idx = {int(x): j for j, x in enumerate(t_open)}
        rows = []
        for name, (side, cond) in signals(o).items():
            cond = cond.fillna(False).to_numpy()
            prev_true_t = -10 ** 18
            busy_until = {tp: -1 for tp in TPSL}
            for r_i in np.nonzero(cond)[0]:
                tt = int(o.t.iloc[r_i]) * 1000            # закрытие 4h-свечи
                fresh = tt - prev_true_t > 24 * H
                prev_true_t = tt
                if not fresh:
                    continue
                i = idx.get(tt - H)                        # последняя 1h-свеча внутри сигнальной 4h
                if i is None or i + HOR >= len(k):
                    continue
                px = c[i]
                hh, ll = h[i + 1:i + 1 + HOR], l[i + 1:i + 1 + HOR]
                for tp, sl in TPSL:
                    if i <= busy_until[(tp, sl)]:
                        continue
                    if side > 0:
                        it, isl = np.nonzero(hh >= px * (1 + tp / 100))[0], np.nonzero(ll <= px * (1 - sl / 100))[0]
                    else:
                        it, isl = np.nonzero(ll <= px * (1 - tp / 100))[0], np.nonzero(hh >= px * (1 + sl / 100))[0]
                    it = it[0] if len(it) else 10 ** 6
                    isl = isl[0] if len(isl) else 10 ** 6
                    if it == isl == 10 ** 6:
                        j, pnl, res = HOR - 1, side * (c[i + HOR] / px - 1) * 100, "по времени"
                    elif isl <= it:
                        j, pnl, res = isl, -sl, "стоп"
                    else:
                        j, pnl, res = it, tp, "тейк"
                    busy_until[(tp, sl)] = i + 1 + j
                    rows.append({"сигнал": name, "монета": sym, "вход": tt, "тейк/стоп": f"{tp}/{sl}",
                                 "итог": res, "pnl %": pnl - FEE, "часов": j + 1,
                                 "капа $M": o["капа $M"].iloc[r_i], "L/S топ": o["L/S топ"].iloc[r_i],
                                 "L/S все": o["L/S все"].iloc[r_i],
                                 "сверху %капы": o["14д вверх10 %капы"].iloc[r_i], "снизу %капы": o["14д вниз10 %капы"].iloc[r_i]})
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
    a.assign(дата=a["дата"].dt.strftime("%Y-%m-%d %H:%M")).to_csv(f"{OUT}/trades.csv.gz", index=False)
    months = max(1.0, (a["дата"].max() - a["дата"].min()).days / 30.4)
    rows = []
    for (sig, ts), g in a.groupby(["сигнал", "тейк/стоп"]):
        tp, sl = map(float, ts.split("/"))
        r = {"сигнал": sig, "тейк/стоп %": ts, "сделок": len(g), "в месяц": round(len(g) / months, 1),
             "монет": g["монета"].nunique(), "тейков": f"{(g['итог'] == 'тейк').mean() * 100:.0f}%",
             "стопов": f"{(g['итог'] == 'стоп').mean() * 100:.0f}%", "ср. pnl %": round(g["pnl %"].mean(), 2),
             "ср. R": round((g["pnl %"] / sl).mean(), 2)}
        for y in sorted(a["год"].unique()):
            gy = g[g["год"] == y]
            r[f"{y}: сделок / ср.pnl"] = f"{len(gy)} / {gy['pnl %'].mean():+.2f}" if len(gy) else "-"
        e1, d1 = deposit(g, 1)
        e2, d2 = deposit(g, 2)
        r["$100 → без плеча"] = f"{e1:.0f} ({d1:.0f}%)"
        r["$100 → 2x"] = f"{e2:.0f} ({d2:.0f}%)"
        rows.append(r)
    d = pd.DataFrame(rows)
    w = open(f"{OUT}/liq_trade_summary.md", "w").write
    w("# Сделки: ликвидность относительно капы × толпа\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
    w("В скобках у депозита — максимальная просадка (по закрытым сделкам).\n\n")
    w(d.to_markdown(index=False) + "\n\n")
    for sig, g in a[a["тейк/стоп"] == "10/5"].groupby("сигнал"):
        w(f"## {sig} (тейк 10 / стоп 5): монеты и месяцы\n\n")
        w("Монеты: " + ", ".join(f"{s} {v}" for s, v in g["монета"].value_counts().head(25).items()) + "\n\n")
        w("По месяцам: " + ", ".join(f"{m}: {v}" for m, v in g["месяц"].value_counts().sort_index().items()) + "\n\n")
    print(open(f"{OUT}/liq_trade_summary.md").read())


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
