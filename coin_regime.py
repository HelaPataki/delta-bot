"""
Проверка на всех монетах режима, найденного на M (coin_study). Живых ботов не трогает.

Часовые свечи. «Тихо» = диапазон цены за 48 ч < 15% и объём за 48 ч не выше 1,2 от обычного (среднее за 14 дней).
Сигналы (первый час, когда условие стало верным, до этого 24 ч было неверным):
  A  ШОРТ: тихо, цена за 48 ч сползла на 4%+, толпа в лонгах (L/S всех аккаунтов >= 1,4)
  B  ШОРТ (для сравнения): то же, но толпа НЕ в лонгах (L/S < 1,0)
  C  ШОРТ (для сравнения): тихо, цена стоит (±2% за 48 ч), толпа в лонгах (>= 1,4)
Вход по закрытию сигнального часа. Тейк/стоп в % цены: 5/5, 8/4, 10/5, 10/10. Не дошло за 7 дней — выход по рынку.
Стоп и тейк в одной часовой свече — считаем стоп. Комиссия 0,1% на круг. Фандинг не учтён.
Пока по монете открыта сделка этого сигнала — новые сигналы этой монеты пропускаются.
"""

import datetime as dt
import glob
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

OUT = os.environ.get("OUT_DIR", "results/coin_regime")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
HOR = 168
FEE = 0.1
TPSL = [(5, 5), (8, 4), (10, 5), (10, 10)]


def simulate(px, hh, ll, cc, tp, sl):
    """Шорт. Возвращает (pnl %, итог, часов)."""
    for j in range(len(hh)):
        if (hh[j] / px - 1) * 100 >= sl:
            return -sl, "стоп", j + 1
        if (ll[j] / px - 1) * 100 <= -tp:
            return tp, "тейк", j + 1
    return -(cc[-1] / px - 1) * 100, "время", len(hh)


def asof(times, values, query, max_age):
    i = np.searchsorted(times, query, side="right") - 1
    ok = (i >= 0) & (query - times[np.clip(i, 0, len(times) - 1)] <= max_age)
    return np.where(ok, values[np.clip(i, 0, len(values) - 1)], np.nan)


def coin(sym, cap):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < 24 * 30:
        return []
    k = k.reset_index(drop=True)
    d0 = pd.Timestamp(k.open_time.iloc[0], unit="ms").date()
    today = dt.datetime.now(dt.timezone.utc).date()
    days, d = [], d0
    while d < today:
        days.append(d)
        d += dt.timedelta(days=1)
    m = PL.metrics(sym, days)
    if m is None or not len(m):
        return []
    tc = (k.open_time + H).to_numpy(dtype=float)
    mt = m.t.to_numpy(dtype=float)
    g = pd.Series(asof(mt, m["glob"].to_numpy(dtype=float), tc, H / 2))
    top = pd.Series(asof(mt, m["top_pos"].to_numpy(dtype=float), tc, H / 2))
    oi = pd.Series(asof(mt, m["oi"].to_numpy(dtype=float), tc, H / 2)).where(lambda x: x > 0)
    fu = PL.funding(sym, d0.replace(day=1))
    fund = asof(fu.t.to_numpy(dtype=float), fu.rate8.to_numpy(dtype=float), tc, 9 * H) if fu is not None and len(fu) else np.full(len(k), np.nan)
    c, v = k.close, k.quote_volume
    ret48 = (c / c.shift(48) - 1) * 100
    rng48 = (k.high.rolling(48).max() / k.low.rolling(48).min() - 1) * 100
    volr = v.rolling(48).sum() / (v.rolling(336).sum() / 7)
    d48 = (2 * k.taker_buy_quote_volume - v).rolling(48).sum() / v.rolling(48).sum() * 100
    oi48 = (oi / oi.shift(48) - 1) * 100
    quiet = (rng48 < 15) & (volr < 1.2)
    sig = {"A": quiet & (ret48 <= -4) & (g >= 1.4),
           "B": quiet & (ret48 <= -4) & (g < 1.0),
           "C": quiet & (ret48.abs() <= 2) & (g >= 1.4)}
    h, l, cl = k.high.to_numpy(), k.low.to_numpy(), c.to_numpy()
    rows = []
    for name, cond in sig.items():
        cond = cond.fillna(False).to_numpy()
        prev = -10 ** 9
        busy = {e: -1 for e in TPSL}
        for i in np.nonzero(cond)[0]:
            fresh = i - prev > 24
            prev = i
            if not fresh or i < 336 or i + HOR >= len(k):
                continue
            px = cl[i]
            for tp, sl in TPSL:
                if i <= busy[(tp, sl)]:
                    continue
                pnl, res, hrs = simulate(px, h[i + 1:i + 1 + HOR], l[i + 1:i + 1 + HOR], cl[i + 1:i + 1 + HOR], tp, sl)
                busy[(tp, sl)] = i + hrs
                rows.append({"сигнал": name, "монета": sym, "вход": int(tc[i]), "тейк/стоп": f"{tp}/{sl}", "итог": res,
                             "pnl %": round(pnl - FEE, 3), "часов": hrs, "капа $M": round(cap / 1e6, 1),
                             "L/S все": g.iloc[i], "L/S топ": top.iloc[i], "цена 48ч %": round(ret48.iloc[i], 2),
                             "диапазон 48ч %": round(rng48.iloc[i], 2), "объём к норме": round(volr.iloc[i], 3),
                             "дельта 48ч %": round(d48.iloc[i], 2), "OI 48ч %": oi48.iloc[i], "фандинг": fund[i],
                             "объём 24ч $M": round(v.iloc[i - 23:i + 1].sum() / 1e6, 2)})
    return rows


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


def combine():
    parts = [pd.read_csv(p) for p in glob.glob(f"{OUT}/parts/*.csv.gz") if os.path.getsize(p) > 50]
    a = pd.concat([p for p in parts if len(p)])
    a.to_csv(f"{OUT}/trades.csv.gz", index=False)
    print("сделок:", len(a), "монет:", a["монета"].nunique())


if __name__ == "__main__":
    mode = os.environ.get("MODE", "shard")
    if mode == "prep":
        os.makedirs(OUT, exist_ok=True)
        LC.OUT = OUT
        LC.prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
