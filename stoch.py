"""
Двойной стохастик на 4ч (идея Анастасии). Только история. Живых ботов не трогает.

Два Stochastic (как на TradingView): Stoch(21, 9, 9) и Stoch(9, 3, 3).
  %K = SMA(100·(close − мин. low за N) / (макс. high за N − мин. low за N), сглаживание K); %D = SMA(%K, сглаживание D).
СИГНАЛ ЛОНГ: у ОБОИХ стохастиков %K пересекает %D снизу вверх в зоне перепроданности —
  на свече пересечения %D < 30 и на свече до неё %K < 30.
СИГНАЛ ШОРТ — зеркально: %K пересекает %D сверху вниз, %D > 70 и %K свечой раньше > 70.
«Одновременно» проверяем в трёх вариантах: на одной и той же 4ч-свече; с разницей до 1 свечи; до 3 свечей
  (сигнал — на закрытии свечи, где случилось второе пересечение).
Вход: по открытию следующей 4ч-свечи (сразу после закрытия сигнальной).
Без конкретного стопа/тейка: смотрим 10 дней (60 свечей) после входа, когда цена впервые дошла до
  +3/5/8/10/15/20% в нашу сторону и до −3/5/8/10/15/20% против нас. Оба уровня в одной свече — считаем, что сначала «против».
Для сравнения — «любая свеча»: то же самое для входа на каждой 4ч-свече всех монет (базовый уровень рынка).
Если сигнал не лучше «любой свечи», значит, индикатор ничего не добавляет.
Монеты: все USDT-перпетуалы Binance; период сигналов: 2025-01-01 — вчера. Комиссия 0,1% на круг в результатах сделок.
"""

import json
import os
import sys
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-11-01")       # запас на разгон индикаторов

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

import history as Hst

OUT = os.environ.get("OUT_DIR", "results/stoch")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
FROM = pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000
H4 = 4 * 3600 * 1000
HOR = 60                                   # 10 дней
LEVELS = [3, 5, 8, 10, 15, 20]
WINDOWS = [0, 1, 3]                         # допуск между пересечениями, свечей
FEE = 0.1
GRID = [(5, 5), (8, 8), (10, 10), (10, 5), (15, 5), (15, 8), (20, 10), (5, 10), (3, 3)]   # тейк/стоп


def kyiv(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(Hst.KYIV).strftime("%Y-%m-%d %H:%M")


def stoch(h, l, c, n, sk, sd):
    hh = pd.Series(h).rolling(n, min_periods=n).max()
    ll = pd.Series(l).rolling(n, min_periods=n).min()
    raw = 100 * (pd.Series(c) - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(sk, min_periods=sk).mean()
    d = k.rolling(sd, min_periods=sd).mean()
    return k.to_numpy(), d.to_numpy()


def crosses(k, d, side):
    kp, dp = np.r_[np.nan, k[:-1]], np.r_[np.nan, d[:-1]]
    if side > 0:
        return (kp <= dp) & (k > d) & (d < 30) & (kp < 30)
    return (kp >= dp) & (k < d) & (d > 70) & (kp > 70)


def recent(mask, w):
    """Было ли событие на этой свече или в предыдущих w свечах."""
    if w == 0:
        return mask
    return pd.Series(mask.astype(float)).rolling(w + 1, min_periods=1).max().to_numpy() > 0


def hits(o, h, l, c):
    """Для входа по открытию свечи i+1: номер свечи (1..60), где впервые достигнут каждый уровень; 0 — не достигнут."""
    n = len(c)
    m = n - 1 - HOR
    if m <= 0:
        return None
    e = o[1:m + 1]
    hw, lw = swv(h[1:], HOR)[:m], swv(l[1:], HOR)[:m]
    out = {}
    for L in LEVELS:
        for side in (1, -1):
            up = hw >= e[:, None] * (1 + L / 100)
            dn = lw <= e[:, None] * (1 - L / 100)
            fav, adv = (up, dn) if side > 0 else (dn, up)
            out[(side, "fav", L)] = np.where(fav.any(1), fav.argmax(1) + 1, 0)
            out[(side, "adv", L)] = np.where(adv.any(1), adv.argmax(1) + 1, 0)
    out["end"] = c[HOR:HOR + m] / e - 1          # закрытие через 60 свечей
    return out, m


def coin(sym):
    k = Hst.load_klines(sym, "4h")
    if k is None or len(k) < 300:
        return [], None
    k = k.reset_index(drop=True)
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    qv = k.quote_volume.to_numpy()
    r = hits(o, h, l, c)
    if r is None:
        return [], None
    hz, m = r
    k1, d1 = stoch(h, l, c, 21, 9, 9)
    k2, d2 = stoch(h, l, c, 9, 3, 3)
    turn24 = pd.Series(qv).rolling(6, min_periods=6).sum().to_numpy()
    ok_t = (t >= FROM)[:m]
    rows = []
    for side in (1, -1):
        a, b = crosses(k1, d1, side), crosses(k2, d2, side)
        for w in WINDOWS:
            sig = ((a & recent(b, w)) | (b & recent(a, w)))[:m] & ok_t
            for i in np.where(sig)[0]:
                row = {"symbol": sym, "t": int(t[i]) + H4, "side": side, "окно": w, "оборот 24ч $": turn24[i],
                       "K21": k1[i], "K9": k2[i], "конец 10д, %": side * hz["end"][i] * 100}
                for L in LEVELS:
                    row[f"+{L}"] = int(hz[(side, "fav", L)][i])
                    row[f"-{L}"] = int(hz[(side, "adv", L)][i])
                rows.append(row)
    # базовый уровень: все свечи периода — только агрегаты
    base = {}
    for side in (1, -1):
        sel = ok_t
        agg = {"n": int(sel.sum())}
        for L in LEVELS:
            f, a_ = hz[(side, "fav", L)][sel], hz[(side, "adv", L)][sel]
            agg[f"fav{L}"] = int((f > 0).sum())
            agg[f"adv{L}"] = int((a_ > 0).sum())
            agg[f"first{L}"] = int(((f > 0) & ((a_ == 0) | (f < a_))).sum())
        for tp, sl in GRID:
            f, a_ = hz[(side, "fav", tp)][sel], hz[(side, "adv", sl)][sel]
            end = side * hz["end"][sel] * 100
            win = (f > 0) & ((a_ == 0) | (f < a_))
            loss = (a_ > 0) & ((f == 0) | (a_ <= f))
            res = np.where(win, tp, np.where(loss, -sl, end)) - FEE
            agg[f"g{tp}/{sl}"] = float(np.nansum(res))
            agg[f"gw{tp}/{sl}"] = int(win.sum())
        base[side] = agg
    return rows, base


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(s for s in Hst.list_symbols() if s.endswith("USDT"))
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return [], None

    rows, bases = [], []
    with ThreadPoolExecutor(4) as ex:
        for r, b in ex.map(work, mine):
            rows += r
            if b:
                for side, agg in b.items():
                    bases.append({"side": side, **agg})
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/sig{SHARD}.csv.gz", index=False)
    pd.DataFrame(bases).to_csv(f"{OUT}/parts/base{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} сигналов, монет {len(mine)}")


def read(prefix):
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        if fn.startswith(prefix):
            try:
                parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def trade(g, tp, sl):
    f, a = g[f"+{tp}"].to_numpy(), g[f"-{sl}"].to_numpy()
    win = (f > 0) & ((a == 0) | (f < a))
    loss = (a > 0) & ((f == 0) | (a <= f))
    return np.where(win, tp, np.where(loss, -sl, g["конец 10д, %"].to_numpy())) - FEE


def stats(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {"сделок": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(),
            "PF": pos / neg if neg > 0 else np.inf, "серия минусов": best}


def combine():
    ev = read("sig").sort_values("t").reset_index(drop=True)
    bs = read("base").groupby("side").sum()
    ev["время (Киев)"] = ev.t.map(kyiv)
    ev.to_csv(f"{OUT}/signals.csv.gz", index=False)
    weeks = (ev.t.max() - ev.t.min()) / (7 * 24 * 3600 * 1000) or 1
    w = open(f"{OUT}/stoch_summary.md", "w").write
    w("# Двойной стохастик, 4ч\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Период сигналов: {kyiv(ev.t.min())[:10]} — {kyiv(ev.t.max())[:10]} ({weeks:.0f} нед.), монет с сигналами: "
      f"{ev.symbol.nunique()}.\n\n")
    names = {1: "ЛОНГ (перепроданность, < 30)", -1: "ШОРТ (перекупленность, > 70)"}

    for side in (1, -1):
        b = bs.loc[side]
        w(f"## {names[side]}\n\n")
        w("### Сколько сигналов\n\n")
        w(pd.DataFrame([{"окно": f"до {x} свечей" if x else "одна свеча", "сигналов": int((ev[(ev.side == side) &
                         (ev.окно == x)]).shape[0]), "в неделю": (ev[(ev.side == side) & (ev.окно == x)]).shape[0] / weeks}
                        for x in WINDOWS]).round(1).to_markdown(index=False) + "\n\n")
        w("### Как скоро цена доходит до уровня (за 10 дней)\n\n")
        w("«Сначала наш» — уровень в нашу сторону взят раньше, чем такой же против нас (тейк = стопу). "
          "«Любая свеча» — то же для входа на любой свече, для сравнения.\n\n")
        for x in WINDOWS:
            g = ev[(ev.side == side) & (ev.окно == x)]
            if not len(g):
                continue
            rows = []
            for L in LEVELS:
                f, a = g[f"+{L}"], g[f"-{L}"]
                first = ((f > 0) & ((a == 0) | (f < a))).mean() * 100
                rows.append({"уровень": f"±{L}%", "дошло в нашу, %": (f > 0).mean() * 100,
                             "медиана, ч": f[f > 0].median() * 4 if (f > 0).any() else np.nan,
                             "дошло против, %": (a > 0).mean() * 100,
                             "медиана против, ч": a[a > 0].median() * 4 if (a > 0).any() else np.nan,
                             "сначала наш, %": first,
                             "любая свеча: сначала наш, %": b[f"first{L}"] / b.n * 100})
            w(f"Окно «{f'до {x} свечей' if x else 'одна свеча'}», сигналов {len(g)}:\n\n")
            w(pd.DataFrame(rows).round(1).to_markdown(index=False) + "\n\n")
        w("### Если поставить тейк/стоп (результат в % цены, с комиссией; не дошло за 10 дней — по рынку)\n\n")
        rows = []
        for x in WINDOWS:
            g = ev[(ev.side == side) & (ev.окно == x)].sort_values("t")
            for tp, sl in GRID:
                rows.append({"окно": x, "тейк/стоп": f"{tp}/{sl}", **stats(trade(g, tp, sl)),
                             "любая свеча: средняя, %": b[f"g{tp}/{sl}"] / b.n,
                             "любая свеча: винрейт, %": b[f"gw{tp}/{sl}"] / b.n * 100})
        w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")
        g = ev[(ev.side == side) & (ev.окно == 1)].copy()
        g["год"] = pd.to_datetime(g.t, unit="ms").dt.year
        w("### Окно «до 1 свечи» по годам, тейк/стоп 10/10 и 5/5\n\n")
        rows = []
        for y, x in g.groupby("год"):
            x = x.sort_values("t")
            rows.append({"год": y, **{f"10/10 {k}": v for k, v in stats(trade(x, 10, 10)).items()},
                         "5/5 средняя, %": trade(x, 5, 5).mean()})
        w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/stoch_summary.md").read()[:8000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
