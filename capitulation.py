"""
«Купить капитуляцию рынка»: лонг после общего слива всего рынка. История 2021–2026, только архив Binance.
Живых ботов не трогает.

Откуда идея: в liq_fade_feat лонги после выноса лонгов работали, когда сливался весь рынок
(вынос лонгов по рынку > $67M за час, BTC −2% за сутки), но таких дней было всего 16. Здесь — длинная проверка.
Ликвидаций по часам за прошлые годы нет, поэтому общий слив определяем по ценам:
  «ликвидный рынок» = 100 самых торгуемых USDT-фьючерсов Binance (оборот за 30 дней, пересчёт раз в день;
  делистнутые монеты тоже учитываются — без подгонки под выживших).
СЛИВ (каждое определение проверяется отдельно; после срабатывания следующие 24 ч не считаются):
  A30  — за час упали на 3%+ не меньше 30% ликвидных монет;
  A50  — то же, не меньше 50% монет;
  M3   — медианная монета упала за час на 3%+;
  B5   — BTC −5% за 24 ч и час красный;
  D50  — за 24 ч упали на 10%+ не меньше 50% ликвидных монет.
Вход: по открытию следующего часа. Что покупаем (депозит делится поровну):
  BTC; ETH; «топ-5 крупных» — 5 самых торгуемых, кроме BTC/ETH; «5 сильнее всех упавших» — из 50 самых торгуемых
  те 5, что упали за этот час сильнее всего; «1 сильнее всех упавшая»; «весь рынок» — все 100 поровну (для сравнения).
Выходы у каждой монеты (в % цены): тейк/стоп 5/5, 10/10, тейк 10/стоп 20, 20/10 (не дошло за 72 ч — по рынку),
  либо закрыть через 24 ч / 72 ч / 7 дней. Стоп и тейк в одной свече — стоп. Комиссия 0,1% на круг. Без плеча.
Результат события = средний результат монет корзины.
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

os.environ.setdefault("START", "2020-12-01")

import numpy as np
import pandas as pd

import history as Hst

OUT = os.environ.get("OUT_DIR", "results/capitulation")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
T0 = int(pd.Timestamp("2021-01-01", tz="UTC").value // 10 ** 6)
FEE = 0.1
TOP = 100
SKIP = ("USDC", "BUSD", "TUSD", "FDUSD", "DOM", "DEFIUSDT", "XAU", "XAG", "PAXG")
TPSL = [(5, 5), (10, 10), (10, 20), (20, 10)]
TIMEX = [24, 72, 168]
HOLD = 72
KYIV = ZoneInfo("Europe/Kyiv")


def kyiv(ms):
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").tz_convert(KYIV).strftime("%Y-%m-%d %H:%M")


# ---------------- данные ----------------
def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = [s for s in Hst.list_symbols() if not any(x in s for x in SKIP)]
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет в архиве:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            k = Hst.load_klines(s, "1h")
            if k is None or len(k) < 24 * 40:
                return s, None
            k = k[k.open_time >= T0 - 31 * 24 * H]
            return s, k
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return s, None

    out = {}
    with ThreadPoolExecutor(4) as ex:
        for s, k in ex.map(work, mine):
            if k is None or not len(k):
                continue
            out[f"{s}|t"] = (k.open_time.to_numpy() // H).astype("int64")
            for c, n in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("quote_volume", "q")):
                out[f"{s}|{n}"] = k[c].to_numpy().astype("float32")
    np.savez_compressed(f"{OUT}/parts/k{SHARD}.npz", **out)
    print(f"часть {SHARD}: {len(out) // 6} монет из {len(mine)}")


def load_matrix():
    data = {}
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        z = np.load(f"{OUT}/parts/{fn}")
        for key in z.files:
            s, f = key.split("|")
            data.setdefault(s, {})[f] = z[key]
    syms = sorted(data)
    h0 = min(d["t"].min() for d in data.values())
    h1 = max(d["t"].max() for d in data.values())
    n = int(h1 - h0 + 1)
    M = {f: np.full((n, len(syms)), np.nan, dtype="float32") for f in "ohlcq"}
    for j, s in enumerate(syms):
        d = data[s]
        idx = (d["t"] - h0).astype("int64")
        for f in "ohlcq":
            M[f][idx, j] = d[f]
    t = (np.arange(n) + h0) * H
    return syms, t, M


# ---------------- события ----------------
def liquid_sets(t, M):
    q = np.nan_to_num(M["q"]).astype("float64")
    have = np.isfinite(M["c"]).astype("int32")
    cq, ch = np.cumsum(q, 0), np.cumsum(have, 0)
    W = 720
    turn = cq.copy()
    turn[W:] -= cq[:-W]
    cnt = ch.copy()
    cnt[W:] -= ch[:-W]
    turn[cnt < 600] = -1
    sets = np.zeros(len(t), dtype=object)
    cur = None
    for i in range(len(t)):
        if i % 24 == 0 or cur is None:
            row = turn[i - 1] if i > 0 else turn[0]
            ok = np.where(row > 0)[0]
            cur = ok[np.argsort(-row[ok])][:TOP]
            cur_turn = cur
        sets[i] = cur_turn
    return sets


def find_events(t, M, syms, sets):
    o, c = M["o"], M["c"]
    r1 = c / o - 1
    r24 = np.full_like(c, np.nan)
    r24[23:] = c[23:] / o[:-23] - 1
    bi = syms.index("BTCUSDT")
    defs = {k: [] for k in ("A30", "A50", "M3", "B5", "D50")}
    last = {k: -10 ** 9 for k in defs}
    start = int(np.searchsorted(t, T0))
    for i in range(start, len(t) - 1):
        S = sets[i]
        if len(S) < 50:
            continue
        x = r1[i, S]
        x = x[np.isfinite(x)]
        if len(x) < 50:
            continue
        y = r24[i, S]
        y = y[np.isfinite(y)]
        share3, med = (x <= -0.03).mean(), np.median(x)
        share10 = (y <= -0.10).mean() if len(y) >= 50 else 0
        cond = {"A30": share3 >= 0.30, "A50": share3 >= 0.50, "M3": med <= -0.03,
                "B5": r24[i, bi] <= -0.05 and r1[i, bi] < 0, "D50": share10 >= 0.50}
        for k, v in cond.items():
            if v and i - last[k] >= 24:
                last[k] = i
                defs[k].append((i, share3, med, share10))
    return defs


def trade(M, j, i0):
    o, h, l, c = M["o"][:, j], M["h"][:, j], M["l"][:, j], M["c"][:, j]
    n = len(o)
    if i0 + 168 > n:
        return None
    e = o[i0]
    if not np.isfinite(e) or e <= 0:
        return None
    hh, ll, cc = h[i0:i0 + HOLD], l[i0:i0 + HOLD], c[i0:i0 + HOLD]
    if not (np.isfinite(hh).all() and np.isfinite(ll).all()):
        return None
    res = {}
    for tp, sl in TPSL:
        hs, ht = ll <= e * (1 - sl / 100), hh >= e * (1 + tp / 100)
        js = int(np.argmax(hs)) if hs.any() else HOLD
        jt = int(np.argmax(ht)) if ht.any() else HOLD
        r = (cc[-1] / e - 1) * 100 if js == HOLD and jt == HOLD else (-sl if js <= jt else tp)
        res[f"{tp}/{sl}"] = r - FEE
    for k in TIMEX:
        x = c[i0 + k - 1]
        if not np.isfinite(x):
            return None
        res[f"{k}ч"] = (x / e - 1) * 100 - FEE
    return res


EXITS = [f"{a}/{b}" for a, b in TPSL] + [f"{k}ч" for k in TIMEX]


def baskets(i, syms, M, S):
    r1 = M["c"][i] / M["o"][i] - 1
    bi, ei = syms.index("BTCUSDT"), syms.index("ETHUSDT")
    big = [j for j in S if j not in (bi, ei)][:5]
    top50 = [j for j in S[:50] if np.isfinite(r1[j])]
    fall = sorted(top50, key=lambda j: r1[j])[:5]
    return {"BTC": [bi], "ETH": [ei], "топ-5 крупных": big, "5 сильнее всех упавших": fall,
            "1 сильнее всех упавшая": fall[:1], "весь рынок": list(S)}


def stats(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.cumsum(r)
    dd = (np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum).max()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {"событий": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(), "медиана, %": np.median(r),
            "худшее, %": r.min(), "PF": pos / neg if neg else np.inf, "серия минусов": best, "просадка, %": dd}


def combine():
    syms, t, M = load_matrix()
    print("матрица:", M["c"].shape)
    sets = liquid_sets(t, M)
    defs = find_events(t, M, syms, sets)
    rows = []
    for k, evs in defs.items():
        for i, share3, med, share10 in evs:
            for bname, js in baskets(i, syms, M, sets[i]).items():
                res = [trade(M, j, i + 1) for j in js]
                res = [r for r in res if r]
                if not res:
                    continue
                avg = {x: float(np.mean([r[x] for r in res])) for x in EXITS}
                rows.append({"слив": k, "t": int(t[i]), "время (Киев)": kyiv(t[i]), "год": pd.Timestamp(int(t[i]), unit="ms").year,
                             "доля упавших за час": share3, "медиана монеты за час, %": med * 100,
                             "доля −10% за сутки": share10, "корзина": bname,
                             "монеты": ",".join(syms[j].replace("USDT", "") for j in js[:5]), "в корзине": len(res), **avg})
    ev = pd.DataFrame(rows)
    ev.to_csv(f"{OUT}/events.csv.gz", index=False)
    years = (t[-1] - T0) / (365.25 * 24 * H)
    w = open(f"{OUT}/capitulation_summary.md", "w").write
    w("# Купить капитуляцию рынка (2021–2026)\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Монет в истории: {len(syms)}; период {kyiv(T0)[:10]} — {kyiv(t[-1])[:10]} ({years:.1f} года).\n\n")
    w("Как читать: результат события — в % цены без плеча (2x — умножить на 2). «Просадка» — худшее падение суммы "
      "результатов от пика, если заходить в каждое событие одинаковой суммой.\n\n")
    for k in defs:
        e = ev[ev.слив == k]
        if not len(e):
            w(f"## {k}: событий нет\n\n")
            continue
        ne = e.t.nunique()
        w(f"## {k}: {ne} событий ({ne / years:.1f} в год)\n\n")
        w("Средний результат события, % (строки — что покупаем, столбцы — выход):\n\n")
        w(e.pivot_table(index="корзина", values=EXITS, aggfunc="mean")[EXITS].round(2).to_markdown() + "\n\n")
        w("Доля событий в плюс, %:\n\n")
        w(e.pivot_table(index="корзина", values=EXITS, aggfunc=lambda x: (x > 0).mean() * 100)[EXITS].round(0).to_markdown() + "\n\n")
        w("По годам (средний результат события, выход 72ч / 10/10; в скобках — событий):\n\n")
        yr = []
        for (b, y), g in e.groupby(["корзина", "год"]):
            yr.append({"корзина": b, "год": y, "v": f"{g['72ч'].mean():.1f} / {g['10/10'].mean():.1f} ({len(g)})"})
        w(pd.DataFrame(yr).pivot(index="корзина", columns="год", values="v").to_markdown() + "\n\n")
    # сводка лучших
    best = []
    for (k, b), g in ev.groupby(["слив", "корзина"]):
        g = g.sort_values("t")
        for x in EXITS:
            s = stats(g[x])
            yrs = g.groupby("год")[x].mean()
            best.append({"слив": k, "корзина": b, "выход": x, **s, "годов в плюсе": f"{(yrs > 0).sum()}/{len(yrs)}"})
    bb = pd.DataFrame(best)
    bb.to_csv(f"{OUT}/grid.csv", index=False)
    w("## Сводка: 25 лучших сочетаний по средней (не меньше 15 событий)\n\n")
    w(bb[bb.событий >= 15].sort_values("средняя, %", ascending=False).head(25).round(2).to_markdown(index=False) + "\n\n")
    # события для глаз
    e = ev[(ev.слив == "A30") & (ev.корзина == "5 сильнее всех упавших")]
    cols = ["время (Киев)", "доля упавших за час", "медиана монеты за час, %", "монеты", "10/10", "24ч", "72ч"]
    w("## Все события A30 (корзина «5 сильнее всех упавших»), чтобы посмотреть на графике\n\n"
      "Время — начало часа слива; вход — следующий час.\n\n")
    w(e[cols].round(2).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/capitulation_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
