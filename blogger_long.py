"""
Лонг-сетап блогера (пример: BREV 17.11.2026) — восстановление по открытым данным и прогон по истории.
Только история. Живых ботов не трогает.

Данные: архив Binance (data.binance.vision): 15m-свечи спота и фьючерса, metrics (OI и long/short по аккаунтам, раз в 5 мин).
Монеты: все USDT-перпетуалы Binance, у которых есть спот-пара USDT на Binance.
Период: с 2025-01-01 до вчерашнего дня (сегодняшних данных в архиве ещё нет).

ПРОВЕРКА — на закрытии каждой 15m-свечи. Пять признаков, каждый даёт 0–3 балла (скор до 15, как у блогера).
Пороги баллов — МОИ (блогер их не раскрывает), подобраны под его пример:
  • спот-объём за 24ч / средний дневной за 3 дня до этого: x2 → 1, x3 → 2, x4 → 3
  • RSI(14) 15m: ≥60 → 1, ≥70 → 2, ≥80 → 3                     (BREV: 92)
  • цена фьючерса выше дневного VWAP (от 00:00 UTC): >0% → 1, >2% → 2, >4% → 3   (BREV: +5.3%)
  • толпа — long/short по аккаунтам: <2.5 → 1, <2.0 → 2, <1.5 → 3     (BREV: 1.69)
  • OI за 4ч: >0% → 1, >1% → 2, >3% → 3                          (BREV: +1.2%)
  Сигнал: все пять признаков дают хотя бы 1 балл. Скор BREV по этим порогам = 3+3+3+2+2 = 13 (у блогера тоже 13).
  Одна монета — не чаще одного сигнала в 24 ч (первый по времени).

ВХОД: по открытию следующей 15m-свечи фьючерса (по рынку).
ВЫХОДЫ (в % цены, без плеча; с плечом 2x — умножить на 2; комиссия 0,1% на круг; фандинг не учтён):
  • «как у блогера»: стоп на минимуме последних 24ч; тейк 6 / 10 / 15 / 21 / 27%;
  • лесенка блогера: по 25% позиции на +6 / +15 / +21 / +27.6%, стоп на минимуме 24ч
    (и вариант «после TP1 стоп в безубыток»);
  • короткие стопы для сравнения: 5 / 8 / 10% с теми же тейками.
  Не дошло ни до стопа, ни до тейка за 72 ч — закрываем по рынку. Стоп и тейк в одной 15m-свече — считаем стоп.
Чего НЕТ: стен в стакане (истории стакана бесплатно не существует) и «исторических сопротивлений» — вместо них
фиксированные тейки.
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

OUT = os.environ.get("OUT_DIR", "results/blogger_long")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
D0 = dt.date.fromisoformat(os.environ.get("FROM", "2025-01-01"))
M15 = 15 * 60 * 1000
H = 3600 * 1000
DAY = 96                  # 15m-свечей в сутках
HOLD = 72 * 4             # 72 часа
FEE = 0.1
TAKES = [6, 10, 15, 21, 27]
STOPS = ["24ч", 5, 8, 10]
LADDER = [6, 15, 21, 27.6]
COOL = 24 * H


# ---------------- данные ----------------
def kl15(market, sym, d0, d1):
    today = dt.datetime.now(dt.timezone.utc).date()
    this_month = dt.date(today.year, today.month, 1)
    base = "data/spot" if market == "spot" else "data/futures/um"
    urls = []
    for m in Hst.months(d0, min(d1, today)):
        if m < this_month:
            urls.append(f"{Hst.ARCHIVE}/{base}/monthly/klines/{sym}/15m/{sym}-15m-{m:%Y-%m}.zip")
        else:
            d = max(m, d0)
            while d < today and d <= d1:
                urls.append(f"{Hst.ARCHIVE}/{base}/daily/klines/{sym}/15m/{sym}-15m-{d:%Y-%m-%d}.zip")
                d += dt.timedelta(days=1)
    with ThreadPoolExecutor(4) as ex:
        frames = [f for f in ex.map(PL.zip_csv, urls) if f is not None]
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].iloc[:, :12].astype(float)
    df.columns = Hst.COLS
    ot = df.open_time.to_numpy()
    df["open_time"] = np.where(ot > 1e14, ot // 1000, ot).astype("int64")
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def rsi(close, n=14):
    d = pd.Series(close).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).where(up.notna()).to_numpy()


def grade(x, steps, lower=False):
    """0–3 балла: steps — пороги на 1, 2, 3 балла."""
    x = np.asarray(x, dtype=float)
    g = np.zeros(len(x), dtype=int)
    for s in steps:
        g += (x < s) if lower else (x > s)
    g[~np.isfinite(x)] = 0
    return g


# ---------------- сделки ----------------
def first(mask):
    return int(np.argmax(mask)) if mask.any() else HOLD


def simulate(o, h, l, c, i0, stop24):
    """Вход по открытию свечи i0. Возвращает dict результатов (% цены, с комиссией) или None."""
    if i0 + HOLD > len(c):
        return None
    e = o[i0]
    hh, ll, cc = h[i0:i0 + HOLD], l[i0:i0 + HOLD], c[i0:i0 + HOLD]
    if not (e > 0) or not np.isfinite(cc[-1]):
        return None
    sd24 = (1 - stop24 / e) * 100
    if not sd24 > 0:
        return None
    end = (cc[-1] / e - 1) * 100
    res = {"стоп 24ч, %": sd24}
    jt = {tp: first(hh >= e * (1 + tp / 100)) for tp in set(TAKES) | set(LADDER)}
    for st in STOPS:
        sd = sd24 if st == "24ч" else float(st)
        js = first(ll <= e * (1 - sd / 100))
        for tp in TAKES:
            if js == HOLD and jt[tp] == HOLD:
                r = end
            elif js <= jt[tp]:
                r = -sd
            else:
                r = tp
            res[f"стоп {st} / тейк {tp}"] = r - FEE
        if st == "24ч":
            # до какого уровня цена дошла ДО стопа (или за 72ч)
            mx = (np.nanmax(hh[:js]) / e - 1) * 100 if js > 0 else 0.0
            res["макс. рост до стопа, %"] = mx
            res["стоп выбит"] = js < HOLD
            res["часов до стопа"] = js / 4 if js < HOLD else np.nan
            # лесенка
            r = 0.0
            for tp in LADDER:
                r += 0.25 * (tp if jt[tp] < js else (-sd if js < HOLD else end))
            res["лесенка, стоп 24ч"] = r - FEE
            j1 = jt[LADDER[0]]
            if js <= j1:
                r = -sd if js < HOLD else end
            else:
                after = ll[j1 + 1:] <= e
                jbe = j1 + 1 + int(np.argmax(after)) if after.any() else HOLD
                r = 0.0
                for tp in LADDER:
                    r += 0.25 * (tp if jt[tp] < jbe else (0.0 if jbe < HOLD else end))
            res["лесенка + безубыток после TP1"] = r - FEE
    return res


def coin(sym):
    d1 = dt.datetime.now(dt.timezone.utc).date()
    d_pre = D0 - dt.timedelta(days=5)
    sp = kl15("spot", sym, d_pre, d1)
    if (sp is None or len(sp) < 2000) and sym.startswith("1000"):
        sp = kl15("spot", sym[7:] if sym.startswith("1000000") else sym[4:], d_pre, d1)
    if sp is None or len(sp) < 2000:
        return []
    fu = kl15("um", sym, d_pre, d1)
    if fu is None or len(fu) < 2000:
        return []
    t0, t1 = int(fu.open_time.min()), int(fu.open_time.max())
    grid = pd.DataFrame({"open_time": np.arange(t0, t1 + M15, M15, dtype="int64")})
    k = grid.merge(fu, on="open_time", how="left")
    k = k.merge(sp[["open_time", "quote_volume"]].rename(columns={"quote_volume": "spot_vol"}), on="open_time", how="left")
    t = k.open_time.to_numpy()
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    v = k.volume.to_numpy()

    sv = pd.Series(k.spot_vol.to_numpy())
    vol24 = sv.rolling(DAY, min_periods=DAY).sum()
    prev3 = sv.shift(DAY).rolling(3 * DAY, min_periods=3 * DAY).sum() / 3
    vx = (vol24 / prev3.replace(0, np.nan)).to_numpy()
    r15 = rsi(pd.Series(c).ffill().to_numpy())
    day = t // (24 * H)
    tp_ = (h + l + c) / 3
    pv = pd.Series(np.nan_to_num(tp_ * v)).groupby(day).cumsum().to_numpy()
    vv = pd.Series(np.nan_to_num(v)).groupby(day).cumsum().to_numpy()
    vwap = np.where(vv > 0, pv / np.where(vv > 0, vv, 1), np.nan)
    vw = (c / vwap - 1) * 100
    low24 = pd.Series(l).rolling(DAY, min_periods=DAY // 2).min().to_numpy()

    g_vx, g_rsi, g_vw = grade(vx, [2, 3, 4]), grade(r15, [60, 70, 80]), grade(vw, [0, 2, 4])
    ok = (t >= pd.Timestamp(D0, tz="UTC").timestamp() * 1000) & (g_vx > 0) & (g_rsi > 0) & (g_vw > 0)
    cand = np.where(ok)[0]
    if not len(cand):
        return []

    # OI и толпа — только за нужные дни
    days = {pd.Timestamp(int(t[i]), unit="ms", tz="UTC").date() + dt.timedelta(days=d) for i in cand for d in (-1, 0)}
    met = PL.metrics(sym, days)
    oi, crowd = PL.AsOf(met, "oi_usd"), PL.AsOf(met, "glob")
    hc = np.where((t + M15) % H == 0)[0]                        # свечи, закрывающие час
    r1h_v = rsi(pd.Series(c).ffill().to_numpy()[hc])            # RSI 1h (справочно)
    rows, last = [], -10 ** 15
    for i in cand:
        tc = int(t[i]) + M15                                      # момент закрытия свечи
        if tc - last < COOL:
            continue
        o_now, o_4h = oi.at(tc, 20 * 60 * 1000), oi.at(tc - 4 * H, 20 * 60 * 1000)
        oich = (o_now / o_4h - 1) * 100 if o_now and o_4h and np.isfinite(o_now) and np.isfinite(o_4h) else np.nan
        cr = crowd.at(tc, 20 * 60 * 1000)
        g_oi, g_cr = grade([oich], [0, 1, 3])[0], grade([cr], [2.5, 2.0, 1.5], lower=True)[0]
        if g_oi == 0 or g_cr == 0:
            continue
        r = simulate(o, h, l, c, i + 1, low24[i])
        if r is None:
            continue
        last = tc
        hi = np.searchsorted(hc, i, side="right") - 1
        rows.append({"symbol": sym, "t": tc, "скор": int(g_vx[i] + g_rsi[i] + g_vw[i] + g_oi + g_cr),
                     "спот x": vx[i], "RSI 15м": r15[i], "RSI 1ч": r1h_v[hi] if hi >= 0 else np.nan,
                     "VWAP, %": vw[i], "толпа": cr, "OI 4ч, %": oich, "спот 24ч $": vol24[i],
                     "цена 24ч, %": (c[i] / c[i - DAY] - 1) * 100 if i >= DAY else np.nan,
                     "вход": o[i + 1], **r})
    return rows


# ---------------- запуск ----------------
def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = [s for s in Hst.list_symbols() if s.endswith("USDT")]
    json.dump(sorted(syms), open(f"{OUT}/universe.json", "w"))
    print("перпетуалов:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/sig{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} сигналов, монет {len(mine)}")


def stats(r, weeks):
    r = r.dropna().to_numpy()
    if len(r) == 0:
        return {}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    cum = np.cumsum(r)
    dd = (np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum).max()
    return {"сделок": len(r), "в неделю": len(r) / weeks, "винрейт, %": (r > 0).mean() * 100,
            "средняя, %": r.mean(), "сумма, %": r.sum(), "PF": pos / neg if neg > 0 else np.inf,
            "серия минусов": best, "просадка, %": dd}


EXITS = [f"стоп {s} / тейк {tp}" for s in STOPS for tp in TAKES] + ["лесенка, стоп 24ч", "лесенка + безубыток после TP1"]


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["время (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/signals.csv.gz", index=False)
    weeks = (ev.t.max() - ev.t.min()) / (7 * 24 * H) or 1
    w = open(f"{OUT}/blogger_long_summary.md", "w").write
    w("# Лонг-сетап блогера — прогон по истории\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Период сигналов: {PL.kyiv(ev.t.min())[:10]} — {PL.kyiv(ev.t.max())[:10]} ({weeks:.0f} нед.), "
      f"монет с сигналами: {ev.symbol.nunique()}, сигналов всего: {len(ev)}.\n\n")
    w("Как читать: «средняя» — средний результат сделки в % цены после комиссии (2x — умножь на 2); «сумма» — если "
      "каждая сделка одинаковой суммой; PF — заработано / потеряно (больше 1 — в плюсе); «просадка» — худшее падение "
      "суммы от пика. Сделки по разным монетам могут идти одновременно — здесь каждая считается отдельно.\n\n")

    groups = [("скор ≥ 13 (как BREV)", ev[ev.скор >= 13]), ("скор 11–12", ev[(ev.скор >= 11) & (ev.скор <= 12)]),
              ("скор ≥ 11", ev[ev.скор >= 11]), ("все сигналы (скор 5–15)", ev)]
    w("## Сколько сигналов\n\n")
    w(pd.DataFrame([{"группа": n, "сигналов": len(g), "в неделю": len(g) / weeks,
                     "стоп 24ч, медиана %": g["стоп 24ч, %"].median()} for n, g in groups]).round(2)
      .to_markdown(index=False) + "\n\n")

    w("## До какого уровня цена доходит, пока не выбит стоп на минимуме 24ч (или за 72 ч)\n\n")
    lv = []
    for n, g in groups:
        mx = g["макс. рост до стопа, %"]
        lv.append({"группа": n, **{f"≥ +{x:g}%": (mx >= x).mean() * 100 for x in [3] + LADDER + [40]},
                   "стоп выбит за 72ч, %": g["стоп выбит"].astype(bool).mean() * 100})
    w(pd.DataFrame(lv).round(1).to_markdown(index=False) + "\n\n")

    for n, g in groups:
        w(f"## {n}: все варианты выхода\n\n")
        g = g.sort_values("t")
        tab = [{"выход": x, **stats(g[x], weeks)} for x in EXITS]
        w(pd.DataFrame(tab).round(2).to_markdown(index=False) + "\n\n")

    g = ev[ev.скор >= 11].copy()
    g["стоп 24ч"] = pd.cut(g["стоп 24ч, %"], [0, 8, 15, 25, 1000], labels=["< 8%", "8–15%", "15–25%", "> 25%"])
    w("## Скор ≥ 11: зависит ли результат от ширины стопа 24ч (выход «стоп 24ч / тейк 15» и лесенка)\n\n")
    per = []
    for b, x in g.groupby("стоп 24ч", observed=True):
        per.append({"стоп 24ч": b, **stats(x["стоп 24ч / тейк 15"], weeks),
                    "лесенка средняя, %": x["лесенка, стоп 24ч"].mean()})
    w(pd.DataFrame(per).round(2).to_markdown(index=False) + "\n\n")

    g["месяц"] = pd.to_datetime(g.t, unit="ms").dt.strftime("%Y-%m")
    w("## Скор ≥ 11 по месяцам (лесенка, стоп 24ч)\n\n")
    per = [{"месяц": m_, **stats(x["лесенка, стоп 24ч"], 1)} for m_, x in g.groupby("месяц")]
    w(pd.DataFrame(per).drop(columns=["в неделю"], errors="ignore").round(2).to_markdown(index=False) + "\n\n")

    show = ["время (Киев)", "symbol", "скор", "спот x", "RSI 15м", "VWAP, %", "толпа", "OI 4ч, %", "стоп 24ч, %",
            "макс. рост до стопа, %", "лесенка, стоп 24ч"]
    w("## Последние 30 сигналов со скором ≥ 11 (для проверки глазами)\n\n")
    w(ev[ev.скор >= 11].tail(30)[show].round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/blogger_long_summary.md").read()[:8000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
