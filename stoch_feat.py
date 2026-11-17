"""
Двойной стохастик 4ч, ЛОНГ — доработка: точка входа и признаки, отличающие удачные сигналы от неудачных.
Только история. Живых ботов не трогает. Берёт сигналы из results/stoch/signals.csv.gz
(лонг, окно «до 1 свечи»; отдельно помечены сигналы «на одной свече»).

ВХОДЫ (всё считается по 1ч-свечам фьючерса, горизонт 10 дней, стоп и тейк в одной свече — стоп, комиссия 0,1%):
  • «сразу» — по открытию первой часовой свечи после закрытия сигнальной 4ч-свечи (как в первом прогоне);
  • «лимитка −2%» / «лимитка −4%» — ждём до 24 ч, пока цена опустится на 2% / 4% ниже; не дошла — сделки нет;
  • «пробой хая» — ждём до 24 ч первую часовую свечу, закрывшуюся выше хая сигнальной 4ч-свечи, вход на следующей;
  • «стох 1ч» — ждём до 24 ч, пока на 1ч Stoch(9,3,3) %K пересечёт %D снизу вверх, вход на следующей свече.
  Тейк/стоп считаются от фактической цены входа.

ПРИЗНАКИ (на момент сигнала, без заглядывания вперёд):
  глубина стохастиков (K21, K9); падение от хая 7 дней; расстояние от минимума 30 дней; объём сигнальной 4ч-свечи
  к среднему за 20 свечей; дельта сигнальной свечи и за 24ч (покупки − продажи, % объёма); цвет/тело свечи;
  оборот 24ч; OI за 24ч и 3 дня; толпа (long/short по аккаунтам) и её изменение за 24ч; топ-трейдеры (по позициям);
  фандинг; BTC за 24ч и 7 дней; час сигнала (Киев).
Чтобы не подогнать: смотрим, совпадает ли эффект признака в 2025 и в 2026 отдельно.
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
import stoch as S

OUT = os.environ.get("OUT_DIR", "results/stoch_feat")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
HOR = 240
WAIT = 24
FEE = 0.1
TPSL = [(5, 5), (10, 10), (10, 5), (15, 5), (15, 8)]
ENTRIES = ["сразу", "лимитка −2%", "лимитка −4%", "пробой хая", "стох 1ч"]
D0 = dt.date(2024, 11, 20)


def sim(e, hh, ll, cc, tp, sl, first_stop_only=False):
    """Лонг от цены e по свечам hh/ll/cc. first_stop_only: в первой свече (свеча исполнения лимитки) тейк не проверяем."""
    up = hh >= e * (1 + tp / 100)
    dn = ll <= e * (1 - sl / 100)
    if first_stop_only and len(up):
        up = up.copy()
        up[0] = False
    jt = int(np.argmax(up)) if up.any() else len(up)
    js = int(np.argmax(dn)) if dn.any() else len(dn)
    if jt == len(up) and js == len(dn):
        return (cc[-1] / e - 1) * 100 - FEE
    return (-sl if js <= jt else tp) - FEE


def at(ts, vals, q):
    i = np.searchsorted(ts, q, side="right") - 1
    return vals[i] if i >= 0 else np.nan


def coin(sym, sigs, btc):
    k = PL.klines_any("um", sym, D0, dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 500:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    qv, tb = k.quote_volume.to_numpy(), k.taker_buy_quote_volume.to_numpy()
    k1, d1 = S.stoch(h, l, c, 9, 3, 3)
    up1 = np.r_[False, (k1[:-1] <= d1[:-1]) & (k1[1:] > d1[1:])]
    days = {pd.Timestamp(int(x), unit="ms", tz="UTC").date() + dt.timedelta(days=d) for x in sigs.t for d in (-3, -1, 0)}
    met = PL.metrics(sym, days)
    oi, glob, top = PL.AsOf(met, "oi_usd"), PL.AsOf(met, "glob"), PL.AsOf(met, "top_pos")
    fu = PL.funding(sym, D0)
    rows = []
    for _, s in sigs.iterrows():
        ts = int(s.t)
        i0 = int(np.searchsorted(t, ts))                    # первая часовая свеча после сигнала
        if i0 >= len(t) or t[i0] != ts or i0 < 24 * 30 or i0 + WAIT + HOR > len(t):
            continue
        sc = slice(i0 - 4, i0)                              # сигнальная 4ч-свеча = 4 последних часа
        hi4, cl4, op4 = h[sc].max(), c[i0 - 1], o[i0 - 4]
        v4 = qv[sc].sum()
        prev20 = np.array([qv[i0 - 4 * (j + 2):i0 - 4 * (j + 1)].sum() for j in range(20)])
        r = {"symbol": sym, "t": ts, "одна свеча": bool(s.одна), "K21": s.K21, "K9": s.K9,
             "от хая 7д, %": (cl4 / h[i0 - 168:i0].max() - 1) * 100,
             "от мин 30д, %": (cl4 / l[i0 - 720:i0].min() - 1) * 100,
             "объём свечи x": v4 / prev20.mean() if prev20.mean() > 0 else np.nan,
             "дельта свечи, %": (2 * tb[sc].sum() - v4) / v4 * 100 if v4 > 0 else np.nan,
             "дельта 24ч, %": (2 * tb[i0 - 24:i0].sum() - qv[i0 - 24:i0].sum()) / qv[i0 - 24:i0].sum() * 100,
             "тело свечи, %": (cl4 / op4 - 1) * 100, "оборот 24ч $": qv[i0 - 24:i0].sum()}
        on, o24, o72 = oi.at(ts, 3 * H), oi.at(ts - 24 * H, 3 * H), oi.at(ts - 72 * H, 3 * H)
        r["OI 24ч, %"] = (on / o24 - 1) * 100 if o24 else np.nan
        r["OI 3д, %"] = (on / o72 - 1) * 100 if o72 else np.nan
        g0, g24 = glob.at(ts, 3 * H), glob.at(ts - 24 * H, 3 * H)
        r["толпа L/S"], r["толпа за 24ч"] = g0, g0 - g24
        r["топы L/S"] = top.at(ts, 3 * H)
        r["фандинг, %"] = at(fu.t.to_numpy(), fu.rate8.to_numpy(), ts) if fu is not None else np.nan
        r["BTC 24ч, %"] = (at(btc[0], btc[1], ts) / at(btc[0], btc[1], ts - 24 * H) - 1) * 100
        r["BTC 7д, %"] = (at(btc[0], btc[1], ts) / at(btc[0], btc[1], ts - 168 * H) - 1) * 100
        r["час (Киев)"] = pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert(Hst.KYIV).hour

        # входы
        entries = {"сразу": (o[i0], i0, False)}
        for name, x in (("лимитка −2%", 2), ("лимитка −4%", 4)):
            lim = o[i0] * (1 - x / 100)
            f = np.where(l[i0:i0 + WAIT] <= lim)[0]
            entries[name] = (lim, i0 + int(f[0]), True) if len(f) else None
        f = np.where(c[i0:i0 + WAIT] > hi4)[0]
        entries["пробой хая"] = (o[i0 + f[0] + 1], i0 + int(f[0]) + 1, False) if len(f) else None
        f = np.where(up1[i0:i0 + WAIT])[0]
        entries["стох 1ч"] = (o[i0 + f[0] + 1], i0 + int(f[0]) + 1, False) if len(f) else None
        for name, v in entries.items():
            if v is None:
                continue
            e, j, lim = v
            hh, ll, cc = h[j:j + HOR], l[j:j + HOR], c[j:j + HOR]
            r[f"{name}: вход к сигналу, %"] = (e / o[i0] - 1) * 100
            for tp, sl in TPSL:
                r[f"{name} {tp}/{sl}"] = sim(e, hh, ll, cc, tp, sl, lim)
        rows.append(r)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    e = pd.read_csv("results/stoch/signals.csv.gz")
    same = set(zip(e[(e.side == 1) & (e.окно == 0)].symbol, e[(e.side == 1) & (e.окно == 0)].t))
    g = e[(e.side == 1) & (e.окно == 1)][["symbol", "t", "K21", "K9"]].copy()
    g["одна"] = [(s, t) in same for s, t in zip(g.symbol, g.t)]
    g.to_csv(f"{OUT}/sigs.csv.gz", index=False)
    print("сигналов:", len(g), "из них на одной свече:", int(g.одна.sum()))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    sg = pd.read_csv(f"{OUT}/sigs.csv.gz")
    syms = sorted(sg.symbol.unique())
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    b = PL.klines_any("um", "BTCUSDT", D0, dt.datetime.now(dt.timezone.utc).date())
    btc = (b.open_time.to_numpy() + H, b.close.to_numpy())       # закрытие часа

    def work(s):
        try:
            return coin(s, sg[sg.symbol == s], btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/f{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} сигналов, монет {len(mine)}")


FEATS = ["K21", "K9", "от хая 7д, %", "от мин 30д, %", "объём свечи x", "дельта свечи, %", "дельта 24ч, %",
         "тело свечи, %", "оборот 24ч $", "OI 24ч, %", "OI 3д, %", "толпа L/S", "толпа за 24ч", "топы L/S",
         "фандинг, %", "BTC 24ч, %", "BTC 7д, %", "час (Киев)"]


def st(r):
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
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["год"] = pd.to_datetime(ev.t, unit="ms").dt.year
    ev["время (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/features.csv.gz", index=False)
    w = open(f"{OUT}/stoch_feat_summary.md", "w").write
    w("# Двойной стохастик 4ч, лонг — точка входа и признаки\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Сигналов: {len(ev)} (на одной свече: {int(ev['одна свеча'].sum())}). Результаты — % цены на сделку "
      "с комиссией, без плеча.\n\n")

    w("## 1. Точка входа\n\n")
    for lbl, g0 in (("сигналы на одной свече", ev[ev["одна свеча"]]), ("все (до 1 свечи)", ev)):
        rows = []
        for en in ENTRIES:
            col = f"{en} 10/5"
            if col not in g0:
                continue
            filled = g0[col].notna()
            for tp, sl in TPSL:
                x = g0[filled].sort_values("t")
                rows.append({"вход": en, "исполнено, %": filled.mean() * 100, "тейк/стоп": f"{tp}/{sl}",
                             **st(x[f"{en} {tp}/{sl}"]),
                             "2025 средняя": x[x.год == 2025][f"{en} {tp}/{sl}"].mean(),
                             "2026 средняя": x[x.год == 2026][f"{en} {tp}/{sl}"].mean()})
        w(f"### {lbl}\n\n" + pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")

    w("## 2. Признаки (вход «сразу», все сигналы до 1 свечи)\n\n")
    w("Каждый признак разбит на 5 равных групп. «винрейт 10/10» — тейк 10% раньше стопа 10%; «средняя 15/5» — "
      "результат сделки тейк 15 / стоп 5. Ищем признаки, где разница между группами есть И в 2025, И в 2026.\n\n")
    summary = []
    for f in FEATS:
        x = ev[[f, "год", "сразу 10/10", "сразу 15/5", "сразу 10/5"]].dropna(subset=[f])
        if x[f].nunique() < 5:
            continue
        try:
            x["группа"] = pd.qcut(x[f], 5, duplicates="drop")
        except ValueError:
            continue
        rows = []
        for gname, y in x.groupby("группа", observed=True):
            rows.append({"группа": str(gname), "сигналов": len(y),
                         "винрейт 10/10, %": (y["сразу 10/10"] > 0).mean() * 100,
                         "средняя 10/5": y["сразу 10/5"].mean(), "средняя 15/5": y["сразу 15/5"].mean(),
                         "15/5 в 2025": y[y.год == 2025]["сразу 15/5"].mean(),
                         "15/5 в 2026": y[y.год == 2026]["сразу 15/5"].mean()})
        tb = pd.DataFrame(rows)
        a, b = tb["15/5 в 2025"].to_numpy(), tb["15/5 в 2026"].to_numpy()
        same_dir = np.sign(a[-1] - a[0]) == np.sign(b[-1] - b[0])
        corr = np.corrcoef(a, b)[0, 1] if len(a) > 2 else np.nan
        summary.append({"признак": f, "разброс 15/5 (макс−мин)": tb["средняя 15/5"].max() - tb["средняя 15/5"].min(),
                        "лучшая группа": tb.loc[tb["средняя 15/5"].idxmax(), "группа"],
                        "лучшая: средняя 15/5": tb["средняя 15/5"].max(),
                        "2025 и 2026 согласны (корр.)": corr, "направление совпадает": bool(same_dir)})
        w(f"### {f}\n\n" + tb.round(2).to_markdown(index=False) + "\n\n")
    sm = pd.DataFrame(summary).sort_values("разброс 15/5 (макс−мин)", ascending=False)
    w("## 3. Сводка: какие признаки сильнее всего разделяют (сверху — самые сильные)\n\n"
      "Доверяем только тем, где 2025 и 2026 согласны (корреляция ближе к 1).\n\n")
    w(sm.round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/stoch_feat_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
