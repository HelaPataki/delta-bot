"""
Вход ПРОТИВ ликвидаций (идея блогера): цена растёт, шортистов массово выносит → заходим в ШОРТ
(топливо роста сгорело — разворот). Зеркало: цена падает, выносит лонгистов → ЛОНГ.
Только история. Живых ботов не трогает.

Данные: реальные ликвидации Binance — Coinalyze (liquidation-history, только Binance, в $);
  1h-свечи — архив Binance; OI по часам — архив Binance (metrics), OI по дням — Coinalyze.
  • ПО ЧАСАМ — сколько Coinalyze хранит (≈ 3–4 месяца; для 80 монет добавлены сохранённые с июля данные liq_check);
  • ПО ДНЯМ — с 2025-01-01 (грубее, но история длиннее).
Важно: Binance публикует не все ликвидации (не больше одной в секунду на монету) — реальные суммы выше.

СИГНАЛ «шорт» (часовая свеча; для дней — то же по дню):
  • свеча зелёная (закрытие выше открытия);
  • шортов ликвидировано ≥ 3× больше, чем лонгов, и не меньше $20k;
  • размер выноса — одно из двух мерил (каждый порог считается отдельно):
      «% OI»  — ликвидации шортов / OI на начало свечи: 0,25 / 0,5 / 1 / 2 / 3 % (дни: 1 / 2 / 3 / 5 / 10 %);
      «x нормы» — во сколько раз больше обычного для этой монеты (среднее за 7 дней до; дни — 30 дней): x5 / x10 / x20 / x50
        (дни: x3 / x5 / x10).
  СИГНАЛ «лонг» — зеркально (свеча красная, лонгов вынесло ≥ 3× больше шортов).
Вход: по открытию следующей свечи (по рынку, сразу после закрытия сигнальной — данные уже есть).
Одна монета — не чаще одного сигнала в 24 ч (дни — в 48 ч).
Выходы (в % цены): тейк/стоп 3/3, 5/5, 10/10, тейк 5/стоп 10, 10/5, 10/20, 20/10; не дошло за 72 ч (дни — 10 дней) — по рынку.
  Плюс выходы по времени: закрыть через 4 ч / 24 ч / 72 ч.
Стоп и тейк в одной часовой свече — считаем стоп. Комиссия 0,1% на круг. Фандинг не учтён.
Результат сделки — в % цены без плеча (с плечом 2x — умножить на 2).
"""

import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import history as Hst
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_fade")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
KEY = os.environ.get("COINALYZE_KEY", "")
TOPN = int(os.environ.get("TOPN", "200"))
H = 3600 * 1000
D0 = dt.date(2025, 1, 1)
FEE = 0.1
DOM = 3.0          # перекос: нужная сторона ≥ 3× другой
MIN_USD = 20000
TPSL = [(3, 3), (5, 5), (10, 10), (5, 10), (10, 5), (10, 20), (20, 10)]
TIMEX = [4, 24, 72]
CFG = {
    "h": {"hold": 72, "norm": 168, "cool": 24, "oi": [0.25, 0.5, 1, 2, 3], "x": [5, 10, 20, 50], "name": "ПО ЧАСАМ"},
    "d": {"hold": 240, "norm": 30, "cool": 2, "oi": [1, 2, 3, 5, 10], "x": [3, 5, 10], "name": "ПО ДНЯМ"},
}


# ---------------- Coinalyze ----------------
def cz(endpoint, syms, interval, frm, to, rename):
    out = {}
    for i in range(0, len(syms), 10):
        batch = syms[i:i + 10]
        for _ in range(8):
            time.sleep(3)                                  # бережём общий ключ (живые боты)
            try:
                r = requests.get(f"https://api.coinalyze.net/v1/{endpoint}", headers={"api_key": KEY}, timeout=60,
                                 params={"symbols": ",".join(f"{s}_PERP.A" for s in batch), "interval": interval,
                                         "from": int(frm), "to": int(to), "convert_to_usd": "true"})
            except requests.RequestException as e:
                print("Coinalyze сеть:", e)
                time.sleep(10)
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)) + 2)
                continue
            if r.status_code != 200:
                print("Coinalyze", endpoint, r.status_code, r.text[:200])
                break
            for item in r.json():
                h = pd.DataFrame(item.get("history") or [])
                if len(h):
                    out[item["symbol"].replace("_PERP.A", "")] = h.rename(columns=rename)
            break
    rows = []
    for s, h in out.items():
        h["symbol"] = s
        rows.append(h)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    today = dt.datetime.now(dt.timezone.utc).date()
    last = dt.date(today.year, today.month, 1) - dt.timedelta(days=1)
    m = dt.date(last.year, last.month, 1)

    def vol(s):
        f = PL.zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m:%Y-%m}.zip")
        if f is None:
            return 0.0
        f = f[pd.to_numeric(f.iloc[:, 0], errors="coerce").notna()]
        return float(f.iloc[:, 7].astype(float).sum())

    with ThreadPoolExecutor(8) as ex:
        vols = dict(zip(syms, ex.map(vol, syms)))
    uni = [s for s, v in sorted(vols.items(), key=lambda x: -x[1])[:TOPN] if v > 0]
    print("монет:", len(uni))
    now = int(time.time())
    d0 = int(pd.Timestamp(D0, tz="UTC").timestamp())
    lh = cz("liquidation-history", uni, "1hour", now - 150 * 86400, now, {"l": "liq_long", "s": "liq_short"})
    print("часы: монет", lh.symbol.nunique() if len(lh) else 0)
    old = "results/liq_check/real.csv.gz"
    if os.path.exists(old):
        r = pd.read_csv(old)
        r = r[r.interval == "1hour"].rename(columns={"real_long": "liq_long", "real_short": "liq_short"})
        r = r[r.symbol.isin(uni)][["t", "liq_long", "liq_short", "symbol"]]
        lh = pd.concat([r, lh], ignore_index=True).drop_duplicates(["symbol", "t"], keep="last")
    lh.to_csv(f"{OUT}/liq_h.csv.gz", index=False)
    ld = cz("liquidation-history", uni, "daily", d0, now, {"l": "liq_long", "s": "liq_short"})
    ld.to_csv(f"{OUT}/liq_d.csv.gz", index=False)
    od = cz("open-interest-history", uni, "daily", d0 - 3 * 86400, now, {"c": "oi_close"})
    od[["symbol", "t", "oi_close"]].to_csv(f"{OUT}/oi_d.csv.gz", index=False)
    print("дни: ликвидации", ld.symbol.nunique() if len(ld) else 0, "OI", od.symbol.nunique() if len(od) else 0)
    have = set(lh.symbol) | set(ld.symbol) if len(lh) or len(ld) else set()
    json.dump([s for s in uni if s in have], open(f"{OUT}/universe.json", "w"))


# ---------------- сделки ----------------
def simulate(o, h, l, c, i0, side, hold):
    """Вход по открытию свечи i0. side=+1 лонг, −1 шорт. Возвращает dict результатов (% цены, с комиссией) или None."""
    n = len(c)
    if i0 + hold > n:
        return None
    e = o[i0]
    if not e > 0:
        return None
    hh, ll, cc = h[i0:i0 + hold], l[i0:i0 + hold], c[i0:i0 + hold]
    res = {}
    for tp, sl in TPSL:
        if side < 0:
            stop, take = e * (1 + sl / 100), e * (1 - tp / 100)
            hs, ht = hh >= stop, ll <= take
        else:
            stop, take = e * (1 - sl / 100), e * (1 + tp / 100)
            hs, ht = ll <= stop, hh >= take
        js = int(np.argmax(hs)) if hs.any() else hold
        jt = int(np.argmax(ht)) if ht.any() else hold
        if js == hold and jt == hold:
            r = side * (cc[-1] / e - 1) * 100
        elif js <= jt:
            r = -sl
        else:
            r = tp
        res[f"{tp:g}/{sl:g}"] = r - FEE
    for k in TIMEX:
        if k <= hold:
            res[f"{k}ч"] = side * (cc[k - 1] / e - 1) * 100 - FEE
    adverse = (hh.max() / e - 1) * 100 if side < 0 else (1 - ll.min() / e) * 100
    res["против, %"] = adverse
    return res


def coin(sym, lh, ld, od):
    k = PL.klines_any("um", sym, D0, dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 500:
        return []
    k = k[k.open_time >= pd.Timestamp(D0, tz="UTC").timestamp() * 1000].reset_index(drop=True)
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    rows = []

    # ---- часы ----
    g = lh[lh.symbol == sym]
    if len(g):
        cfg = CFG["h"]
        liq = pd.DataFrame({"t": t}).merge(g.assign(t=g.t.astype("int64") * 1000)[["t", "liq_long", "liq_short"]],
                                           on="t", how="left")
        first = g.t.min() * 1000
        have = (t >= first) & (t <= g.t.max() * 1000)
        LL, SS = liq.liq_long.fillna(0).to_numpy(copy=True), liq.liq_short.fillna(0).to_numpy(copy=True)
        LL[~have], SS[~have] = np.nan, np.nan
        normL = pd.Series(LL).shift(1).rolling(cfg["norm"], min_periods=cfg["norm"]).mean().to_numpy()
        normS = pd.Series(SS).shift(1).rolling(cfg["norm"], min_periods=cfg["norm"]).mean().to_numpy()
        cand = np.where(have & ((SS >= MIN_USD) & (SS >= DOM * LL) & (c > o) |
                                (LL >= MIN_USD) & (LL >= DOM * SS) & (c < o)))[0]
        oi = None
        if len(cand):
            days = {pd.Timestamp(t[i], unit="ms", tz="UTC").date() + dt.timedelta(days=d) for i in cand for d in (-1, 0)}
            met = PL.metrics(sym, days)
            oi = PL.AsOf(met, "oi_usd")
        for i in cand:
            short = SS[i] >= DOM * LL[i] and c[i] > o[i]
            side, v, nm = (-1, SS[i], normS[i]) if short else (+1, LL[i], normL[i])
            r = simulate(o, h, l, c, i + 1, side, cfg["hold"])
            if r is None:
                continue
            oiu = oi.at(t[i], 2 * H) if oi is not None else np.nan
            rows.append({"tf": "h", "symbol": sym, "t": int(t[i]), "side": side, "liq $": v,
                         "% OI": v / oiu * 100 if oiu and np.isfinite(oiu) else np.nan,
                         "x нормы": v / nm if nm and np.isfinite(nm) and nm > 0 else np.nan,
                         "OI $": oiu, "рост свечи, %": (c[i] / o[i] - 1) * 100, **r})

    # ---- дни ----
    g = ld[ld.symbol == sym]
    if len(g):
        cfg = CFG["d"]
        day = (t // (24 * H)) * 24 * H
        kd = pd.DataFrame({"day": day, "o": o, "c": c}).groupby("day").agg(o=("o", "first"), c=("c", "last"), n=("o", "size"))
        kd = kd[kd.n == 24]
        g = g.assign(day=g.t.astype("int64") * 1000).set_index("day")
        dd = kd.join(g[["liq_long", "liq_short"]], how="left")
        lo, hi = g.index.min(), g.index.max()
        dd = dd[(dd.index >= lo) & (dd.index <= hi)]
        dd[["liq_long", "liq_short"]] = dd[["liq_long", "liq_short"]].fillna(0)
        for col in ("liq_long", "liq_short"):
            dd["n_" + col] = dd[col].shift(1).rolling(cfg["norm"], min_periods=cfg["norm"]).mean()
        oo = od[od.symbol == sym]
        oimap = dict(zip(oo.t.astype("int64") * 1000, oo.oi_close))
        idx = {int(x): j for j, x in enumerate(t)}
        for d, x in dd.iterrows():
            short = x.liq_short >= MIN_USD and x.liq_short >= DOM * x.liq_long and x.c > x.o
            long_ = x.liq_long >= MIN_USD and x.liq_long >= DOM * x.liq_short and x.c < x.o
            if not (short or long_):
                continue
            side, v, nm = (-1, x.liq_short, x.n_liq_short) if short else (+1, x.liq_long, x.n_liq_long)
            j = idx.get(int(d) + 24 * H)
            if j is None:
                continue
            r = simulate(o, h, l, c, j, side, cfg["hold"])
            if r is None:
                continue
            oiu = oimap.get(int(d) - 24 * H, np.nan)
            rows.append({"tf": "d", "symbol": sym, "t": int(d), "side": side, "liq $": v,
                         "% OI": v / oiu * 100 if oiu and np.isfinite(oiu) else np.nan,
                         "x нормы": v / nm if nm and np.isfinite(nm) and nm > 0 else np.nan,
                         "OI $": oiu, "рост свечи, %": (x.c / x.o - 1) * 100, **r})
    return rows


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    lh, ld, od = (pd.read_csv(f"{OUT}/{f}.csv.gz") for f in ("liq_h", "liq_d", "oi_d"))
    mine = [s for i, s in enumerate(sorted(syms)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, lh, ld, od)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/ev{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} событий, монет {len(mine)}")


# ---------------- итог ----------------
EXITS = [f"{a:g}/{b:g}" for a, b in TPSL] + [f"{k}ч" for k in TIMEX]


def cooldown(g, hours):
    keep, last = [], {}
    for i, s, t in zip(g.index, g.symbol, g.t):
        if s in last and t - last[s] < hours * H:
            continue
        last[s] = t
        keep.append(i)
    return g.loc[keep]


def stats(g, col, weeks):
    r = g.sort_values("t")[col].dropna().to_numpy()
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


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["время (Киев)"] = ev.t.map(PL.kyiv)
    ev.to_csv(f"{OUT}/events.csv.gz", index=False)
    w = open(f"{OUT}/liq_fade_summary.md", "w").write
    w("# Вход против ликвидаций (стратегия блогера)\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w("Как читать: «средняя» — средний результат сделки в % цены после комиссии (2x плечо — умножь на 2); "
      "«сумма» — если брать каждую сделку одинаковой суммой; PF — заработано / потеряно (больше 1 — в плюсе); "
      "«просадка» — худшее падение суммы от пика, в тех же %; «против» — насколько цена успела уйти против нас за время сделки.\n\n")
    trades_out = []
    for tf, cfg in CFG.items():
        e = ev[ev.tf == tf]
        if not len(e):
            w(f"## {cfg['name']}: нет данных\n\n")
            continue
        weeks = (e.t.max() - e.t.min()) / (7 * 24 * H) or 1
        w(f"## {cfg['name']}: {PL.kyiv(e.t.min())[:10]} — {PL.kyiv(e.t.max())[:10]} ({weeks:.0f} нед.), "
          f"монет {e.symbol.nunique()}\n\n")
        for side, sname in ((-1, "ШОРТ после выноса шортов (как у блогера)"), (+1, "ЛОНГ после выноса лонгов (зеркало)")):
            es = e[e.side == side]
            w(f"### {sname}\n\n")
            look, grid = [], []
            for meas, vals in (("% OI", cfg["oi"]), ("x нормы", cfg["x"])):
                for v in vals:
                    g = cooldown(es[es[meas] >= v], 24 * cfg["cool"] if tf == "d" else cfg["cool"])
                    if len(g) < 5:
                        continue
                    lbl = f"{meas} ≥ {v:g}"
                    trades_out.append(g.assign(сигнал=f"{tf} {'шорт' if side < 0 else 'лонг'} {lbl}"))
                    adv = g["против, %"]
                    look.append({"порог": lbl, "сигналов": len(g), "в неделю": len(g) / weeks,
                                 **{f"через {k}ч, % (медиана)": g[f"{k}ч"].median() for k in TIMEX},
                                 "против ≥5%": (adv >= 5).mean() * 100, "против ≥10%": (adv >= 10).mean() * 100,
                                 "против ≥20%": (adv >= 20).mean() * 100})
                    for x in EXITS:
                        s = stats(g, x, weeks)
                        if s:
                            grid.append({"порог": lbl, "выход": x, **s})
            if not look:
                w("Сигналов почти нет.\n\n")
                continue
            w("Что происходит после сигнала (результат в нашу сторону, с комиссией; «против» — доля сделок, где цена "
              "сначала ушла против нас на столько):\n\n")
            w(pd.DataFrame(look).round(2).to_markdown(index=False) + "\n\n")
            gd = pd.DataFrame(grid)
            w("Тейк/стоп 10/10 по всем порогам:\n\n")
            w(gd[gd.выход == "10/10"].round(2).to_markdown(index=False) + "\n\n")
            ok = gd[gd.сделок >= max(10, weeks)]                      # хотя бы ~1 сделка в неделю
            w("Лучшие 12 сочетаний порог × выход (не меньше ~1 сделки в неделю), по средней сделке:\n\n")
            w(ok.sort_values("средняя, %", ascending=False).head(12).round(2).to_markdown(index=False) + "\n\n")
            if len(ok):
                b = ok.sort_values("средняя, %", ascending=False).iloc[0]
                meas, v = b.порог.split(" ≥ ")
                g = cooldown(es[es[meas] >= float(v)], 24 * cfg["cool"] if tf == "d" else cfg["cool"])
                ts = pd.to_datetime(g.t, unit="ms")
                g = g.assign(месяц=ts.dt.strftime("%Y-%m") if tf == "h" else ts.dt.to_period("Q").astype(str))
                g["OI группа"] = pd.cut(g["OI $"], [0, 20e6, 100e6, np.inf], labels=["OI < $20M", "$20–100M", "> $100M"])
                w(f"Лучшее ({b.порог}, выход {b.выход}) — по {'месяцам' if tf == 'h' else 'кварталам'} и по размеру OI:\n\n")
                per = [{"период": m_, **stats(x, b.выход, 1)} for m_, x in g.groupby("месяц")]
                w(pd.DataFrame(per).drop(columns=["в неделю"]).round(2).to_markdown(index=False) + "\n\n")
                per = [{"OI": m_, **stats(x, b.выход, 1)} for m_, x in g.groupby("OI группа", observed=True)]
                w(pd.DataFrame(per).drop(columns=["в неделю"]).round(2).to_markdown(index=False) + "\n\n")
    if trades_out:
        pd.concat(trades_out).to_csv(f"{OUT}/trades.csv.gz", index=False)
    print(open(f"{OUT}/liq_fade_summary.md").read()[:8000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
