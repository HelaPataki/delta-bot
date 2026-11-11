"""
Сбор фандинга без направления: купить монету на споте + зашортить её фьючерс на ту же сумму.
Цена не важна, доход — фандинг (шорт получает положительный, платит отрицательный) + изменение спреда спот/фьюч.
Живого бота не трогает.

Правила заданы заранее, не подбирались:
  Монеты: USDT-перпетуалы Binance, у которых есть спот на Binance.
  Каждый день в 02:00 Киева (00:00 UTC) — ранжирование по среднему фандингу за 3 прошлых дня.
  Ликвидность (вчера): оборот фьюча ≥ $20M, спота ≥ $5M.
  Портфель — топ-N монет со средним фандингом ≥ 0,01% за 8 ч (базовая ставка Binance), равными долями.
  Если подходит меньше N — остальное лежит в USDT без дохода.
  Держим сутки; монета, оставшаяся в топе, не перекладывается.
  Комиссии: вход и выход по 0,15% (спот 0,1% + фьюч 0,05%).
  Варианты: N = 1, 3, 5, 10; плюс «BTC + ETH всегда».
Доход считается на сумму позиции (notional). На депозит: при марже фьюча 1x половина денег на споте,
половина маржой → доход на депозит = ½; при 3x — ¾ (но риск ликвидации шорта при пампе выше).
"""

import datetime as dt
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

OUT = os.environ.get("OUT_DIR", "results/funding_carry")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
D0 = dt.date(2021, 1, 1)
DAY = 86_400_000
FEE = 0.15
NS = [1, 3, 5, 10]


def zip_df(url):
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    return df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].astype(float)


def daily_klines(base, sym):
    today = dt.datetime.now(dt.timezone.utc).date()
    this_month = dt.date(today.year, today.month, 1)
    urls = []
    for m in Hst.months(D0, today):
        if m < this_month:
            urls.append(f"{Hst.ARCHIVE}/{base}/monthly/klines/{sym}/1d/{sym}-1d-{m:%Y-%m}.zip")
        else:
            d = m
            while d < today:
                urls.append(f"{Hst.ARCHIVE}/{base}/daily/klines/{sym}/1d/{sym}-1d-{d:%Y-%m-%d}.zip")
                d += dt.timedelta(days=1)
    fr = [f for f in map(zip_df, urls) if f is not None]
    if not fr:
        return None
    k = pd.concat(fr).iloc[:, :8]
    k.columns = ["t", "o", "h", "l", "c", "v", "ct", "qv"]
    k["t"] = np.where(k.t > 1e14, k.t // 1000, k.t)
    return k.drop_duplicates("t").set_index("t").sort_index()


def funding_raw(sym):
    fr = []
    for m in Hst.months(D0, dt.datetime.now(dt.timezone.utc).date()):
        f = zip_df(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if f is not None:
            f = f.iloc[:, [0, f.shape[1] - 1]]
            f.columns = ["t", "rate"]
            fr.append(f)
    if not fr:
        return None
    return pd.concat(fr).drop_duplicates("t").sort_values("t")


def coin(sym):
    spot_sym = sym                                    # только монеты без множителя (1000PEPE и т.п. пропускаем)
    fut = daily_klines("data/futures/um", sym)
    if fut is None or len(fut) < 30:
        return None
    spot = daily_klines("data/spot", spot_sym)
    if spot is None or len(spot) < 30:
        return None
    f = funding_raw(sym)
    if f is None:
        return None
    # фандинг дня D = расчёты с временем в (D 00:00, D+1 00:00] UTC, в %
    tt = np.round(f.t.to_numpy() / 60000) * 60000
    day_open = (np.ceil(tt / DAY) - 1) * DAY
    fd = pd.Series(f.rate.to_numpy() * 100).groupby(day_open).sum()
    cnt = pd.Series(1, index=f.index).groupby(day_open).sum()
    idx = fut.index.intersection(spot.index)
    d = pd.DataFrame({"symbol": sym, "t": idx, "fut_c": fut.c.reindex(idx).to_numpy(), "spot_c": spot.c.reindex(idx).to_numpy(),
                      "fut_qv": fut.qv.reindex(idx).to_numpy(), "spot_qv": spot.qv.reindex(idx).to_numpy(),
                      "fund": fd.reindex(idx).fillna(0).to_numpy(), "n_fund": cnt.reindex(idx).fillna(0).to_numpy(),
                      "fut_h": fut.h.reindex(idx).to_numpy()})
    return d


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = [s for s in Hst.list_symbols() if s == Hst.base_asset(s) + "USDT"]
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(syms)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(work, mine) if p is not None]
    out = pd.concat(parts) if parts else pd.DataFrame()
    out.to_csv(f"{OUT}/parts/days{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, со спотом {len(parts)}")


def simulate(a, N=None, fixed=None):
    """Возвращает дневной ряд дохода портфеля (% на notional) и журнал."""
    a = a.sort_values(["symbol", "t"]).copy()
    g = a.groupby("symbol")
    a["avg8"] = g.fund.transform(lambda x: x.shift(1).rolling(3).sum()) / 9      # средний % за 8 ч за 3 дня до D
    a["liq"] = (g.fut_qv.shift(1) >= 20e6) & (g.spot_qv.shift(1) >= 5e6)
    a["ret_basis"] = (g.spot_c.pct_change() - g.fut_c.pct_change()) * 100
    a["spike"] = (a.fut_h / g.fut_c.shift(1) - 1) * 100
    days = np.sort(a.t.unique())
    by_day = {t: x for t, x in a.groupby("t")}
    held, rows, log = set(), [], []
    for t in days:
        x = by_day[t]
        if fixed:
            pick = set(x[x.symbol.isin(fixed)].symbol)
            slots = len(fixed)
        else:
            c = x[x.liq & (x.avg8 >= 0.01)].sort_values("avg8", ascending=False)
            pick = set(c.symbol.head(N))
            slots = N
        if t < pd.Timestamp("2021-01-04", tz="UTC").timestamp() * 1000:
            held = pick
            continue
        changes = len(pick - held) + len(held - pick)
        xs = x.set_index("symbol")
        r = 0.0
        for s in pick:
            fr = xs.at[s, "fund"]
            b = xs.at[s, "ret_basis"]
            r += (fr + (0 if np.isnan(b) else b)) / slots
            log.append({"t": t, "symbol": s, "fund": fr, "basis": b, "spike": xs.at[s, "spike"]})
        r -= changes * FEE / slots
        rows.append({"t": t, "ret": r, "n": len(pick)})
        held = pick
    return pd.DataFrame(rows), pd.DataFrame(log)


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True)
    variants = {f"топ-{n}": dict(N=n) for n in NS}
    variants["BTC + ETH всегда"] = dict(fixed=["BTCUSDT", "ETHUSDT"])
    w = open(f"{OUT}/funding_carry_summary.md", "w").write
    w("# Сбор фандинга: спот + шорт фьючерса\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
    year_rows, month_all = [], []
    for name, kw in variants.items():
        d, log = simulate(a, **kw)
        d["date"] = pd.to_datetime(d.t, unit="ms")
        d["год"] = d.date.dt.year
        d["мес"] = d.date.dt.strftime("%Y-%m")
        for y, g in d.groupby("год"):
            eq = (1 + g.ret / 100).cumprod()
            mdd = (eq / eq.cummax() - 1).min() * 100
            mon = g.groupby("мес").ret.apply(lambda r: ((1 + r / 100).prod() - 1) * 100)
            tot = (eq.iloc[-1] - 1) * 100
            year_rows.append({"вариант": name, "год": y, "дней": len(g),
                              "доход на позицию, %": round(tot, 1),
                              "на депозит при 1x, %": round(tot / 2, 1),
                              "на депозит при 3x, %": round(tot * 0.75, 1),
                              "просадка, %": round(mdd, 1),
                              "месяцев в плюсе": f"{(mon > 0).sum()} из {len(mon)}",
                              "худший месяц, %": round(mon.min(), 1),
                              "среднее монет": round(g.n.mean(), 1)})
        if len(log):
            log["date"] = pd.to_datetime(log.t, unit="ms").dt.strftime("%Y-%m-%d")
            log.to_csv(f"{OUT}/log_{name.replace(' ', '_').replace('+', '')}.csv.gz", index=False)
            month_all.append((name, log))
    yr = pd.DataFrame(year_rows)
    yr.to_csv(f"{OUT}/funding_carry_years.csv", index=False)
    w("## По годам\n\n" + yr.to_markdown(index=False) + "\n\n")
    w("## Риск: самые сильные дневные пампы держимых монет (шорт-ногу может ликвидировать при 3x)\n\n")
    for name, log in month_all:
        top = log.sort_values("spike", ascending=False).head(5)
        w(f"**{name}:** " + "; ".join(f"{r.symbol} {r.date} +{r.spike:.0f}%" for r in top.itertuples()) + "\n\n")
    w("## Какие монеты чаще всего в портфеле (топ-5)\n\n")
    for name, log in month_all:
        if name == "топ-5":
            vc = log.symbol.value_counts().head(20)
            w(", ".join(f"{s} {v} дн." for s, v in vc.items()) + "\n\n")
    print(yr.to_string())


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
