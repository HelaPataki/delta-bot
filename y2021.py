"""
Отдельный прогон финальной стратегии (шорт + лонг) на 2021–2024 — годы, на которых пороги не подбирались.
Живой бот и старые скрипты не меняются: отсюда только вызываются их функции.

Шорт: фандинг 24ч ≤ −0,2% + OI 24ч ≥ +60% + цена 24ч ≥ −2%, первый сигнал подряд; вход по рынку через 8 мин,
      1:1,8 (стоп 4,3%), выход через 24 ч.
Лонг: объём ≥ x5, дельта ≤ −17%, POC в нижних 30%, закрытие не ниже POC; условный ордер POC + 5 тиков с 8-й минуты,
      живёт до конца следующей свечи; 1:4, до 10 дней.
Монеты: все USDT-перпетуалы архива, включая делистнутые. Два варианта:
  А — с фильтром капитализации $10M–$400M (только монеты, которые есть на CoinGecko, как раньше);
  Б — без капитализации, только оборот свечи ≥ $50k.

MODE=long_shard / short_shard (SHARD, NSHARDS) → части; MODE=report → сделки, таблицы по годам, симуляция $100.
"""

import datetime as dt
import glob
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("SIG_FROM", "2021-01-01")
os.environ.setdefault("DELAY", "8")
os.environ.setdefault("RR_LIST", "4")
os.environ.setdefault("HOLD_LIST", "240")

import numpy as np
import pandas as pd

import history as Hst
import oi_funding as OF
import oi_sim_data as OS
import long_sim_data as LS
import poc as P

OUT = "results/y2021"
FROM, END = "2021-01-01", "2025-01-01"
FROM_MS = pd.Timestamp(FROM, tz="UTC").timestamp() * 1000
END_MS = pd.Timestamp(END, tz="UTC").timestamp() * 1000
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
KYIV = "Europe/Kyiv"

# без фильтра капитализации на этапе сбора; неизвестная капитализация = −1 (отсеивается в варианте А)
Hst.MCAP_MIN, Hst.MCAP_MAX = -np.inf, np.inf

# ---------- фандинг во время сделки: с декабря 2020 (в старом скрипте — только с декабря 2024) ----------
_fund = {}


def funding_series(sym):
    if sym in _fund:
        return _fund[sym]
    frames = []
    for m in Hst.months(dt.date(2020, 12, 1), dt.date(2025, 2, 1)):
        r = Hst.get(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if r is None or r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "last_funding_rate" not in df.columns:
            df = df.iloc[:, :3]
            df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        frames.append(df[["calc_time", "last_funding_rate"]].astype(float))
    f = pd.concat(frames).sort_values("calc_time") if frames else pd.DataFrame(columns=["calc_time", "last_funding_rate"])
    _fund[sym] = (f.calc_time.to_numpy(), f.last_funding_rate.to_numpy())
    return _fund[sym]


OS.funding_series = funding_series
LS.funding_series = funding_series


# ---------- OI только до конца 2024 ----------
def open_interest(sym):
    d, last = dt.date.fromisoformat(FROM) - dt.timedelta(days=2), dt.date.fromisoformat(END) + dt.timedelta(days=1)
    days = []
    while d <= last:
        days.append(d)
        d += dt.timedelta(days=1)

    def one(day):
        df = OF.get_zip_csv(f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip")
        if df is None:
            return None
        ts = pd.to_datetime(df.create_time, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ms]")
        return pd.DataFrame({"t": ts.astype("int64").astype(float), "oi": df.sum_open_interest.astype(float)})

    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, days) if p is not None]
    if not parts:
        return None
    return pd.concat(parts).sort_values("t").drop_duplicates("t")


OF.open_interest = open_interest


def my_symbols():
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    print(f"часть {SHARD}: монет {len(mine)} из {len(syms)}", flush=True)
    return mine, caps


# ---------- лонг ----------
def long_shard():
    os.makedirs(f"{OUT}/long_parts", exist_ok=True)
    mine, caps = my_symbols()
    Hst.START = "2020-12-01"

    def work(sym):
        try:
            k = Hst.load_klines(sym)
            if k is None or len(k) < 20:
                return []
            a = Hst.analyse(sym, k, caps.get(Hst.base_asset(sym), -1.0))
            a = a[(a.open_time >= FROM_MS) & (a.open_time < END_MS) & (a.delta_pct <= -17) & (a.vol_ratio >= 5)]
            out = []
            for r in a.to_dict("records"):
                r["candle_open_kyiv"] = pd.Timestamp(r["open_time"], unit="ms", tz="UTC").tz_convert(KYIV).strftime("%Y-%m-%d %H:%M")
                p = P.poc_for(r)
                if not (p["poc"] == p["poc"]) or p["poc_pos"] > 30 or r["close"] < p["poc"]:
                    continue
                r.update(p)
                for t in LS.run(r):
                    t["mcap"] = r["mcap_est"]
                    out.append(t)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: {e}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(work, mine), 1):
            rows += r
            if i % 20 == 0:
                print(f"{i}/{len(mine)}, сделок {len(rows)}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(f"{OUT}/long_parts/part{SHARD}.csv.gz", index=False)
    print("лонг-сделок:", len(rows))


# ---------- шорт ----------
def short_shard():
    os.makedirs(f"{OUT}/short_parts", exist_ok=True)
    mine, caps = my_symbols()

    def work(sym):
        try:
            df = OF.coin(sym, caps.get(Hst.base_asset(sym), -1.0))
            return None if df is None else df[df.open_time < END_MS]
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(4) as ex:
        for i, p in enumerate(ex.map(work, mine), 1):
            if p is not None and len(p):
                parts.append(p)
            if i % 20 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    if not parts:
        print("в этой части нет данных")
        return
    allp = pd.concat(parts)
    print("заполнено:", {c: f"{allp[c].notna().mean() * 100:.0f}%" for c in ("fund24", "oi_chg24", "px24")})
    allp.to_csv(f"{OUT}/short_parts/part{SHARD}.csv.gz", index=False)


# ---------- отчёт ----------
def net(df):
    return df.pnl_pct + df.funding_pct - 0.1


def year_table(df, side):
    df = df.assign(год=pd.to_datetime(df.open_time, unit="ms").dt.year, net=net(df))
    rows = []
    for y, g in list(df.groupby("год")) + [("все", df)]:
        months = g.groupby(pd.to_datetime(g.open_time, unit="ms").dt.to_period("M")).net.sum()
        rows.append({"год": y, "сторона": side, "сделок": len(g), "винрейт": f"{(g.result == 'тейк').mean() * 100:.0f}%",
                     "% на сделку": f"{g.net.mean():+.2f}%", "месяцев в плюсе": f"{(months > 0).sum()} из {len(months)}"})
    return rows


def sim_rows(sh, lg, label):
    import combo_sim as CS
    rows = []
    years = sorted(set(pd.to_datetime(pd.concat([sh.open_time, lg.open_time]), unit="ms").dt.year))
    for y in years + ["2021–2024"]:
        if y == "2021–2024":
            s, l = sh, lg
        else:
            s = sh[pd.to_datetime(sh.open_time, unit="ms").dt.year == y]
            l = lg[pd.to_datetime(lg.open_time, unit="ms").dt.year == y]
        ev = sorted(CS.events(s, "шорт") + CS.events(l, "лонг"), key=lambda e: e[0])
        if not ev:
            continue
        row = {"вариант": label, "период": y, "шортов": len(s), "лонгов": len(l)}
        for lev in (1, 2):
            res = [CS.simulate(ev, lev, np.random.default_rng(i)) for i in range(CS.N_RUNS)]
            fin, dd = np.array([r[0] for r in res]), np.array([r[1] for r in res])
            row[f"x{lev} средний"] = f"${np.median(fin):,.0f}"
            row[f"x{lev} плохой (5%)"] = f"${np.percentile(fin, 5):,.0f}"
            row[f"x{lev} просадка тяжёлая"] = f"{np.percentile(dd, 5) * 100:.0f}%"
        rows.append(row)
        print(row, flush=True)
    return rows


def report():
    os.makedirs(OUT, exist_ok=True)
    # шорт-сделки из собранных частей (логика отбора и сделки — из oi_sim_data)
    OS.IN_DIR, OS.OUT_DIR = f"{OUT}/short_parts", f"{OUT}/short_trades"
    OS.RRS, OS.HOLDS, OS.DELAY = [1.8], [24.0], 8
    os.environ["ONLY_FILTER"] = "1"
    os.makedirs(OS.OUT_DIR, exist_ok=True)
    OS.main()
    sh = pd.read_csv(f"{OS.OUT_DIR}/oi_sim_trades.csv.gz")
    parts = pd.concat(pd.read_csv(p) for p in glob.glob(f"{OUT}/short_parts/*.csv.gz") if os.path.getsize(p) > 50)
    sh = sh.merge(parts[["symbol", "open_time", "mcap"]].drop_duplicates(["symbol", "open_time"]), on=["symbol", "open_time"], how="left")
    sh = sh[(sh.rr == 1.8) & (sh.hold_h == 24)]
    lg = pd.concat(pd.read_csv(p) for p in glob.glob(f"{OUT}/long_parts/*.csv.gz") if os.path.getsize(p) > 50)
    lg = lg[(lg.rr == 4.0) & (lg.hold_h == 240.0)]
    sh.to_csv(f"{OUT}/short_trades_all.csv.gz", index=False)
    lg.to_csv(f"{OUT}/long_trades_all.csv.gz", index=False)

    capA = lambda d: d[d.mcap.between(10e6, 400e6)]  # noqa: E731
    variants = [("А: капитализация $10M–$400M", capA(sh), capA(lg)), ("Б: все монеты (оборот ≥ $50k)", sh, lg)]

    out = ["# Финальная стратегия на 2021–2024 (годы, на которых пороги не подбирались)\n",
           "Шорт: фандинг ≤ −0,2% + OI ≥ +60%, вход через 8 мин, 1:1,8, выход 24 ч. "
           "Лонг: x5, дельта ≤ −17%, POC внизу, условный ордер POC + 5 тиков, 1:4, до 10 дней. "
           "Комиссия 0,1%, фандинг учтён.\n",
           "Для сравнения: 2025–2026 (на них подбирали) — шорт +1,02% на сделку, лонг +2,39%; $100 → $1,135 (x1), $6,276 (x2).\n"]
    first = pd.to_datetime(parts[parts.oi_chg24.notna()].open_time, unit="ms").min() if len(parts) else None
    out.append(f"Первая свеча с данными открытого интереса в архиве: {first}.\n")
    for label, s, l in variants:
        out.append(f"## {label}\n")
        out.append(pd.DataFrame(year_table(s, "шорт") + year_table(l, "лонг")).to_markdown(index=False) + "\n")
    out.append("## Симуляция $100, 2 слота, шорт + лонг, каждый год отдельно и весь период\n")
    rows = []
    for label, s, l in variants:
        rows += sim_rows(s, l, label)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    open(f"{OUT}/y2021_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    {"long_shard": long_shard, "short_shard": short_shard, "report": report}[os.environ.get("MODE", "report")]()
