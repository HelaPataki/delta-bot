"""
Перегретые лонги / шорты: фандинг + открытый интерес + цена стоит. 4h-свечи, сигналы с 2026-01-01.

Всё берётся на момент закрытия сигнальной свечи (T), только из того, что уже было известно:
  открытый интерес (OI) — последняя 5-минутная запись <= T (архив Binance metrics); рост к значению 24 ч и 4 ч назад;
  фандинг — средний из уже начисленных за последние 24 ч (в пересчёте на 8 ч);
  цена — изменение за 24 ч (6 свечей).
Шорт («перегретые лонги»): фандинг >= F, OI за 24 ч >= +X%, цена за 24 ч <= +2%.
Лонг (зеркально, «перегретые шорты»): фандинг <= −F, OI за 24 ч >= +X%, цена за 24 ч >= −2%.
Учитывается только первая свеча серии, где условие стало выполняться.
Результат — направление от закрытия свечи за 10 дней (без сделок).

Запуск частями (SHARD = 0..N-1 из NSHARDS) сохраняет признаки в results/oi_funding/parts/,
затем MODE=combine собирает сводку.
"""

import io
import os
import sys
import time
import zipfile
import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.environ.get("OUT_DIR", "results/oi_funding")
SIG_START = pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000
Hst.START = "2025-12-01"            # декабрь — только для расчёта окон до первых сигналов
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
HOR = 60
H = 3600 * 1000


def get_zip_csv(url):
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f)


def funding(sym):
    frames = []
    for m in Hst.months(dt.date(2025, 12, 1), dt.datetime.now(dt.timezone.utc).date()):
        df = get_zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    f = pd.concat(frames)
    f.columns = [str(c).strip().lower() for c in f.columns]
    if "last_funding_rate" not in f.columns:                  # файлы без заголовка
        f = f.iloc[:, :3]
        f.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    f = f.rename(columns={"calc_time": "t", "last_funding_rate": "rate", "funding_interval_hours": "hours"})
    if "hours" not in f:
        f["hours"] = 8
    f["rate8"] = f.rate.astype(float) * 8 / f.hours.astype(float)
    return f[["t", "rate8"]].astype(float).sort_values("t")


def open_interest(sym):
    d, today = dt.date(2025, 12, 30), dt.datetime.now(dt.timezone.utc).date()
    days = []
    while d < today:
        days.append(d)
        d += dt.timedelta(days=1)

    def one(day):
        df = get_zip_csv(f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip")
        if df is None:
            return None
        ts = pd.to_datetime(df.create_time, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ms]")
        return pd.DataFrame({"t": ts.astype("int64").astype(float),
                             "oi": df.sum_open_interest.astype(float)})

    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, days) if p is not None]
    if not parts:
        return None
    return pd.concat(parts).sort_values("t").drop_duplicates("t")


def asof(times, values, query):
    """Последнее значение с временем <= query (NaN, если нет)."""
    idx = np.searchsorted(times, query, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)
    return out


def coin(sym, mcap_now):
    k = Hst.load_klines(sym)
    if k is None or len(k) < 50:
        return None
    oi = open_interest(sym)
    fu = funding(sym)
    if oi is None or fu is None:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    qv = k.quote_volume.to_numpy()
    T = k.open_time.to_numpy() + 4 * H                       # момент закрытия свечи
    ot, ov = oi.t.to_numpy(), oi.oi.to_numpy()
    oi_now = asof(ot, ov, T)
    # запись открытого интереса не старше 30 минут, иначе считаем, что данных нет
    fresh = (T - asof(ot, ot.astype(float), T)) <= 30 * 60 * 1000
    oi_now = np.where(fresh, oi_now, np.nan)
    oi24, oi4 = asof(ot, ov, T - 24 * H), asof(ot, ov, T - 4 * H)
    ft, fr = fu.t.to_numpy(), fu.rate8.to_numpy()
    cs = np.concatenate([[0], np.cumsum(fr)])
    i1, i0 = np.searchsorted(ft, T, side="right"), np.searchsorted(ft, T - 24 * H, side="right")
    cnt = i1 - i0
    fund24 = np.where(cnt > 0, (cs[i1] - cs[i0]) / np.maximum(cnt, 1), np.nan) * 100   # в %
    px24 = (c / pd.Series(c).shift(6).to_numpy() - 1) * 100
    mcap = mcap_now * c / c[-1]

    n = len(k)
    Hw, Lw = Hst.window(h, n), Hst.window(l, n)
    up, dn = Hw / c[:, None] - 1, Lw / c[:, None] - 1

    def first(hit):
        a = hit.any(axis=1)
        return np.where(a, np.argmax(hit, axis=1), 10 ** 6)

    u4, d4 = first(up >= 0.04), first(dn <= -0.04)
    u10, d10 = first(up >= 0.10), first(dn <= -0.10)
    df = pd.DataFrame({
        "symbol": sym, "open_time": k.open_time.to_numpy(), "close": c, "turnover": qv, "mcap": mcap,
        "fund24": fund24, "oi_chg24": (oi_now / oi24 - 1) * 100, "oi_chg4": (oi_now / oi4 - 1) * 100, "px24": px24,
        "complete": np.sum(~np.isnan(Hw), axis=1) == HOR,
        "first_dn": d4 < u4, "first_up": u4 < d4, "dn10": d10 < 10 ** 6, "up10": u10 < 10 ** 6,
        "dn10_before_up10": d10 < u10, "up10_before_dn10": u10 < d10,
    })
    keep = (df.open_time >= SIG_START) & (df.turnover >= Hst.MIN_TURNOVER) & df.mcap.between(Hst.MCAP_MIN, Hst.MCAP_MAX)
    return df[keep]


def run_shard():
    os.makedirs(f"{OUT_DIR}/parts", exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    mine = [s for i, s in enumerate(s for s in syms if Hst.base_asset(s) in caps) if i % NSHARDS == SHARD]
    print(f"часть {SHARD}: монет {len(mine)}")

    def work(sym):
        try:
            return coin(sym, caps[Hst.base_asset(sym)])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(4) as ex:
        for i, p in enumerate(ex.map(work, mine), 1):
            if p is not None and len(p):
                parts.append(p)
            if i % 20 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    allp = pd.concat(parts)
    print("заполнено:", {c: f"{allp[c].notna().mean() * 100:.0f}%" for c in ("fund24", "oi_chg24", "oi_chg4", "px24")})
    allp.to_csv(f"{OUT_DIR}/parts/part{SHARD}.csv.gz", index=False)


def first_only(df, mask):
    m = mask.astype(int)
    prev = m.groupby(df.symbol).shift(1).fillna(0)
    return mask & (prev == 0)


def stats(g, side):
    g = g[g.complete]
    if len(g) == 0:
        return None
    if side == "шорт":
        return {"сигналов": len(g), "сначала −4%": f"{g.first_dn.mean() * 100:.0f}%",
                "сначала +4%": f"{g.first_up.mean() * 100:.0f}%", "−10% за 10д": f"{g.dn10.mean() * 100:.0f}%",
                "+10% за 10д": f"{g.up10.mean() * 100:.0f}%", "−10% раньше +10%": f"{g.dn10_before_up10.mean() * 100:.0f}%"}
    return {"сигналов": len(g), "сначала +4%": f"{g.first_up.mean() * 100:.0f}%",
            "сначала −4%": f"{g.first_dn.mean() * 100:.0f}%", "+10% за 10д": f"{g.up10.mean() * 100:.0f}%",
            "−10% за 10д": f"{g.dn10.mean() * 100:.0f}%", "+10% раньше −10%": f"{g.up10_before_dn10.mean() * 100:.0f}%"}


def combine():
    import glob
    a = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{OUT_DIR}/parts/part*.csv.gz"))], ignore_index=True)
    a = a.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print("свечей:", len(a), "заполнено:", {c: f"{a[c].notna().mean() * 100:.0f}%" for c in ("fund24", "oi_chg24", "oi_chg4", "px24")})
    has = a.fund24.notna() & a.oi_chg24.notna() & a.px24.notna()
    if not has.any():
        raise SystemExit("нет свечей с фандингом и открытым интересом одновременно — см. проценты выше")
    a = a[has].reset_index(drop=True)
    months = (a.open_time.max() - a.open_time.min()) / 86_400_000 / 30.4
    out = []
    for side in ("шорт", "лонг"):
        sg = 1 if side == "шорт" else -1
        flat = (a.px24 <= 2) if side == "шорт" else (a.px24 >= -2)
        rows = [{"условие": "все свечи (для сравнения)", **(stats(a, side) or {})}]
        for F in (0.03, 0.05, 0.1):
            fm = (a.fund24 >= F) if side == "шорт" else (a.fund24 <= -F)
            rows.append({"условие": f"только фандинг {'≥' if sg > 0 else '≤ −'}{F:g}%", **(stats(a[first_only(a, fm)], side) or {})})
            for X in (10, 20):
                om = a.oi_chg24 >= X
                m = first_only(a, fm & om & flat)
                r = stats(a[m], side)
                if r:
                    rows.append({"условие": f"фандинг {'≥' if sg > 0 else '≤ −'}{F:g}% + OI 24ч ≥ +{X}% + цена стоит",
                                 "в месяц": f"{m.sum() / months:.1f}", **r})
                m4 = first_only(a, fm & om & (a.oi_chg4 >= 5) & flat)
                r = stats(a[m4], side)
                if r:
                    rows.append({"условие": f"   … и OI за 4ч ≥ +5%", "в месяц": f"{m4.sum() / months:.1f}", **r})
        for X in (10, 20):
            rows.append({"условие": f"только OI 24ч ≥ +{X}%", **(stats(a[first_only(a, a.oi_chg24 >= X)], side) or {})})
            rows.append({"условие": f"OI 24ч ≥ +{X}% + цена стоит", **(stats(a[first_only(a, (a.oi_chg24 >= X) & flat)], side) or {})})
        out.append((side, pd.DataFrame(rows)))
    with open(f"{OUT_DIR}/oi_funding_summary.md", "w") as f:
        f.write("# Перегретые лонги / шорты: фандинг + открытый интерес (с 01.01.2026)\n\n")
        f.write("Всё на момент закрытия сигнальной 4h-свечи. Фандинг — средний начисленный за 24 ч (на 8 ч). "
                "OI — последняя запись до закрытия свечи. «Цена стоит» — за 24 ч не больше +2% (для лонга не меньше −2%). "
                "Направление от закрытия свечи, 10 дней. Только первая свеча серии.\n\n")
        for side, t in out:
            title = "Шорт — перегретые лонги (фандинг плюсовой)" if side == "шорт" else "Лонг — перегретые шорты (фандинг минусовой)"
            f.write(f"## {title}\n\n" + t.fillna("").to_markdown(index=False) + "\n\n")
    print(open(f"{OUT_DIR}/oi_funding_summary.md").read())


if __name__ == "__main__":
    combine() if os.environ.get("MODE") == "combine" else run_shard()
