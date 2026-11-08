"""
Что было ДО пампа. Исследование событий с контрольной группой. 4h- или 1h-свечи (INTERVAL) + архив Binance metrics + фандинг.

Памп: от закрытия свечи (точка отсчёта) high за следующие 72 ч >= +30%. Окна признаков заданы в часах;
норма объёма — 12 предыдущих свечей (на 1h это 12 ч), как в ботах.
Если так выходит у нескольких свечей подряд — это один памп. Стартом считается последняя свеча «базы»
перед взлётом: самая поздняя свеча серии, чьё закрытие в пределах +3% от минимального закрытия серии.
(Самая ранняя не годится: при пампе одной свечой она оказалась бы за 72 ч до него, посреди боковика.)
Капа $10M–$400M (оценка: текущая капа × цена / текущая цена), оборот свечи >= $50K.

Контроль — все остальные подходящие свечи, кроме самого пампа и 72 ч после старта.
Для каждого признака: доля пампов в группе и «подъём» = во сколько раз чаще среднего.
Рядом — доля дампов (−20% за 72 ч), чтобы видеть: признак предсказывает рост или просто «будет сильное движение».

Все признаки — только из того, что известно на закрытии свечи старта (как в живой торговле).

Режимы: MODE=prep (список монет + капа), затем SHARD=0..N-1 (признаки по частям), затем MODE=combine (сводка).
"""

import json
import os
import sys
import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import oi_funding as OF

warnings.filterwarnings("ignore", category=RuntimeWarning)

INTERVAL = os.environ.get("INTERVAL", "4h")          # 4h или 1h
IH = {"4h": 4, "1h": 1}[INTERVAL]                       # часов в свече
PP_OUT = "results/pre_pump" if INTERVAL == "4h" else f"results/pre_pump_{INTERVAL}"


def C(hours):
    """Сколько свечей в этом количестве часов."""
    return max(int(hours // IH), 1)


SIG_FROM = os.environ.get("SIG_FROM", "2025-01-01")
SIG_START = pd.Timestamp(SIG_FROM, tz="UTC").timestamp() * 1000
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT_SYMBOLS", "0"))
PUMP = 0.30          # +30%
WIN = C(72)          # 72 ч
DUMP = 0.20          # −20% за 72 ч — для сравнения
BASE_BAND = 0.03     # старт = последняя свеча серии с закрытием в пределах +3% от минимума серии
HELD = 0.15         # «удержался»: через 7 дней закрытие >= +15% от старта
H = 3600 * 1000


# ---------------- данные ----------------
def metrics(sym, first_ms):
    """5-минутные OI, OI в $, соотношения лонг/шорт из архива metrics."""
    d = (pd.Timestamp(first_ms, unit="ms") - pd.Timedelta(days=8)).date()
    today = dt.datetime.now(dt.timezone.utc).date()
    days = []
    while d < today:
        days.append(d)
        d += dt.timedelta(days=1)

    def one(day):
        df = OF.get_zip_csv(f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip")
        if df is None or "create_time" not in df:
            return None
        ts = pd.to_datetime(df.create_time, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ms]")
        out = pd.DataFrame({"t": ts.astype("int64").astype(float)})
        for src, dst in (("sum_open_interest", "oi"), ("sum_open_interest_value", "oi_usd"),
                         ("count_long_short_ratio", "ls_acc"), ("sum_toptrader_long_short_ratio", "ls_top")):
            out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else np.nan
        return out

    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, days) if p is not None]
    if not parts:
        return None
    return pd.concat(parts).sort_values("t").drop_duplicates("t").reset_index(drop=True)


def fwd(arr, n, w, fn):
    """fn по следующим w свечам после каждой свечи (NaN, если окна не хватает)."""
    pad = np.concatenate([arr, np.full(w, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, w)[1:n + 1]
    return fn(win, axis=1)


# ---------------- признаки по монете ----------------
def coin(sym, mcap_now, btc):
    k = Hst.load_klines(sym, INTERVAL)
    if k is None or len(k) < 300:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    qv, tbq, cnt = k.quote_volume.to_numpy(), k.taker_buy_quote_volume.to_numpy(), k["count"].to_numpy()
    t = k.open_time.to_numpy()
    n = len(k)
    mcap = mcap_now * c / c[-1]
    elig = (t >= SIG_START) & (qv >= Hst.MIN_TURNOVER) & (mcap >= Hst.MCAP_MIN) & (mcap <= Hst.MCAP_MAX)
    if not elig.any():
        return None

    S = pd.Series
    qs = S(qv)
    f = pd.DataFrame({"symbol": sym, "open_time": t, "close": c, "mcap": mcap, "turnover": qv})

    # --- объём и дельта ---
    norm = qs.shift(1).rolling(12, min_periods=12).median()
    f["vol_x"] = qv / norm
    dq = 2 * tbq - qv
    f["delta_pct"] = dq / qv * 100
    f["volx_max3d"] = f.vol_x.rolling(C(72), min_periods=1).max()
    f["cvd3d"] = S(dq).rolling(C(72)).sum() / qs.rolling(C(72)).sum() * 100
    f["cvd7d"] = S(dq).rolling(C(168)).sum() / qs.rolling(C(168)).sum() * 100
    f["vol_dry"] = qs.rolling(C(168)).median() / qs.shift(C(168)).rolling(C(552), min_periods=C(240)).median()   # 7д к 30д до них
    ats = qs / S(cnt).replace(0, np.nan)
    f["trade_size_x"] = ats.rolling(C(72)).median() / ats.shift(C(72)).rolling(C(648), min_periods=C(240)).median()

    # --- цена и боковик (Вайкофф) ---
    cs = S(c)
    for name, sh in (("px1d", C(24)), ("px3d", C(72)), ("px7d", C(168)), ("px30d", C(720))):
        f[name] = (cs / cs.shift(sh) - 1) * 100
    hi7, lo7 = S(h).rolling(C(168)).max(), S(l).rolling(C(168)).min()
    hi14, lo14 = S(h).rolling(C(336)).max(), S(l).rolling(C(336)).min()
    f["range7d"] = (hi7 / lo7 - 1) * 100
    f["range14d"] = (hi14 / lo14 - 1) * 100
    f["range7d_rank"] = f.range7d.rolling(C(2160), min_periods=C(720)).rank(pct=True) * 100   # 0 = самый узкий за 90 дней
    f["pos_in_range7d"] = (c - lo7) / (hi7 - lo7) * 100
    # спринг: прокол минимума прошлых 14 дней, закрытие обратно выше, объём >= x2
    prior_lo = S(l).shift(1).rolling(C(336)).min()
    spring = (l < prior_lo) & (c > prior_lo) & (f.vol_x >= 2)
    f["spring3d"] = spring.astype(int).rolling(C(72), min_periods=1).max().fillna(0).astype(bool)
    # поглощение: объём >= x3, дельта <= −17%, а закрытие в верхней половине свечи
    absorb = (f.vol_x >= 3) & (f.delta_pct <= -17) & (c >= (h + l) / 2)
    f["absorb7d"] = absorb.astype(int).rolling(C(168), min_periods=1).sum()

    # --- контекст BTC ---
    f["btc7d"] = pd.merge(f[["open_time"]], btc, on="open_time", how="left").btc7d.to_numpy()

    # --- OI, лонг/шорт, фандинг на момент закрытия ---
    T = t + IH * H
    m = metrics(sym, t[elig].min())
    cols = ["oi_chg1d", "oi_chg3d", "oi_chg7d", "oi_mcap", "ls_acc", "ls_acc_chg3d", "ls_top"]
    if m is not None and len(m):
        mt = m.t.to_numpy()
        fresh = (T - OF.asof(mt, mt, T)) <= 30 * 60 * 1000
        oi_now = np.where(fresh, OF.asof(mt, m.oi.to_numpy(), T), np.nan)
        for name, back in (("oi_chg1d", 24), ("oi_chg3d", 72), ("oi_chg7d", 168)):
            f[name] = (oi_now / OF.asof(mt, m.oi.to_numpy(), T - back * H) - 1) * 100
        f["oi_mcap"] = np.where(fresh, OF.asof(mt, m.oi_usd.to_numpy(), T), np.nan) / mcap * 100
        ls = np.where(fresh, OF.asof(mt, m.ls_acc.to_numpy(), T), np.nan)
        f["ls_acc"] = ls
        f["ls_acc_chg3d"] = ls - OF.asof(mt, m.ls_acc.to_numpy(), T - 72 * H)
        f["ls_top"] = np.where(fresh, OF.asof(mt, m.ls_top.to_numpy(), T), np.nan)
    else:
        for col in cols:
            f[col] = np.nan
    fu = OF.funding(sym)
    if fu is not None and len(fu):
        ft, fr = fu.t.to_numpy(), fu.rate8.to_numpy()
        csum = np.concatenate([[0], np.cumsum(fr)])
        for name, back in (("fund1d", 24), ("fund3d", 72)):
            i1, i0 = np.searchsorted(ft, T, side="right"), np.searchsorted(ft, T - back * H, side="right")
            cntf = i1 - i0
            f[name] = np.where(cntf > 0, (csum[i1] - csum[i0]) / np.maximum(cntf, 1), np.nan) * 100
    else:
        f["fund1d"] = f["fund3d"] = np.nan

    # --- что было после (метки) ---
    up = fwd(h, n, WIN, np.max) / c - 1
    dn = fwd(l, n, WIN, np.min) / c - 1
    complete = ~np.isnan(fwd(c, n, WIN, np.min))
    hit = complete & (up >= PUMP)
    # серия подряд идущих «hit»-свечей = один памп. Старт — последняя свеча базы перед взлётом:
    # самая поздняя свеча серии, чьё закрытие не выше минимального закрытия серии + BASE_BAND.
    start = np.zeros(n, bool)
    run_beg = np.flatnonzero(hit & ~np.concatenate([[False], hit[:-1]]))
    run_end = np.flatnonzero(hit & ~np.concatenate([hit[1:], [False]]))
    for b0, e0 in zip(run_beg, run_end):
        cc = c[b0:e0 + 1]
        start[b0 + np.flatnonzero(cc <= cc.min() * (1 + BASE_BAND))[-1]] = True
    after = np.zeros(n, bool)                         # сам памп и 72 ч после старта — не контроль
    for i in np.flatnonzero(start):
        after[i + 1:i + 1 + WIN] = True
    f["pump"] = start
    f["dump"] = complete & (dn <= -DUMP)
    f["group"] = np.where(start, "pump", np.where(hit | after, "skip", "control"))
    f["complete"] = complete

    # --- подробности пампа ---
    det = {x: np.full(n, np.nan) for x in ("peak_pct", "hours_to_peak", "candles_to_30", "dd_before_peak",
                                            "retrace_after_peak", "close_7d_pct")}
    for i in np.flatnonzero(start):
        hw, lw = h[i + 1:i + 1 + WIN], l[i + 1:i + 1 + WIN]
        p = int(np.argmax(hw))
        det["peak_pct"][i] = (hw[p] / c[i] - 1) * 100
        det["hours_to_peak"][i] = IH * (p + 1)
        det["candles_to_30"][i] = int(np.argmax(hw >= c[i] * (1 + PUMP))) + 1
        det["dd_before_peak"][i] = (lw[:p + 1].min() / c[i] - 1) * 100
        after_pk = l[i + 1 + p + 1:i + 1 + p + 1 + WIN]
        det["retrace_after_peak"][i] = (after_pk.min() / hw[p] - 1) * 100 if len(after_pk) else np.nan
        det["close_7d_pct"][i] = (c[i + C(168)] / c[i] - 1) * 100 if i + C(168) < n else np.nan
    for x, v in det.items():
        f[x] = v

    f = f[elig & complete & (f.group != "skip")]
    return f


# ---------------- запуск ----------------
def prep():
    os.makedirs(PP_OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps[Hst.base_asset(s)] for s in syms if Hst.base_asset(s) in caps}
    with open(f"{PP_OUT}/universe.json", "w") as fh:
        json.dump(uni, fh)
    print("монет:", len(uni))


def run_shard():
    os.makedirs(f"{PP_OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{PP_OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]
    if LIMIT:
        mine = mine[:LIMIT]
    b = Hst.load_klines("BTCUSDT", INTERVAL)
    btc = pd.DataFrame({"open_time": b.open_time, "btc7d": (b.close / b.close.shift(C(168)) - 1) * 100})
    print(f"часть {SHARD}: монет {len(mine)}", flush=True)

    def work(sym):
        try:
            return coin(sym, uni[sym], btc)
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(3) as ex:
        for i, p in enumerate(ex.map(work, mine), 1):
            if p is not None and len(p):
                parts.append(p)
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    allp = pd.concat(parts)
    print("пампов:", int(allp.pump.sum()), "| OI заполнен:", f"{allp.oi_chg1d.notna().mean() * 100:.0f}%",
          "| фандинг:", f"{allp.fund1d.notna().mean() * 100:.0f}%")
    allp.round(4).to_csv(f"{PP_OUT}/parts/part{SHARD}.csv.gz", index=False)


# ---------------- сводка ----------------
FEATURES = [
    ("range7d", "ширина диапазона 7д, %"),
    ("range7d_rank", "узость диапазона 7д среди своих 90 дней (0 = самый узкий)"),
    ("range14d", "ширина диапазона 14д, %"),
    ("pos_in_range7d", "где цена в диапазоне 7д (0 = низ, 100 = верх)"),
    ("vol_dry", "объём 7д к 30д до них («высыхание»)"),
    ("vol_x", "объём свечи старта к норме (12 свечей)"),
    ("volx_max3d", "макс. аномалия объёма за 3д"),
    ("delta_pct", "дельта свечи старта, %"),
    ("cvd3d", "накопленная дельта 3д, % объёма"),
    ("cvd7d", "накопленная дельта 7д, % объёма"),
    ("trade_size_x", "средний размер сделки 3д к 30д"),
    ("px1d", "цена за 1д, %"), ("px3d", "цена за 3д, %"), ("px7d", "цена за 7д, %"), ("px30d", "цена за 30д, %"),
    ("oi_chg1d", "OI за 1д, %"), ("oi_chg3d", "OI за 3д, %"), ("oi_chg7d", "OI за 7д, %"),
    ("oi_mcap", "OI в $ к капе, %"),
    ("fund1d", "фандинг средний 1д, % за 8ч"), ("fund3d", "фандинг средний 3д, % за 8ч"),
    ("ls_acc", "лонг/шорт по аккаунтам"), ("ls_acc_chg3d", "изменение лонг/шорт за 3д"),
    ("ls_top", "лонг/шорт топ-трейдеров (по позициям)"),
    ("mcap", "капа, $"), ("btc7d", "BTC за 7д, %"),
]
FLAGS = [("spring3d", "был спринг за 3д"), ("absorb7d", "свечей поглощения за 7д > 0")]


def pct(x):
    return f"{x * 100:.1f}%" if pd.notna(x) else "-"


def bucket_table(a, col, base_p, base_d):
    x = a[col]
    ok = x.notna()
    if ok.sum() < 1000:
        return None
    try:
        b = pd.qcut(x[ok], 10, duplicates="drop")
    except ValueError:
        return None
    g = a[ok].groupby(b, observed=True)
    rows = []
    for iv, s in g:
        pr, dr = s.pump.mean(), s.dump.mean()
        rows.append({"диапазон": f"{iv.left:.3g} … {iv.right:.3g}", "свечей": len(s), "пампов": int(s.pump.sum()),
                     "доля пампов": f"{pr * 100:.2f}%", "подъём": f"{pr / base_p:.2f}×",
                     "доля дампов −20%": f"{dr * 100:.2f}%", "подъём дампов": f"{dr / base_d:.2f}×",
                     "перекос вверх": f"{(pr / base_p) / max(dr / base_d, 1e-9):.2f}"})
    return pd.DataFrame(rows)


def combine():
    files = sorted(f for f in os.listdir(f"{PP_OUT}/parts") if f.endswith(".csv.gz"))
    a = pd.concat([pd.read_csv(f"{PP_OUT}/parts/{f}") for f in files], ignore_index=True)
    a = a.replace([np.inf, -np.inf], np.nan)          # деление на 0 (нулевой объём / OI) — как «нет данных»
    a["pump"] = a.pump.astype(bool)
    a["dump"] = a.dump.astype(bool)
    a["absorb7d_flag"] = a.absorb7d > 0
    months = (a.open_time.max() - a.open_time.min()) / 86_400_000 / 30.4
    base_p, base_d = a.pump.mean(), a.dump.mean()
    P = a[a.pump].copy()
    P["тип"] = np.where(P.close_7d_pct >= HELD * 100, "удержался", "пила")
    h30 = P.candles_to_30 * IH                         # часов от старта до +30%
    P["форма"] = np.select([h30 <= 4, h30 <= 12], ["до 4 ч", "4–12 ч"], "дольше 12 ч")

    out = open(f"{PP_OUT}/pre_pump_summary.md", "w")
    w = out.write
    w(f"# Что было до пампа ({INTERVAL}-свечи)\n\n")
    w(f"Памп: от закрытия {INTERVAL}-свечи high за 72 ч ≥ +{PUMP * 100:.0f}%, старт — последняя свеча базы перед взлётом "
      f"(закрытие в пределах +{BASE_BAND * 100:.0f}% от минимума серии). "
      f"Капа ${Hst.MCAP_MIN / 1e6:.0f}M–{Hst.MCAP_MAX / 1e6:.0f}M, оборот свечи ≥ ${Hst.MIN_TURNOVER / 1e3:.0f}K, "
      f"с {SIG_FROM}. Признаки — только то, что известно на закрытии свечи старта.\n\n")
    w(f"Всего свечей: {len(a):,} | монет: {a.symbol.nunique()} | пампов: {len(P)} ({len(P) / months:.1f} в месяц)\n\n")
    w(f"**Базовая доля:** памп после {base_p * 100:.2f}% свечей; дамп −20% за 72 ч после {base_d * 100:.2f}%.\n"
      "«Подъём» = доля в группе / базовая доля. 2× значит «в два раза чаще среднего».\n"
      "«Перекос вверх» = подъём пампов / подъём дампов: > 1 — признак тянет именно вверх, ≈ 1 — просто к сильному движению.\n\n")

    w("## Пампы: форма и что было потом\n\n")
    t1 = P.groupby("форма").agg(пампов=("pump", "size"),
                                медиана_пика=("peak_pct", "median"),
                                часов_до_пика=("hours_to_peak", "median"),
                                просадка_до_пика=("dd_before_peak", "median"),
                                откат_после_пика=("retrace_after_peak", "median"),
                                удержались=("тип", lambda s: f"{(s == 'удержался').mean() * 100:.0f}%")).reindex(["до 4 ч", "4–12 ч", "дольше 12 ч"]).dropna(how="all")
    t1["доля"] = (t1.пампов / len(P) * 100).round(0).astype(int).astype(str) + "%"
    w(t1.round(1).to_markdown() + "\n\n")
    w("Просадка до пика — медиана худшей цены до вершины от закрытия старта (выбило бы стоп?). "
      "Откат после пика — от вершины, в следующие 72 ч. «Удержался» — через 7 дней закрытие ≥ +15% от старта.\n\n")
    dd = P.dd_before_peak
    w("Как часто до пика цена сначала ходила вниз: " + ", ".join(
        f"ниже −{x}%: {(dd <= -x).mean() * 100:.0f}%" for x in (2, 4.3, 7, 10)) + "\n\n")

    w("## Портрет: медианы признаков\n\n")
    rows = []
    for col, name in FEATURES:
        rows.append({"признак": name, "контроль": a.loc[~a.pump, col].median(), "все пампы": P[col].median(),
                     "удержался": P.loc[P.тип == "удержался", col].median(), "пила": P.loc[P.тип == "пила", col].median(),
                     "до 4 ч": P.loc[P.форма == "до 4 ч", col].median()})
    for col, name in FLAGS:
        cc = col if col == "spring3d" else "absorb7d_flag"
        rows.append({"признак": name, "контроль": a.loc[~a.pump, cc].mean() * 100, "все пампы": P[cc].mean() * 100,
                     "удержался": P.loc[P.тип == "удержался", cc].mean() * 100,
                     "пила": P.loc[P.тип == "пила", cc].mean() * 100,
                     "до 4 ч": P.loc[P.форма == "до 4 ч", cc].mean() * 100})
    w(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g") + "\n\n")
    w("(для двух последних строк — % свечей, где признак был)\n\n")

    w("## Каждый признак по десяткам (децилям)\n\n")
    w("Ищем строки, где подъём пампов высокий, а подъём дампов — нет. "
      "Если оба растут вместе — признак говорит «будет сильное движение», а не направление.\n\n")
    for col, name in FEATURES:
        tb = bucket_table(a, col, base_p, base_d)
        if tb is not None:
            w(f"### {name} (`{col}`)\n\n" + tb.to_markdown(index=False) + "\n\n")
    for col, name in FLAGS:
        cc = col if col == "spring3d" else "absorb7d_flag"
        rows = []
        for val, s in a.groupby(a[cc]):
            rows.append({"": "да" if val else "нет", "свечей": len(s), "пампов": int(s.pump.sum()),
                         "доля пампов": f"{s.pump.mean() * 100:.2f}%", "подъём": f"{s.pump.mean() / base_p:.2f}×",
                         "подъём дампов": f"{s.dump.mean() / base_d:.2f}×"})
        w(f"### {name}\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")

    w("## Комбинации «сжатой пружины»\n\n")
    combos = {
        "узкий диапазон (ранг ≤ 20)": a.range7d_rank <= 20,
        "узкий + объём высох (≤ 0.7)": (a.range7d_rank <= 20) & (a.vol_dry <= 0.7),
        "узкий + OI за 3д ≥ +10%": (a.range7d_rank <= 20) & (a.oi_chg3d >= 10),
        "узкий + OI 3д ≥ +10% + фандинг 3д < 0": (a.range7d_rank <= 20) & (a.oi_chg3d >= 10) & (a.fund3d < 0),
        "OI 3д ≥ +20%, цена 3д в ±5%": (a.oi_chg3d >= 20) & (a.px3d.abs() <= 5),
        "OI 3д ≥ +20%, цена 3д в ±5%, фандинг 3д < 0": (a.oi_chg3d >= 20) & (a.px3d.abs() <= 5) & (a.fund3d < 0),
        "накопленная дельта 7д ≥ +5%, цена 7д в ±5%": (a.cvd7d >= 5) & (a.px7d.abs() <= 5),
        "спринг за 3д + узкий диапазон 14д (≤ 30%)": a.spring3d & (a.range14d <= 30),
        "поглощение за 7д + цена 7д в ±5%": a.absorb7d_flag & (a.px7d.abs() <= 5),
        "лонг/шорт аккаунтов ≤ 1 (толпа в шортах) + OI 3д ≥ +10%": (a.ls_acc <= 1) & (a.oi_chg3d >= 10),
        # из первого прогона: признаки, где пампы растут сильнее дампов
        "аномалия объёма за 3д ≥ x10 + накопл. дельта 7д ≥ 0": (a.volx_max3d >= 10) & (a.cvd7d >= 0),
        "аномалия объёма за 3д ≥ x10 + толпа в шортах (≤ 1)": (a.volx_max3d >= 10) & (a.ls_acc <= 1),
        "толпа в шортах (≤ 0.82) + фандинг 1д ≤ −0.02%": (a.ls_acc <= 0.82) & (a.fund1d <= -0.02),
        "толпа в шортах (≤ 1) + накопл. дельта 7д ≥ 0": (a.ls_acc <= 1) & (a.cvd7d >= 0),
        "толпа в шортах (≤ 1) + OI 3д ≥ +10% + накопл. дельта 7д ≥ 0":
            (a.ls_acc <= 1) & (a.oi_chg3d >= 10) & (a.cvd7d >= 0),
        "размер сделки ≥ 1.22 + накопл. дельта 7д ≥ 0": (a.trade_size_x >= 1.22) & (a.cvd7d >= 0),
    }
    rows = []
    for name, m in combos.items():
        s = a[m.fillna(False)]
        if len(s) == 0:
            continue
        rows.append({"условие": name, "свечей": len(s), "монет": s.symbol.nunique(),
                     "пампов": int(s.pump.sum()), "доля пампов": f"{s.pump.mean() * 100:.2f}%",
                     "подъём": f"{s.pump.mean() / base_p:.2f}×", "подъём дампов": f"{s.dump.mean() / base_d:.2f}×",
                     "перекос вверх": f"{(s.pump.mean() / base_p) / max(s.dump.mean() / base_d, 1e-9):.2f}",
                     "доля пампов, которых поймали": f"{s.pump.sum() / len(P) * 100:.0f}%"})
    w(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    w(f"Свечей — сколько {INTERVAL}-свечей подходило под условие (одна монета может подходить много свечей подряд).\n")
    out.close()

    ev = P.copy()
    ev.insert(1, "start_close_kyiv", pd.to_datetime(ev.open_time + IH * H, unit="ms", utc=True)
              .dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M"))
    ev["тип"], ev["форма"] = P["тип"], P["форма"]
    ev.drop(columns=["group", "complete", "pump"]).round(3).to_csv(f"{PP_OUT}/pre_pump_events.csv", index=False)
    print(open(f"{PP_OUT}/pre_pump_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
