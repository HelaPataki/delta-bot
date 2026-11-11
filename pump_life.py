"""
Жизнь пампа до самой смерти. Только исследование, без сделок «вживую». Живого бота не трогает.

Данные: архив Binance (data.binance.vision): 1h-свечи фьючерсов и спота, фандинг, metrics (OI каждые 5 мин,
соотношения лонгов/шортов), капа — оценка по CoinGecko (текущая капа × цена тогда / цена сейчас).

ПАМП (эпизод): от минимума 1h-свечи цена в следующие 72 ч выросла на 30%+ (старт — самый низкий low).
  Эпизод живёт, пока не случилось одно из двух:
    • цена откатила 70% всего роста (от старта до максимума эпизода) — «смерть»;
    • 30 дней без нового максимума.
ВОЛНЫ внутри эпизода (зигзаг 20%): вершина волны подтверждается, когда цена ушла на 20% ниже её максимума;
  новая волна начинается, когда цена поднялась на 20% от минимума отката.
МОМЕНТ НАБЛЮДЕНИЯ: закрытие 1h-свечи, на которой откат от вершины впервые достиг 20% (а также 30/40/50/60%).
  Все признаки считаются только по данным до этого момента (без заглядывания в будущее).
ГЛАВНЫЙ ВОПРОС: перебьёт ли цена эту вершину в следующие 30 дней (= стоп шортиста за хаем выбит).
"""

import datetime as dt
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-12-01")

import numpy as np
import pandas as pd

import history as Hst

OUT = os.environ.get("OUT_DIR", "results/pump_life")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
FROM = pd.Timestamp(os.environ.get("FROM", "2025-01-01"), tz="UTC").timestamp() * 1000
H = 3600 * 1000
W = 72                 # окно пампа, часов
GAIN = 0.30            # памп от 30%
DEATH = 0.70           # смерть = откат 70% роста
QUIET = 720            # 30 дней без нового хая
ZZ = 0.20              # зигзаг волн
DEPTHS = [20, 30, 40, 50, 60]
AFTER = 720            # смотрим 30 дней после момента наблюдения
HOR = 240              # сделка: 10 дней
RR = 1.8
STOP_BUF = 0.005
FEATURED = os.environ.get("FEATURED", "RAVEUSDT,SIRENUSDT,MYXUSDT,COAIUSDT,AIAUSDT").split(",")


def kyiv(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(Hst.KYIV).strftime("%Y-%m-%d %H:%M")


# ---------------- данные ----------------
def zip_csv(url):
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, header=None)


def klines_any(market, sym, d0, d1):
    """1h-свечи spot/um за даты d0..d1 (месячные файлы + дневные за текущий месяц)."""
    today = dt.datetime.now(dt.timezone.utc).date()
    this_month = dt.date(today.year, today.month, 1)
    base = "data/spot" if market == "spot" else "data/futures/um"
    urls = []
    for m in Hst.months(d0, min(d1, today)):
        if m < this_month:
            urls.append(f"{Hst.ARCHIVE}/{base}/monthly/klines/{sym}/1h/{sym}-1h-{m:%Y-%m}.zip")
        else:
            d = max(m, d0)
            while d < today and d <= d1:
                urls.append(f"{Hst.ARCHIVE}/{base}/daily/klines/{sym}/1h/{sym}-1h-{d:%Y-%m-%d}.zip")
                d += dt.timedelta(days=1)
    frames = [f for f in map(zip_csv, urls) if f is not None]
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]                 # убрать заголовки
    df = df.iloc[:, :12].astype(float)
    df.columns = Hst.COLS
    ot = df.open_time.to_numpy()
    df["open_time"] = np.where(ot > 1e14, ot // 1000, ot)        # спот с 2025 — микросекунды
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def funding(sym, d0):
    frames = []
    for m in Hst.months(d0, dt.datetime.now(dt.timezone.utc).date()):
        f = zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if f is not None:
            f = f[pd.to_numeric(f.iloc[:, 0], errors="coerce").notna()].iloc[:, :3].astype(float)
            f.columns = ["t", "hours", "rate"]
            frames.append(f)
    if not frames:
        return None
    f = pd.concat(frames).drop_duplicates("t").sort_values("t")
    f["rate8"] = f.rate * 8 / f.hours.where(f.hours > 0, 8) * 100       # % за 8 часов
    return f[["t", "rate8"]].reset_index(drop=True)


MCOLS = ["oi", "oi_usd", "top_acc", "top_pos", "glob", "taker"]


def metrics(sym, days):
    def one(day):
        url = f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip"
        r = Hst.get(url)
        if r is None or r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        ts = pd.to_datetime(df.create_time, utc=True).astype("int64") // 10 ** 6
        cols = ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
        out = pd.DataFrame({"t": ts.astype(float)})
        for k, col in zip(MCOLS, cols):
            out[k] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
        return out

    with ThreadPoolExecutor(6) as ex:
        parts = [p for p in ex.map(one, sorted(days)) if p is not None]
    if not parts:
        return None
    return pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)


class AsOf:
    """Последнее значение ряда с временем <= запроса."""

    def __init__(self, df, col):
        self.ok = df is not None and len(df) > 0
        if self.ok:
            d = df[["t", col]].dropna()
            self.t, self.v = d.t.to_numpy(), d[col].to_numpy()
            self.ok = len(d) > 0

    def at(self, q, max_lag=6 * H):
        if not self.ok:
            return np.nan
        i = np.searchsorted(self.t, q, side="right") - 1
        if i < 0 or q - self.t[i] > max_lag:
            return np.nan
        return self.v[i]

    def span(self, q0, q1, fn):
        if not self.ok:
            return np.nan
        a, b = np.searchsorted(self.t, q0, side="left"), np.searchsorted(self.t, q1, side="right")
        return fn(self.v[a:b]) if b > a else np.nan

    def argmax_time(self, q0, q1):
        if not self.ok:
            return np.nan
        a, b = np.searchsorted(self.t, q0, side="left"), np.searchsorted(self.t, q1, side="right")
        return self.t[a + int(np.nanargmax(self.v[a:b]))] if b > a and np.isfinite(self.v[a:b]).any() else np.nan


def ratio(a, b):
    return a / b - 1 if (b and np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan


# ---------------- эпизоды и волны (только свечи) ----------------
def find_episodes(k):
    h, l = k.high.to_numpy(), k.low.to_numpy()
    t = k.open_time.to_numpy()
    n = len(k)
    pad = np.concatenate([h, np.full(W, np.nan)])
    fmax = np.lib.stride_tricks.sliding_window_view(pad, W + 1)[:n].max(axis=1)
    hit = (fmax / l - 1 >= GAIN) & (t >= FROM)
    eps, i = [], 0
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hit[j + 1] and h[i:j + 2].max() / l[i:j + 2].min() < 10:
            j += 1
        s = i + int(np.argmin(l[i:j + 1]))
        L0, top, ti, e, why = l[s], h[s], s, None, "ещё живой"
        for q in range(s, n):
            if h[q] > top:
                top, ti = h[q], q
            if q > ti and top / L0 - 1 >= GAIN and l[q] <= L0 + (1 - DEATH) * (top - L0):
                e, why = q, "смерть (откат 70%)"
                break
            if q - ti >= QUIET:
                e, why = q, "30 дней без хая"
                break
        if e is None:
            e = n - 1
        eps.append({"s": s, "ti": ti, "e": e, "end": why})
        i = e + 1
    return eps


def find_waves(k, s, e):
    """Зигзаг 20%: список (начало волны ws, вершина hi, момент подтверждения M)."""
    h, l = k.high.to_numpy(), k.low.to_numpy()
    out, state, ws, hw, hi, lw, li = [], "up", s, h[s], s, None, None
    for q in range(s, e + 1):
        if state == "up":
            if h[q] > hw:
                hw, hi = h[q], q
            if l[q] <= hw * (1 - ZZ):
                out.append((ws, hi, q))
                state, lw, li = "down", l[q], q
        else:
            if l[q] < lw:
                lw, li = l[q], q
            if h[q] >= lw * (1 + ZZ):
                state, ws, hw, hi = "up", li, h[q], q
    return out


# ---------------- одна монета ----------------
def coin(sym, cap):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < W + 200:
        return [], [], []
    eps = find_episodes(k)
    if not eps:
        return [], [], []
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    t, qv, tbq = k.open_time.to_numpy(), k.quote_volume.to_numpy(), k.taker_buy_quote_volume.to_numpy()
    n = len(k)
    c_now = c[-1]
    day = lambda ms: pd.Timestamp(ms, unit="ms", tz="UTC").date()

    # какие дни нужны для metrics/спота
    need = set()
    for ep in eps:
        a = t[ep["s"]] - 3 * 24 * H
        b = min(t[min(ep["e"], n - 1)], t[ep["s"]] + 75 * 24 * H) + 24 * H
        d = day(a)
        while d <= day(b):
            need.add(d)
            d += dt.timedelta(days=1)
    met = metrics(sym, need)
    fund = funding(sym, min(need).replace(day=1))
    spot_sym = Hst.base_asset(sym) + "USDT"                     # 1000PEPEUSDT -> спот PEPEUSDT
    spot = klines_any("spot", spot_sym, min(need), max(need))
    sp_qv = pd.Series(spot.quote_volume.to_numpy(), index=spot.open_time.to_numpy()) if spot is not None else None
    sp_c = (pd.Series(spot.close.to_numpy(), index=spot.open_time.to_numpy())
            if spot is not None and spot_sym == sym else None)        # премию считаем только без множителя
    M = {col: AsOf(met, col) for col in MCOLS}
    F = AsOf(fund, "rate8")

    def oi(q):
        return M["oi"].at(q)

    def spot_vol(i0, i1):
        if sp_qv is None:
            return 0.0
        return float(sp_qv.reindex(t[i0:i1 + 1]).fillna(0).sum())

    pre_norm = lambda s: np.nanmedian(qv[max(0, s - 168):s]) if s > 24 else np.nan
    waves_rows, ep_rows, daily = [], [], []
    for en, ep in enumerate(eps):
        s, ti, e = ep["s"], ep["ti"], ep["e"]
        L0, TOP = l[s], h[ti]
        pre = pre_norm(s)
        cap_top = cap * TOP / c_now if cap else np.nan
        # скорость смерти (от финального хая эпизода)
        after = l[ti + 1:min(n, ti + 1 + 60 * 24)]
        dd = after / TOP - 1
        first = lambda x: (int(np.argmax(x)) + 1) if x.any() else np.nan
        waves = find_waves(k, s, e)
        ep_id = f"{sym}|{kyiv(t[s])}"
        ep_rows.append({
            "эпизод": ep_id, "монета": sym, "старт": kyiv(t[s]), "хай": kyiv(t[ti]), "конец": ep["end"],
            "рост до хая, %": (TOP / L0 - 1) * 100, "часов до хая": ti - s, "волн": len(waves),
            "оборот старт→хай / норма": qv[s:ti + 1].mean() / pre if pre > 0 else np.nan,
            "оборот старт→хай, $M": qv[s:ti + 1].sum() / 1e6,
            "капа на хае, $M": cap_top / 1e6, "есть спот": sp_qv is not None and spot_vol(s, ti) > 0,
            "ч до −30%": first(dd <= -0.3), "ч до −50%": first(dd <= -0.5), "ч до −70%": first(dd <= -0.7),
            "макс падение 30д, %": dd[:720].min() * 100 if len(dd) else np.nan,
            "отслежено часов после хая": len(after)})

        for wn, (ws, hi, m) in enumerate(waves, 1):
            top = h[hi]
            t_ws, t_top, t_m = t[ws], t[hi] + H, t[m] + H
            ep_high_sofar = h[s:m + 1].max()
            fut_h = h[m + 1:min(n, m + 1 + AFTER)]
            broke = np.nonzero(fut_h > top)[0]
            broke_i = m + 1 + broke[0] if len(broke) else None
            seen = n - 1 - m
            # глубина отката до перебоя хая
            seg_end = broke_i if broke_i is not None else min(n, m + 1 + AFTER)
            lows = l[hi + 1:seg_end]
            vol_wave = qv[ws:hi + 1].mean()
            prev_wave = waves[wn - 2] if wn > 1 else None
            f_v = qv[ws:hi + 1].sum()
            s_v = spot_vol(ws, hi)
            row = {
                "эпизод": ep_id, "монета": sym, "волна": wn, "вершина": kyiv(t[hi]), "момент −20%": kyiv(t[m]),
                "рост волны, %": (top / l[ws] - 1) * 100, "рост от старта, %": (top / L0 - 1) * 100,
                "часов волны": hi - ws, "часов от старта": hi - s, "часов вершина→−20%": m - hi,
                "вершина = хай эпизода": top >= ep_high_sofar,
                "оборот волны / норма": vol_wave / pre if pre > 0 else np.nan,
                "оборот волны / прошлая волна": (vol_wave / qv[prev_wave[0]:prev_wave[1] + 1].mean()) if prev_wave else np.nan,
                "свеча-вершина / средняя волны": qv[hi] / vol_wave if vol_wave > 0 else np.nan,
                "оборот отката / оборот волны": qv[hi + 1:m + 1].mean() / vol_wave if vol_wave > 0 else np.nan,
                "покупки волны, %": tbq[ws:hi + 1].sum() / f_v * 100 if f_v > 0 else np.nan,
                "покупки отката, %": tbq[hi + 1:m + 1].sum() / qv[hi + 1:m + 1].sum() * 100 if qv[hi + 1:m + 1].sum() > 0 else np.nan,
                "OI за волну, %": ratio(oi(t_top), oi(t_ws)) * 100,
                "OI за откат, %": ratio(oi(t_m), oi(t_top)) * 100,
                "OI от старта пампа, %": ratio(oi(t_m), oi(t[s])) * 100,
                "OI пик раньше вершины, ч": (t_top - M["oi"].argmax_time(t_ws, t_top)) / H,
                "OI$ / капа, %": M["oi_usd"].at(t_m) / (cap * c[m] / c_now) * 100 if cap else np.nan,
                "топ-трейдеры L/S на вершине": M["top_pos"].at(t_top),
                "топ-трейдеры L/S в момент": M["top_pos"].at(t_m),
                "все аккаунты L/S на вершине": M["glob"].at(t_top),
                "все аккаунты L/S в момент": M["glob"].at(t_m),
                "фандинг в момент, %": F.at(t_m, 9 * H),
                "фандинг мин с начала пампа, %": F.span(t[s], t_m, np.nanmin),
                "фандинг средний в волну, %": F.span(t_ws, t_top, np.nanmean),
                "есть спот": s_v > 0,
                "доля спота в волне, %": s_v / (s_v + f_v) * 100 if (s_v + f_v) > 0 else np.nan,
                "премия фьюча к споту на вершине, %": (ratio(c[hi], sp_c.get(t[hi], np.nan)) * 100) if sp_c is not None else np.nan,
                "капа на вершине, $M": cap * top / c_now / 1e6 if cap else np.nan,
                # исход
                "отслежено часов": seen,
                "хай перебит за 30д": broke_i is not None,
                "часов до перебоя": (broke_i - m) if broke_i is not None else np.nan,
                "насколько выше хая ушла, %": (fut_h.max() / top - 1) * 100 if broke_i is not None else np.nan,
                "макс откат до перебоя, %": (lows.min() / top - 1) * 100 if len(lows) else np.nan,
            }
            for d in DEPTHS:
                hitd = np.nonzero(lows <= top * (1 - d / 100))[0]
                row[f"дошла до −{d}%"] = len(hitd) > 0
            for hrs in (24, 72, 168):
                row[f"без хая {hrs}ч"] = (broke_i is None or broke_i - m > hrs) and seen >= hrs
            # сделка: шорт по закрытию свечи момента, стоп за вершиной +0.5%, тейк 1.8R, 10 дней
            e_px = c[m]
            stop = top * (1 + STOP_BUF)
            risk = stop / e_px - 1
            take = e_px * (1 - RR * risk)
            hh, ll = h[m + 1:m + 1 + HOR], l[m + 1:m + 1 + HOR]
            si = np.nonzero(hh >= stop)[0]
            tk = np.nonzero(ll <= take)[0] if take > 0 else np.array([], int)
            si = si[0] if len(si) else 10 ** 6
            tk = tk[0] if len(tk) else 10 ** 6
            if len(hh) < HOR and si == tk == 10 ** 6:
                R = np.nan
            elif si == tk == 10 ** 6:
                R = (e_px - c[m + HOR]) / (stop - e_px)
            else:
                R = -1.0 if si <= tk else RR
            row.update({"стоп шорта, %": risk * 100, "шорт R": R})
            waves_rows.append(row)

        if sym in FEATURED:
            d0 = s - (s % 24)
            for q in range(max(0, d0 - 72), min(n, max(e, ti + 24 * 20) + 1), 24):
                q1 = min(n - 1, q + 23)
                tq = t[q1] + H
                fv, sv = qv[q:q1 + 1].sum(), spot_vol(q, q1)
                daily.append({"эпизод": ep_id, "день": kyiv(t[q])[:10], "open": o[q], "high": h[q:q1 + 1].max(),
                              "low": l[q:q1 + 1].min(), "close": c[q1], "оборот фьюч $M": fv / 1e6,
                              "доля спота %": sv / (sv + fv) * 100 if sv + fv > 0 else np.nan,
                              "покупки %": tbq[q:q1 + 1].sum() / fv * 100 if fv else np.nan,
                              "OI (монет)": oi(tq), "OI $M": M["oi_usd"].at(tq) / 1e6,
                              "фандинг %": F.at(tq, 9 * H), "топ L/S": M["top_pos"].at(tq),
                              "все L/S": M["glob"].at(tq)})
    return waves_rows, ep_rows, daily


# ---------------- запуск ----------------
def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    try:
        caps = Hst.coingecko_mcaps()
    except Exception as e:
        print("CoinGecko:", e)
        caps = {}
    uni = {s: caps.get(Hst.base_asset(s)) for s in syms}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni), "с капой:", sum(v is not None for v in uni.values()))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]

    def work(sym):
        try:
            return coin(sym, uni[sym])
        except Exception as e:
            print(f"{sym}: {e!r}", file=sys.stderr)
            return [], [], []

    W_, E_, D_ = [], [], []
    with ThreadPoolExecutor(3) as ex:
        for w, e, d in ex.map(work, mine):
            W_ += w
            E_ += e
            D_ += d
    pd.DataFrame(W_).to_csv(f"{OUT}/parts/waves{SHARD}.csv.gz", index=False)
    pd.DataFrame(E_).to_csv(f"{OUT}/parts/eps{SHARD}.csv.gz", index=False)
    pd.DataFrame(D_).to_csv(f"{OUT}/parts/daily{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, эпизодов {len(E_)}, вершин волн {len(W_)}")


# ---------------- сводка ----------------
FEATS = ["волна", "рост волны, %", "рост от старта, %", "часов волны", "часов от старта", "часов вершина→−20%",
         "оборот волны / норма", "оборот волны / прошлая волна", "свеча-вершина / средняя волны",
         "оборот отката / оборот волны", "покупки волны, %", "покупки отката, %",
         "OI за волну, %", "OI за откат, %", "OI от старта пампа, %", "OI пик раньше вершины, ч", "OI$ / капа, %",
         "топ-трейдеры L/S на вершине", "топ-трейдеры L/S в момент", "все аккаунты L/S на вершине",
         "все аккаунты L/S в момент", "фандинг в момент, %", "фандинг мин с начала пампа, %",
         "фандинг средний в волну, %", "доля спота в волне, %", "премия фьюча к споту на вершине, %",
         "капа на вершине, $M"]


def load(name):
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        if fn.startswith(name):
            try:
                parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pct(x):
    return f"{x * 100:.0f}%" if pd.notna(x) else "-"


def bins_table(g, col, q=5):
    x = g[[col, "хай перебит за 30д", "шорт R"]].dropna(subset=[col])
    if len(x) < 50 or x[col].nunique() < 3:
        return None
    try:
        b = pd.qcut(x[col], q, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for iv, part in x.groupby(b, observed=True):
        rows.append({"признак": col, "диапазон": f"{iv.left:.3g} … {iv.right:.3g}", "вершин": len(part),
                     "хай перебит": pct(part["хай перебит за 30д"].mean()),
                     "шорт R": f"{part['шорт R'].mean():+.2f}" if part["шорт R"].notna().any() else "-"})
    return rows


def combine():
    w, e, d = load("waves"), load("eps"), load("daily")
    w = w[w["отслежено часов"] >= AFTER].copy()          # исход известен полностью
    w["хай перебит за 30д"] = w["хай перебит за 30д"].astype(bool)
    w.to_csv(f"{OUT}/waves.csv.gz", index=False)
    e.to_csv(f"{OUT}/episodes.csv", index=False)
    d.to_csv(f"{OUT}/featured_daily.csv", index=False)
    out = open(f"{OUT}/pump_life_summary.md", "w")
    wr = out.write
    wr("# Жизнь пампа до смерти (Binance USDT-фьючерсы, 1h, пампы с 2025-01-01)\n\n")
    wr("Памп = +30% от минимума за 72 ч. Волна = зигзаг 20%. Момент наблюдения = закрытие часовой свечи, на которой "
       "откат от вершины волны впервые достиг −20%. «Хай перебит» = в следующие 30 дней цена выше вершины "
       "(шортист со стопом за хаем выбит). Шорт R: шорт по закрытию свечи момента, стоп за вершиной +0,5%, "
       f"тейк {RR:g}R, 10 дней; безубыточный винрейт {1 / (1 + RR) * 100:.0f}%, комиссии не учтены. "
       "OI, соотношения L/S — архив metrics Binance; фандинг — последний расчётный (% за 8ч). "
       "Капа — грубая оценка по текущей капе CoinGecko.\n\n")
    wr(f"Эпизодов: {len(e)}, монет: {e['монета'].nunique()}. Вершин волн с известным исходом: {len(w)}.\n\n")

    wr("## 1. Сколько волн бывает\n\n")
    vc = e["волн"].clip(upper=6).value_counts().sort_index()
    wr("| волн в эпизоде | эпизодов |\n|---|---|\n" + "".join(f"| {'6+' if i == 6 else i} | {v} |\n" for i, v in vc.items()) + "\n")
    wr("Чем закончились эпизоды: " + ", ".join(f"{k}: {v}" for k, v in e["конец"].value_counts().items()) + "\n\n")

    wr("## 2. Откат уже X% от вершины — какой шанс, что хай всё-таки перебьют\n\n")
    rows = []
    for name, g in (("все вершины", w), ("1-я волна", w[w["волна"] == 1]), ("2-я волна", w[w["волна"] == 2]),
                    ("3-я+ волна", w[w["волна"] >= 3]), ("вершина = хай эпизода", w[w["вершина = хай эпизода"]]),
                    ("рост от старта ≥100%", w[w["рост от старта, %"] >= 100]),
                    ("нет спота на Binance", w[~w["есть спот"].astype(bool)]),
                    ("есть спот", w[w["есть спот"].astype(bool)])):
        r = {"группа": name, "вершин": len(g)}
        for dp in DEPTHS:
            gg = g[g[f"дошла до −{dp}%"].astype(bool)]
            r[f"после −{dp}%"] = f"{pct(gg['хай перебит за 30д'].mean())} ({len(gg)})" if len(gg) else "-"
        for hrs in (24, 72, 168):
            gg = g[g[f"без хая {hrs}ч"].astype(bool)]
            r[f"{hrs}ч без хая"] = pct(gg["хай перебит за 30д"].mean()) if len(gg) else "-"
        rows.append(r)
    wr("Внимание: «после −50%» считается только среди вершин, которые дошли до −50% раньше перебоя хая.\n\n")
    wr(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")

    wr("## 3. Чем вершины, которые перебили, отличаются от тех, что стали «смертью» (медианы в момент −20%)\n\n")
    rows = []
    for col in FEATS:
        a, b = w.loc[w["хай перебит за 30д"], col], w.loc[~w["хай перебит за 30д"], col]
        rows.append({"признак": col, "хай перебит (медиана)": f"{a.median():.3g}", "не перебит (медиана)": f"{b.median():.3g}",
                     "есть данных": f"{w[col].notna().mean() * 100:.0f}%"})
    wr(f"Доля перебитых вершин в целом: {pct(w['хай перебит за 30д'].mean())}; средний шорт R по всем: {w['шорт R'].mean():+.2f}.\n\n")
    wr(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")

    wr("## 4. Каждый признак по пятым частям (квинтили): доля перебитых хаев и результат шорта\n\n")
    allrows = []
    for col in FEATS:
        r = bins_table(w, col)
        if r:
            allrows += r
    wr(pd.DataFrame(allrows).to_markdown(index=False) + "\n\n")

    wr("## 5. Скорость смерти: от финального хая эпизода\n\n")
    ee = e[e["отслежено часов после хая"] >= 720].copy()
    for col, edges, lab in (("часов до хая", [0, 12, 24, 72, 168, 720, 10 ** 6], ["<12ч", "12–24ч", "1–3д", "3–7д", "7–30д", "30д+"]),
                            ("рост до хая, %", [30, 50, 100, 200, 500, 10 ** 6], ["30–50%", "50–100%", "100–200%", "200–500%", "500%+"]),
                            ("оборот старт→хай / норма", [0, 3, 10, 30, 100, 10 ** 9], ["<x3", "x3–10", "x10–30", "x30–100", "x100+"])):
        ee["_b"] = pd.cut(ee[col], edges, labels=lab, right=False)
        rows = []
        for b, g in ee.groupby("_b", observed=True):
            rows.append({col: b, "эпизодов": len(g), "волн (медиана)": g["волн"].median(),
                         "дошли до −50% за 30д": pct((g["макс падение 30д, %"] <= -50).mean()),
                         "часов до −30% (медиана)": g["ч до −30%"].median(), "часов до −50% (медиана)": g["ч до −50%"].median(),
                         "макс падение за 30д (медиана), %": round(g["макс падение 30д, %"].median(), 1)})
        wr(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    wr("По споту: " + "; ".join(f"{'есть спот' if k else 'нет спота'}: {len(g)} эп., падение 30д медиана "
                                f"{g['макс падение 30д, %'].median():.0f}%, ч до −50% медиана {g['ч до −50%'].median():.0f}"
                                for k, g in ee.groupby(ee["есть спот"].astype(bool))) + "\n\n")

    wr("## 6. Кейсы: " + ", ".join(FEATURED) + "\n\n")
    show = ["волна", "вершина", "момент −20%", "рост волны, %", "рост от старта, %", "часов волны", "оборот волны / норма",
            "OI за волну, %", "OI за откат, %", "фандинг в момент, %", "фандинг мин с начала пампа, %",
            "доля спота в волне, %", "топ-трейдеры L/S в момент", "хай перебит за 30д", "насколько выше хая ушла, %",
            "макс откат до перебоя, %", "шорт R"]
    wall = load("waves")
    for sym in FEATURED:
        g = wall[wall["монета"] == sym]
        if len(g) == 0:
            wr(f"### {sym}\nНет пампов ≥30% с 2025 года (или монеты нет в архиве фьючерсов).\n\n")
            continue
        for ep_id, gg in g.groupby("эпизод", sort=False):
            er = e[e["эпизод"] == ep_id].iloc[0]
            wr(f"### {ep_id} — рост {er['рост до хая, %']:.0f}% за {er['часов до хая']} ч, волн {er['волн']}, "
               f"{er['конец']}, падение за 30д {er['макс падение 30д, %']:.0f}%\n\n")
            wr(gg[show].round(2).to_markdown(index=False) + "\n\n")
    wr("Подневные данные по кейсам (OI, фандинг, доля спота, L/S) — featured_daily.csv.\n\n")

    wr("## 7. Самые большие пампы\n\n")
    wr(e.sort_values("рост до хая, %", ascending=False).head(30).round(1).to_markdown(index=False) + "\n")
    out.close()
    print(open(f"{OUT}/pump_life_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
