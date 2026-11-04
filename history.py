"""
delta-bot: историческая прогонка.

Сигнал = закрытая 4h-свеча USDT-перпетуала Binance, у которой
    дельта % = (покупки - продажи) / объём * 100 = (2*taker_buy - volume) / volume * 100
    >= +DELTA_PCT  или  <= -DELTA_PCT
плюс фильтры: оборот свечи (USDT) >= MIN_TURNOVER, оценка капитализации в диапазоне.

Для каждого сигнала считаем, что было с ценой 10 дней после закрытия свечи.
Для сравнения считаем то же самое для ВСЕХ свечей, прошедших фильтры оборота и
капитализации (baseline), — чтобы понять, даёт ли дельта что-то сверх обычного.

Данные: архив Binance (data.binance.vision), капитализация: CoinGecko.
"""

import io
import os
import sys
import time
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- настройки ----------------
START = os.environ.get("START", "2025-01-01")
DELTA_PCT = float(os.environ.get("DELTA_PCT", "17"))
MIN_TURNOVER = float(os.environ.get("MIN_TURNOVER", "50000"))   # USDT за 4h-свечу
MCAP_MIN = float(os.environ.get("MCAP_MIN", "10000000"))
MCAP_MAX = float(os.environ.get("MCAP_MAX", "400000000"))
HORIZON_CANDLES = 60                      # 10 дней * 6 свечей по 4h
LEVELS = [4, 10, 20]                      # % от цены закрытия сигнальной свечи
VOL_LOOKBACK = 12                         # сколько предыдущих 4h-свечей считаем «нормой» объёма
VOL_MULTS = [3, 5, 10]                    # во сколько раз объём свечи выше нормы
VOL_MIN = float(os.environ.get("VOL_MIN", "3"))  # минимальный множитель для файла сигналов
LIMIT_SYMBOLS = int(os.environ.get("LIMIT_SYMBOLS", "0"))  # 0 = все (для теста можно 20)
OUT_DIR = os.environ.get("OUT_DIR", "results")

KYIV = ZoneInfo("Europe/Kyiv")
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE = "https://data.binance.vision"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

session = requests.Session()
session.headers["User-Agent"] = "delta-bot-history"


def get(url, tries=4, **kw):
    for i in range(tries):
        try:
            r = session.get(url, timeout=60, **kw)
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return None


# ---------------- список монет ----------------
def list_symbols():
    """Все USDT-перпетуалы из архива, включая уже делистнутые."""
    prefix = "data/futures/um/monthly/klines/"
    syms, marker = [], ""
    while True:
        r = get(S3, params={"prefix": prefix, "delimiter": "/", "marker": marker})
        r.raise_for_status()
        root = ElementTree.fromstring(r.content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for p in root.findall("s:CommonPrefixes/s:Prefix", ns):
            syms.append(p.text[len(prefix):].strip("/"))
        truncated = root.find("s:IsTruncated", ns).text == "true"
        if not truncated:
            break
        marker = root.findall("s:CommonPrefixes/s:Prefix", ns)[-1].text
    return sorted(s for s in syms if s.endswith("USDT") and "_" not in s)


def base_asset(sym):
    b = sym[:-4]
    for p in ("1000000", "100000", "10000", "1000", "1M"):
        if b.startswith(p) and len(b) > len(p):
            return b[len(p):]
    return b


# ---------------- капитализация ----------------
def coingecko_mcaps(pages=14):
    """symbol(upper) -> текущая капитализация (берём монету с наибольшей капой при совпадении тикеров)."""
    caps = {}
    for page in range(1, pages + 1):
        r = None
        for i in range(6):
            r = get("https://api.coingecko.com/api/v3/coins/markets",
                    params={"vs_currency": "usd", "order": "market_cap_desc",
                            "per_page": 250, "page": page})
            if r is not None and r.status_code == 200:
                break
            time.sleep(20)
        if r is None or r.status_code != 200:
            print(f"CoinGecko page {page} failed", file=sys.stderr)
            continue
        for c in r.json():
            s, m = (c.get("symbol") or "").upper(), c.get("market_cap") or 0
            if m and m > caps.get(s, 0):
                caps[s] = m
        time.sleep(7)
    return caps


# ---------------- свечи ----------------
def months(start, end):
    d = dt.date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def read_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    if isinstance(df.iloc[0, 0], str):          # у новых файлов есть заголовок
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = COLS
    return df.astype(float)


def load_klines(sym):
    start = dt.date.fromisoformat(START)
    today = dt.datetime.now(dt.timezone.utc).date()
    this_month = dt.date(today.year, today.month, 1)
    frames = []
    for m in months(start, this_month):
        if m < this_month:
            url = f"{ARCHIVE}/data/futures/um/monthly/klines/{sym}/4h/{sym}-4h-{m:%Y-%m}.zip"
            r = get(url)
            if r is not None and r.status_code == 200:
                frames.append(read_zip(r.content))
        else:  # текущий месяц — дневные файлы
            d = m
            while d < today:
                url = f"{ARCHIVE}/data/futures/um/daily/klines/{sym}/4h/{sym}-4h-{d:%Y-%m-%d}.zip"
                r = get(url)
                if r is not None and r.status_code == 200:
                    frames.append(read_zip(r.content))
                d += dt.timedelta(days=1)
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
    return df.reset_index(drop=True)


# ---------------- расчёт ----------------
def window(arr, n):
    """Матрица n x HORIZON: значения следующих HORIZON свечей после каждой свечи (NaN за концом данных)."""
    padded = np.concatenate([arr, np.full(HORIZON_CANDLES + 1, np.nan)])
    return np.lib.stride_tricks.sliding_window_view(padded, HORIZON_CANDLES)[1:n + 1]


def day_of(idx):
    """Индекс свечи после сигнала (0 = следующая) -> номер дня 1..10."""
    return (4 * (idx + 1) + 23) // 24


def analyse(sym, df, mcap_now):
    n = len(df)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    v, qv = df["volume"].to_numpy(), df["quote_volume"].to_numpy()
    tbv, tbq = df["taker_buy_volume"].to_numpy(), df["taker_buy_quote_volume"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_pct = (2 * tbv - v) / v * 100
    delta_usd = 2 * tbq - qv
    change_pct = (c / o - 1) * 100
    mcap_est = mcap_now * c / c[-1]
    # норма объёма = медиана оборота предыдущих 12 свечей (медиана, чтобы одна прошлая свеча-выброс не задирала норму)
    norm = pd.Series(qv).shift(1).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(norm > 0, qv / norm, np.nan)       # капитализация на момент свечи ≈ текущая * (цена тогда / цена сейчас)

    H, L = window(h, n), window(l, n)
    tracked = np.sum(~np.isnan(H), axis=1)
    complete = tracked == HORIZON_CANDLES
    up = H / c[:, None] - 1
    dn = L / c[:, None] - 1

    res = {
        "symbol": sym,
        "open_time": df["open_time"].to_numpy(),
        "open": o, "close": c,
        "change_pct": change_pct,
        "delta_pct": delta_pct, "delta_usd": delta_usd,
        "turnover_usd": qv, "vol_ratio": vol_ratio, "mcap_est": mcap_est,
        "days_tracked": tracked * 4 / 24,
        "complete": complete,
    }
    first_up4 = first_dn4 = None
    for lev in LEVELS:
        for name, hit in ((f"up{lev}", up >= lev / 100), (f"dn{lev}", dn <= -lev / 100)):
            anyhit = hit.any(axis=1)
            first = np.argmax(hit, axis=1)
            res[f"{name}_day"] = np.where(anyhit, day_of(first), np.nan)
            if lev == 4:
                fi = np.where(anyhit, first, 10 ** 6)
                if name.startswith("up"):
                    first_up4 = fi
                else:
                    first_dn4 = fi
    res["first_move"] = np.select(
        [(first_up4 == 10 ** 6) & (first_dn4 == 10 ** 6), first_up4 < first_dn4, first_dn4 < first_up4],
        ["нет", "рост", "падение"], default="оба в одной свече")
    with np.errstate(invalid="ignore"):
        res["max_up_pct"] = np.nanmax(up, axis=1) * 100
        res["max_dn_pct"] = np.nanmin(dn, axis=1) * 100
    res["max_up_day"] = np.where(tracked > 0, day_of(np.nanargmax(np.nan_to_num(up, nan=-9), axis=1)), np.nan)
    res["max_dn_day"] = np.where(tracked > 0, day_of(np.nanargmin(np.nan_to_num(dn, nan=9), axis=1)), np.nan)

    out = pd.DataFrame(res)
    base_mask = (qv >= MIN_TURNOVER) & (mcap_est >= MCAP_MIN) & (mcap_est <= MCAP_MAX) & (v > 0)
    out = out[base_mask]
    out = out[out["open_time"] >= pd.Timestamp(START, tz="UTC").timestamp() * 1000]
    return out


def stats_row(name, part):
    p = part[part.complete]
    if len(p) == 0:
        return None
    r = {"группа": name, "свечей": len(part), "с полными 10 днями": len(p)}
    for lev in LEVELS:
        for d in ("up", "dn"):
            col = f"{d}{lev}_day"
            sign = "+" if d == "up" else "−"
            r[f"{sign}{lev}% за 10д"] = f"{p[col].notna().mean() * 100:.0f}%"
            r[f"{sign}{lev}% медиана дня"] = f"{p[col].median():.0f}" if p[col].notna().any() else "-"
    fm = p.first_move.value_counts(normalize=True) * 100
    r["сначала рост ±4%"] = f"{fm.get('рост', 0):.0f}%"
    r["сначала падение ±4%"] = f"{fm.get('падение', 0):.0f}%"
    r["медиана макс. роста"] = f"{p.max_up_pct.median():.1f}%"
    r["медиана макс. падения"] = f"{p.max_dn_pct.median():.1f}%"
    return r


def summarize(df, groups):
    rows = [r for r in (stats_row(n, df[m]) for n, m in groups) if r]
    return pd.DataFrame(rows).set_index("группа").T


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = list_symbols()
    print(f"символов в архиве: {len(syms)}")
    caps = coingecko_mcaps()
    print(f"капитализаций CoinGecko: {len(caps)}")
    with_cap = [s for s in syms if base_asset(s) in caps]
    no_cap = [s for s in syms if base_asset(s) not in caps]
    if LIMIT_SYMBOLS:
        with_cap = with_cap[:LIMIT_SYMBOLS]
    print(f"с капитализацией: {len(with_cap)}, без (делист/нет на CoinGecko): {len(no_cap)}")

    def work(sym):
        try:
            df = load_klines(sym)
            if df is None or len(df) < 2:
                return None
            return analyse(sym, df, caps[base_asset(sym)])
        except Exception as e:  # одна битая монета не должна ронять всё
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(8) as ex:
        for i, part in enumerate(ex.map(work, with_cap), 1):
            if part is not None and len(part):
                parts.append(part)
            if i % 50 == 0:
                print(f"обработано {i}/{len(with_cap)}")
    allc = pd.concat(parts, ignore_index=True)

    sig = allc[(allc.delta_pct.abs() >= DELTA_PCT) & (allc.vol_ratio >= VOL_MIN)].copy()
    t = pd.to_datetime(sig.open_time, unit="ms", utc=True).dt.tz_convert(KYIV)
    sig.insert(1, "candle_open_kyiv", t.dt.strftime("%Y-%m-%d %H:%M"))
    sig.insert(2, "candle_close_kyiv", (t + pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d %H:%M"))
    sig.insert(3, "weekday", (t + pd.Timedelta(hours=4)).dt.day_name())
    sig.insert(4, "delta_sign", np.where(sig.delta_pct > 0, "+", "−"))
    sig.insert(5, "candle_color", np.where(sig.close >= sig.open, "зелёная", "красная"))
    sig = sig.drop(columns=["open_time"]).sort_values("candle_close_kyiv")
    for col in ("change_pct", "delta_pct", "vol_ratio", "max_up_pct", "max_dn_pct"):
        sig[col] = sig[col].round(2)
    for col in ("delta_usd", "turnover_usd", "mcap_est"):
        sig[col] = sig[col].round(0)
    sig.to_csv(f"{OUT_DIR}/history_signals.csv", index=False)

    days = (allc.open_time.max() - allc.open_time.min()) / 86_400_000 or 1
    pos, neg = allc.delta_pct >= DELTA_PCT, allc.delta_pct <= -DELTA_PCT
    with open(f"{OUT_DIR}/summary.md", "w") as f:
        f.write("# Прогонка delta-bot: всплеск объёма + дельта\n\n")
        f.write(f"Период: с {START}, 4h-свечи USDT-перпетуалов Binance.\n")
        f.write(f"Фильтры для всех групп: оборот свечи ≥ ${MIN_TURNOVER:,.0f}, "
                f"капитализация ${MCAP_MIN/1e6:g}M–${MCAP_MAX/1e6:g}M (оценка).\n")
        f.write(f"Всплеск объёма = оборот свечи / медиана оборота {VOL_LOOKBACK} предыдущих свечей.\n")
        f.write(f"Монет с капитализацией: {len(with_cap)}; без капитализации (пропущены): {len(no_cap)}.\n\n")
        f.write("Уровни считаются от цены закрытия сигнальной свечи по high/low следующих 60 свечей (10 дней).\n\n")
        for m in VOL_MULTS:
            spike = allc.vol_ratio >= m
            groups = [(f"объём ≥ x{m}, дельта ≥ +{DELTA_PCT:g}%", spike & pos),
                      (f"объём ≥ x{m}, дельта ≤ −{DELTA_PCT:g}%", spike & neg),
                      (f"объём ≥ x{m}, любая дельта", spike)]
            f.write(f"## Объём ≥ x{m}\n\n")
            for name, mask in groups:
                f.write(f"- {name}: {mask.sum()} сигналов, {mask.sum()/days:.2f} в день\n")
            f.write("\n" + summarize(allc, groups).to_markdown() + "\n\n")
        f.write("## Для сравнения\n\n")
        f.write(summarize(allc, [(f"дельта ≥ +{DELTA_PCT:g}% без объёма", pos),
                                 (f"дельта ≤ −{DELTA_PCT:g}% без объёма", neg),
                                 ("все свечи", allc.delta_pct.notna())]).to_markdown() + "\n")
    print(open(f"{OUT_DIR}/summary.md").read())


if __name__ == "__main__":
    main()
