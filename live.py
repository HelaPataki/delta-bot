"""
Живой бот алертов (Coinalyze + Telegram). Две стратегии, проверенные на истории:

ШОРТ  — фандинг 24ч ≤ −0,2% (в пересчёте на 8 ч) + открытый интерес за 24ч ≥ +60% + цена за 24ч не упала больше 2%,
        только первый сигнал подряд по монете; вход по рынку, стоп +4,3%, тейк −7,74%, выход через 24 ч.
ЛОНГ  — объём 4ч-свечи ≥ x5 к медиане 12 прошлых, дельта ≤ −17%, POC в нижних 30% свечи, закрытие не ниже POC;
        условный ордер на POC + 5 тиков (живёт 4 ч), стоп −4,3%, тейк +17,2%, держать до 10 дней.
Общие фильтры: оборот свечи ≥ $50k, капитализация $10M–$400M.

Coinalyze даёт 40 монето-запросов в минуту, поэтому бот запускается заранее (~80 мин до закрытия свечи):
  1) за ~45 мин отбирает кандидатов в шорт (прогнозный фандинг < 0), за ~22 мин — в лонг (объём свечи уже большой);
  2) ждёт закрытия свечи и через минуту проверяет только кандидатов;
  3) шлёт алерты и напоминания (отменить ордер, закрыть шорт, закрыть лонг).
MODE=test — сразу проверяет последнюю закрытую свечу, сообщения помечены «ТЕСТ», напоминания не ставятся.
"""

import csv
import io
import json
import os
import random
import sys
import time
import traceback
import zipfile
import datetime as dt
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np
import requests

# ---------------- настройки стратегии ----------------
MCAP_MIN, MCAP_MAX = 10e6, 400e6
MIN_TURNOVER = 50_000
LEV = 2
STOP = 4.3
SHORT_TAKE = 4.3 * 1.8            # 7,74%
LONG_TAKE = 4.3 * 4               # 17,2%
SHORT_HOLD_H, LONG_HOLD_H = 24, 240
ENTRY_DELAY_MIN = 8               # в расчётах вход через ~8 мин после закрытия свечи
TICKS = 5
LATE_MIN = 15                     # алерт позже 15 мин после закрытия свечи помечается «С ОПОЗДАНИЕМ»

# ---------------- техническое ----------------
CZ = "https://api.coinalyze.net/v1"
ARCHIVE = "https://data.binance.vision"
KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600
STATE_FILE, SIGNALS_FILE = "live_state.json", "live_signals.csv"
MODE = os.environ.get("MODE", "live")
CALLS_PER_MIN = 38

TOKEN = os.environ.get("TG_TOKEN", "")
CHAT = os.environ.get("TG_CHAT", "")
KEY = os.environ.get("COINALYZE_KEY", "")


def log(*a):
    print(dt.datetime.now(KYIV).strftime("%H:%M:%S"), *a, flush=True)


def kyiv(ts, fmt="%d.%m, %H:%M"):
    return dt.datetime.fromtimestamp(ts, KYIV).strftime(fmt)


# ---------------- Telegram ----------------
def tg(text):
    if not TOKEN:
        print("---- TELEGRAM ----\n" + text + "\n------------------")
        return True
    for attempt in range(3):
        try:
            r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                              data={"chat_id": CHAT, "text": text[:4000]}, timeout=30)
            if r.status_code == 200:
                return True
            log("telegram:", r.status_code, r.text[:150])
            if r.status_code != 429 and 400 <= r.status_code < 500:
                return False
        except requests.RequestException as e:
            log("telegram ошибка:", e)
        time.sleep(5 * (attempt + 1))
    return False


# ---------------- Coinalyze с учётом лимита ----------------
_used = deque()   # (время, сколько монет)


def _take(k):
    while True:
        now = time.time()
        while _used and now - _used[0][0] > 61:
            _used.popleft()
        if sum(x[1] for x in _used) + k <= CALLS_PER_MIN:
            _used.append((now, k))
            return
        time.sleep(1)


def cz(path, cost=1, **params):
    for _ in range(6):
        _take(cost)
        try:
            r = requests.get(CZ + path, params=params, headers={"api_key": KEY}, timeout=40)
        except requests.RequestException as e:
            log("coinalyze сеть:", e)
            time.sleep(10)
            continue
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", "30"))
            except ValueError:
                wait = 30
            time.sleep(wait + 1)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Coinalyze {path} -> {r.status_code}: {r.text[:150]}")
        return r.json()
    raise RuntimeError(f"Coinalyze {path}: не удалось получить данные")


def history(endpoint, syms, interval, frm, to, **extra):
    """{символ Coinalyze: [бары]} — бары отсортированы по времени, t в секундах."""
    out = {}
    syms = list(syms)
    for i in range(0, len(syms), 20):
        batch = syms[i:i + 20]
        try:
            data = cz(f"/{endpoint}", cost=len(batch), symbols=",".join(batch), interval=interval,
                      **{"from": int(frm), "to": int(to)}, **extra)
        except RuntimeError as e:
            log(e)
            continue
        for item in data:
            out[item["symbol"]] = sorted(item.get("history") or [], key=lambda b: b["t"])
    return out


def funding_now(syms):
    out = {}
    syms = list(syms)
    for i in range(0, len(syms), 20):
        batch = syms[i:i + 20]
        try:
            for item in cz("/funding-rate", cost=len(batch), symbols=",".join(batch)):
                if item.get("value") is not None:
                    out[item["symbol"]] = float(item["value"])
        except RuntimeError as e:
            log(e)
    return out


# ---------------- список монет ----------------
def base_asset(sym):
    b = sym[:-4]
    for p in ("1000000", "100000", "10000", "1000", "1M"):
        if b.startswith(p) and len(b) > len(p):
            return b[len(p):]
    return b


def coingecko_mcaps(pages=10):
    caps = {}
    for page in range(1, pages + 1):
        data = None
        for _ in range(5):
            try:
                r = requests.get("https://api.coingecko.com/api/v3/coins/markets", timeout=30,
                                 params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page})
                if r.status_code == 200:
                    data = r.json()
                    break
            except requests.RequestException:
                pass
            time.sleep(20)
        if not data:
            continue
        for c in data:
            s, m = (c.get("symbol") or "").upper(), c.get("market_cap") or 0
            if m and m > caps.get(s, 0):
                caps[s] = m
        if (data[-1].get("market_cap") or 0) < MCAP_MIN:
            break
        time.sleep(6)
    return caps


def universe():
    """{символ Coinalyze: символ Binance} для USDT-перпетуалов Binance с капитализацией $10M–$400M."""
    caps = coingecko_mcaps()
    out = {}
    for m in cz("/future-markets"):
        s = m.get("symbol", "")
        if not s.endswith(".A") or not m.get("is_perpetual") or m.get("quote_asset") != "USDT":
            continue
        b = (m.get("symbol_on_exchange") or s.split("_")[0]).upper()
        if not b.endswith("USDT"):
            continue
        cap = caps.get(base_asset(b), 0)
        if MCAP_MIN <= cap <= MCAP_MAX:
            out[s] = b
    log(f"монет в работе: {len(out)} (капитализация ${MCAP_MIN/1e6:g}M–${MCAP_MAX/1e6:g}M)")
    return out


# ---------------- интервал фандинга (из архива Binance) ----------------
_intervals = {}


def funding_interval(bsym):
    if bsym in _intervals:
        return _intervals[bsym]
    hours = None
    today = dt.date.today().replace(day=1)
    for back in (1, 2):
        m = today
        for _ in range(back):
            m = (m - dt.timedelta(days=1)).replace(day=1)
        url = f"{ARCHIVE}/data/futures/um/monthly/fundingRate/{bsym}/{bsym}-fundingRate-{m:%Y-%m}.zip"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))))
            rows = [x for x in rows if x and x[0].strip().isdigit()]
            if rows:
                hours = float(rows[-1][1])
                break
        except Exception as e:  # noqa: BLE001
            log("интервал фандинга", bsym, e)
    _intervals[bsym] = hours or 4.0      # новой монеты нет в прошлом месяце: чаще всего 4 ч
    return _intervals[bsym]


def prefetch_intervals(bsyms):
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(funding_interval, bsyms))


# ---------------- расчёт признаков ----------------
def bar_at(bars, t):
    for b in bars:
        if b["t"] == t:
            return b
    return None


def candle_feats(bars, T):
    """Признаки 4ч-свечи, закрывающейся в T (для незакрытой свечи — по текущим данным)."""
    idx = {b["t"]: i for i, b in enumerate(bars)}
    i = idx.get(T - H4)
    if i is None or i < 12:
        return None
    qv = np.array([(b["o"] + b["h"] + b["l"] + b["c"]) / 4 * (b.get("v") or 0) for b in bars])
    cur, prev = bars[i], bars[i - 12:i]
    norm = float(np.median(qv[i - 12:i]))
    v, bv = cur.get("v") or 0, cur.get("bv")
    f = {
        "close": cur["c"], "high": cur["h"], "low": cur["l"], "turnover": qv[i],
        "vol_ratio": qv[i] / norm if norm > 0 else np.nan,
        "delta_pct": (2 * bv - v) / v * 100 if v and bv is not None else np.nan,
        "px24": (cur["c"] / bars[i - 6]["c"] - 1) * 100 if i >= 6 and idx.get(T - 7 * H4) == i - 6 else np.nan,
        "px24_prev": (bars[i - 1]["c"] / bars[i - 7]["c"] - 1) * 100 if i >= 7 and idx.get(T - 8 * H4) == i - 7 else np.nan,
        "turnover_prev": qv[i - 1] if idx.get(T - 2 * H4) == i - 1 else 0.0,
    }
    return f


def fund24(bars, X, hours):
    """Средний расчётный фандинг за (X−24ч, X], в % на 8 ч. Ставка в момент расчёта = закрытие часового бара перед ним."""
    step = int(hours * H)
    s = (X // step) * step
    rates = []
    while s > X - 24 * H:
        b = bar_at(bars, s - H)
        if b is not None and b.get("c") is not None:
            rates.append(b["c"] * 8 / hours)
        s -= step
    return float(np.mean(rates)) if rates else np.nan


def oi_at(bars, X):
    """Открытый интерес на момент X: закрытие 5-мин бара, закончившегося не позже X и не раньше X−30 мин."""
    best = None
    for b in bars:
        end = b["t"] + 300
        if end <= X and X - end <= 1800:
            best = b
    return best["c"] if best else np.nan


def short_feats(fbars, obars, T, hours):
    o0, o24, o4, o28 = (oi_at(obars, x) for x in (T, T - 24 * H, T - 4 * H, T - 28 * H))
    return {"fund24": fund24(fbars, T, hours), "oi_chg24": (o0 / o24 - 1) * 100,
            "fund24_prev": fund24(fbars, T - 4 * H, hours), "oi_chg24_prev": (o4 / o28 - 1) * 100}


def is_short(c, s):
    """c — признаки свечи, s — фандинг/OI. Условия финального шорта + «первый сигнал подряд»."""
    if c is None or s is None:
        return False
    prev_g1 = (c["turnover_prev"] >= MIN_TURNOVER and s["fund24_prev"] <= -0.03
               and s["oi_chg24_prev"] >= 20 and c["px24_prev"] >= -2)
    return bool(c["turnover"] >= MIN_TURNOVER and s["fund24"] <= -0.2 and s["oi_chg24"] >= 60
                and c["px24"] >= -2 and not prev_g1)


def infer_tick(bars):
    prices = np.unique(np.round([b[k] for b in bars for k in ("o", "h", "l", "c")], 10))
    d = np.diff(prices)
    d = d[d > 0]
    return float(np.round(d.min(), 10)) if len(d) else np.nan


def poc_info(mbars, T):
    w = [b for b in mbars if T - H4 <= b["t"] < T]
    if len(w) < 60:
        return None
    top = max(w, key=lambda b: b.get("v") or 0)
    poc = (top["h"] + top["l"]) / 2
    hi, lo = max(b["h"] for b in w), min(b["l"] for b in w)
    pos = (poc - lo) / (hi - lo) * 100 if hi > lo else 50.0
    return {"poc": poc, "poc_pos": pos, "tick": infer_tick(w)}


def is_long(c, p):
    if c is None or p is None:
        return False
    return bool(c["turnover"] >= MIN_TURNOVER and c["vol_ratio"] >= 5 and c["delta_pct"] <= -17
                and p["poc_pos"] <= 30 and c["close"] >= p["poc"])


def fmt_price(x, tick):
    dec = max(0, int(round(-np.log10(tick)))) if tick and tick > 0 else 6
    return f"{x:.{dec}f}"


# ---------------- тексты ----------------
def short_text(sym, others, T, test):
    roi_sl, roi_tp = STOP * LEV, SHORT_TAKE * LEV
    lines = ["🧪 ТЕСТ — НЕ ТОРГОВАТЬ 🧪"] if test else []
    lines += ["🔴🔴🔴 ШОРТ 🔴🔴🔴", sym, "━━━━━━━━━━━━━━",
              "1️⃣ Рыночный → Продать/Шорт 🔴",
              f"2️⃣ Стоп-лосс ROI:   {roi_sl:.1f} %",
              f"3️⃣ Тейк-профит ROI: {roi_tp:.1f} %",
              f"4️⃣ Закрыть до: {kyiv(T + (SHORT_HOLD_H * 60 + ENTRY_DELAY_MIN) * 60)}",
              "━━━━━━━━━━━━━━",
              f"Плечо: {LEV}x | Весь свободный слот",
              "Оба слота заняты — пропустить"]
    if others:
        lines.append(f"Другие монеты свечи (не трогаем): {', '.join(others)}")
    return "\n".join(lines)


def long_text(sym, level, tick, others, T, test):
    roi_sl, roi_tp = STOP * LEV, LONG_TAKE * LEV
    lines = ["🧪 ТЕСТ — НЕ ТОРГОВАТЬ 🧪"] if test else []
    lines += ["🟢🟢🟢 ЛОНГ 🟢🟢🟢", sym, "━━━━━━━━━━━━━━",
              "1️⃣ Условные → Стоп:",
              f"   {fmt_price(level, tick)}",
              "   Рыночная цена → Купить/Лонг 🟢",
              f"2️⃣ Стоп-лосс ROI:   {roi_sl:.1f} %",
              f"3️⃣ Тейк-профит ROI: {roi_tp:.1f} %",
              f"4️⃣ Ордер живёт до: {kyiv(T + H4)}",
              "   Не сработал — отменить",
              "5️⃣ Держать до 10 дней (бот напомнит)",
              "━━━━━━━━━━━━━━",
              f"Плечо: {LEV}x | Весь свободный слот",
              "Оба слота заняты — пропустить"]
    if others:
        lines.append(f"Другие монеты свечи (не трогаем): {', '.join(others)}")
    return "\n".join(lines)


# ---------------- состояние и журнал ----------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {"last_T": 0, "orders": [], "shorts": [], "longs": []}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)


SIG_COLS = ["свеча закрылась (Киев)", "сторона", "монета", "выбрана", "цена закрытия", "объём x", "дельта %",
            "POC", "POC позиция %", "цена ордера", "фандинг 24ч %", "OI 24ч %", "цена 24ч %", "оборот $"]


def journal(rows):
    new = not os.path.exists(SIGNALS_FILE)
    with open(SIGNALS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIG_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SIG_COLS})


def rnd(x, n=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ---------------- напоминания ----------------
def reminders(st, now, uni_rev):
    keep = []
    for o in st["orders"]:
        if o["expires"] > now + 300:
            keep.append(o)
            continue
        cs = uni_rev.get(o["sym"])
        bars = history("ohlcv-history", [cs], "1min", o["from"], o["expires"]).get(cs, []) if cs else []
        t_fill, prev = None, None
        for b in bars:
            lvl = o["level"]
            if (b["l"] <= lvl <= b["h"]) or (prev is not None and (prev < lvl < b["l"] or b["h"] < lvl < prev)):
                t_fill = b["t"]
                break
            prev = b["c"]
        if t_fill:
            close_at = t_fill + LONG_HOLD_H * H
            st["longs"].append({"sym": o["sym"], "close_at": close_at})
            tg(f"🟢 {o['sym']}: ордер сработал в {kyiv(t_fill, '%H:%M')}\nДержим до {kyiv(close_at)} (бот напомнит)")
        elif bars:
            tg(f"⏰ {o['sym']}: ордер НЕ сработал\n👉 Отмени условный ордер")
        else:
            tg(f"⏰ {o['sym']}: время ордера вышло\n👉 Проверь: не сработал — отмени")
    st["orders"] = keep

    keep = []
    for s in st["shorts"]:
        if s["close_at"] <= now + 1800:
            tg(f"⏰🔴 {s['sym']}: прошло 24 часа\n👉 Закрой ШОРТ по рынку, если ещё открыт")
        else:
            keep.append(s)
    st["shorts"] = keep

    keep = []
    for s in st["longs"]:
        if s["close_at"] <= now + 2 * H:
            tg(f"⏰🟢 {s['sym']}: 10 дней истекают в {kyiv(s['close_at'], '%H:%M')}\n👉 Закрой ЛОНГ по рынку, если ещё открыт")
        else:
            keep.append(s)
    st["longs"] = keep


# ---------------- основной цикл ----------------
def sleep_until(ts):
    while time.time() < ts:
        time.sleep(min(30, max(1, ts - time.time())))


def prescan_short(uni, T, deadline):
    """Кандидаты в шорт (за ~45 мин до закрытия): история фандинга по всем монетам →
    прогнозная ставка < 0 или выплаченный фандинг 24ч ≤ −0,1% → OI за 24ч ≥ +30%."""
    now = time.time()
    fb = history("funding-rate-history", list(uni), "1hour", now - 30 * H, now)
    neg = [s for s, b in fb.items() if b and any((x.get("c") or 0) < 0 for x in b[-25:])]
    prefetch_intervals([uni[s] for s in neg])
    X = int(now // 300 * 300)
    cand = []
    for s in neg:
        hours = funding_interval(uni[s])
        pred = fb[s][-1]["c"] * 8 / hours
        if pred < 0 or fund24(fb[s], X, hours) <= -0.1:
            cand.append(s)
    log(f"фандинг получен для {len(fb)}, кандидатов по фандингу {len(cand)}")
    wl = []
    for i in range(0, len(cand), 20):
        if time.time() > deadline:
            log("не успел проверить OI у всех — остальные идут в список целиком")
            wl += cand[i:]
            break
        batch = cand[i:i + 20]
        ob = history("open-interest-history", batch, "5min", now - 30 * H, time.time(), convert_to_usd="false")
        for s in batch:
            o0, o24 = oi_at(ob.get(s, []), X), oi_at(ob.get(s, []), X - 24 * H)
            oi = (o0 / o24 - 1) * 100
            if oi >= 30 or np.isnan(oi):
                wl.append(s)
    log(f"кандидатов в шорт: {len(wl)}")
    return wl


def prescan_long(uni, T):
    """Кандидаты в лонг: объём текущей (ещё не закрытой) свечи уже ≥ 2,5x к обычному."""
    now = time.time()
    bars = history("ohlcv-history", list(uni), "4hour", T - 20 * H4, now)
    wl = []
    for s, b in bars.items():
        f = candle_feats(b, T)
        if f and f["vol_ratio"] >= 2.5:
            wl.append(s)
    log(f"кандидатов в лонг: {len(wl)} (из {len(bars)} с данными)")
    return wl


def final(uni, T, long_wl, short_wl, test, late_note=""):
    syms = sorted(set(long_wl) | set(short_wl))
    log(f"проверка после закрытия: {len(syms)} монет")
    to = T + 60
    bars = history("ohlcv-history", syms, "4hour", T - 20 * H4, to)
    feats = {s: candle_feats(b, T) for s, b in bars.items()}
    rows, shorts, longs = [], [], []

    if short_wl:
        fb = history("funding-rate-history", short_wl, "1hour", T - 30 * H, to)
        ob = history("open-interest-history", short_wl, "5min", T - 30 * H, to, convert_to_usd="false")
        for s in short_wl:
            c = feats.get(s)
            if c is None or s not in fb or s not in ob:
                continue
            sf = short_feats(fb[s], ob[s], T, funding_interval(uni[s]))
            if is_short(c, sf):
                shorts.append(uni[s])
                rows.append({"сторона": "шорт", "монета": uni[s], "цена закрытия": c["close"], "фандинг 24ч %": rnd(sf["fund24"], 3),
                             "OI 24ч %": rnd(sf["oi_chg24"], 1), "цена 24ч %": rnd(c["px24"]), "оборот $": rnd(c["turnover"], 0),
                             "объём x": rnd(c["vol_ratio"]), "дельта %": rnd(c["delta_pct"])})

    pre = [s for s in long_wl if feats.get(s) and feats[s]["vol_ratio"] >= 5 and feats[s]["delta_pct"] <= -17
           and feats[s]["turnover"] >= MIN_TURNOVER]
    levels = {}
    if pre:
        mb = history("ohlcv-history", pre, "1min", T - H4, T - 60)
        for s in pre:
            p = poc_info(mb.get(s, []), T)
            if is_long(feats[s], p):
                lvl = round((p["poc"] + TICKS * p["tick"]) / p["tick"]) * p["tick"]
                levels[uni[s]] = (lvl, p["tick"])
                longs.append(uni[s])
                c = feats[s]
                rows.append({"сторона": "лонг", "монета": uni[s], "цена закрытия": c["close"], "объём x": rnd(c["vol_ratio"]),
                             "дельта %": rnd(c["delta_pct"]), "POC": p["poc"], "POC позиция %": rnd(p["poc_pos"], 1),
                             "цена ордера": fmt_price(lvl, p["tick"]), "оборот $": rnd(c["turnover"], 0),
                             "цена 24ч %": rnd(c["px24"])})

    pick_s = random.choice(shorts) if shorts else None
    pick_l = random.choice(longs) if longs else None
    for r in rows:
        r["свеча закрылась (Киев)"] = kyiv(T, "%Y-%m-%d %H:%M")
        r["выбрана"] = "да" if r["монета"] in (pick_s, pick_l) else "нет"
    if time.time() - T > LATE_MIN * 60 and not test:
        late_note = f"⚠️ С ОПОЗДАНИЕМ: свеча закрылась в {kyiv(T, '%H:%M')}, сейчас {kyiv(time.time(), '%H:%M')}\n"
    if pick_s:
        tg(late_note + short_text(pick_s, [x for x in shorts if x != pick_s], T, test))
    if pick_l:
        lvl, tick = levels[pick_l]
        tg(late_note + long_text(pick_l, lvl, tick, [x for x in longs if x != pick_l], T, test))
    log(f"сигналы: шорт {shorts}, лонг {longs}")
    return rows, pick_s, pick_l, levels


def main():
    test = MODE == "test"
    now = time.time()
    st = load_state()
    if test:
        T = int(now // H4 * H4)
    else:
        last_close = int(now // H4 * H4)
        if now - last_close < 40 * 60 and st.get("last_T") != last_close:
            T = last_close                      # запуск опоздал: свеча уже закрылась, проверяем её
        else:
            T = last_close + H4
        if T - now > 2.5 * H:
            log("слишком рано, следующая свеча нескоро — выхожу")
            return
        if st.get("last_T") == T:
            log("эта свеча уже обработана")
            return
    late_note = ""
    started_late = (T - now) < 30 * 60 and not test

    uni = universe()
    uni_rev = {b: s for s, b in uni.items()}

    if test:
        tg(f"🧪 Тест запущен: проверяю свечу, закрывшуюся в {kyiv(T, '%H:%M')}. Это займёт ~15–25 минут.")
        short_wl = prescan_short(uni, T, deadline=time.time() + 30 * 60)
        long_wl = list(uni)
    elif started_late:
        log("запуск поздний: предварительный отбор пропускаю")
        short_wl, long_wl = None, list(uni)
    else:
        sleep_until(T - 45 * 60)
        short_wl = prescan_short(uni, T, deadline=T - 25 * 60)
        sleep_until(T - 22 * 60)
        long_wl = prescan_long(uni, T)

    if not test:
        sleep_until(T + 70)
    if short_wl is None:
        short_wl = [s for s, v in funding_now(uni).items() if v < 0]
        prefetch_intervals([uni[s] for s in short_wl])

    rows, pick_s, pick_l, levels = final(uni, T, long_wl, short_wl, test, late_note)
    shorts_n = sum(r["сторона"] == "шорт" for r in rows)
    longs_n = sum(r["сторона"] == "лонг" for r in rows)

    if test:
        tg(f"🧪 Тест закончен. Свеча {kyiv(T, '%H:%M')}: проверено монет {len(uni)}, "
           f"шортов {shorts_n}, лонгов {longs_n}. Бот работает ✅")
        return

    journal(rows)
    if pick_s:
        st["shorts"].append({"sym": pick_s, "close_at": T + (SHORT_HOLD_H * 60 + ENTRY_DELAY_MIN) * 60})
    if pick_l:
        st["orders"].append({"sym": pick_l, "level": levels[pick_l][0], "from": T + 60, "expires": T + H4})
    reminders(st, time.time(), uni_rev)
    st["last_T"] = T
    save_state(st)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        tg("⚠️ Дельта-бот: ошибка при проверке рынка\n" + err[-600:])
        sys.exit(1)
