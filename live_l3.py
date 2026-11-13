"""
Живой бот L3 (отдельный от live.py, свои файлы l3_state.json / l3_signals.csv / l3_universe.json).

L3 ЛОНГ, проверка каждый час (по закрытию часовой свечи), как в тесте l3_hourly:
  • монеты: USDT-перпетуалы Binance с капой $10M–$2B (без акций/металлов);
  • карта ликвидаций «14д» (та же модель, что liq_map / liq_cap): шортовых ликвидаций в пределах +10% от цены
    ≥ 3% капы и больше, чем лонговых в пределах −10%;
  • все аккаунты L/S < 0,8 (толпа в шортах);
  • сигнал «свежий»: прошлые 24 часа условие не выполнялось; по монете одна сделка за раз;
  • вход по рынку, тейк +20%, стоп −10%, выход по времени через 7 дней; до 3 сделок одновременно по ⅓ депозита.

Данные: история до вчера — архив Binance (как в тесте), последние часы — Coinalyze (общий ключ с live.py, поэтому
бот бережливый): сначала по вчерашнему архиву отбираются монеты с L/S < 1,0, и только по ним идёт запрос в Coinalyze.
MODE=test — сообщения с пометкой ТЕСТ, состояние не сохраняется, в конце — кто ближе всего к сигналу.
"""

import csv
import datetime as dt
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

DAYS_BACK = 80
os.environ["START"] = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=DAYS_BACK)).isoformat()

import numpy as np
import pandas as pd
import requests

import history as Hst
import liq_cap as LC
import liq_map as LM
import pump_life as PL
from tradfi import TRADFI

MODE = os.environ.get("MODE", "live")
TEST = MODE == "test"
TOKEN = os.environ.get("TG_TOKEN", "")
CHAT = os.environ.get("TG_CHAT", "")
KEY = os.environ.get("COINALYZE_KEY", "")
CZ = "https://api.coinalyze.net/v1"
KYIV = ZoneInfo("Europe/Kyiv")
H = 3600 * 1000

CAP_MIN, CAP_MAX = 10e6, 2e9
UP_MIN, LS_MAX, PREFILTER_LS = 3.0, 0.8, 1.0
TAKE, STOP, HOLD_H, SLOTS = 20.0, 10.0, 168, 3
HL_DAYS = 14
CALLS_PER_MIN = int(os.environ.get("CZ_PER_MIN", "20"))      # половина лимита ключа — вторая половина у live.py

STATE_FILE, SIGNALS_FILE, UNI_FILE = "l3_state.json", "l3_signals.csv", "l3_universe.json"


def log(*a):
    print(dt.datetime.now(KYIV).strftime("%H:%M:%S"), *a, flush=True)


def kyiv(ms, fmt="%d.%m, %H:%M"):
    return dt.datetime.fromtimestamp(ms / 1000, KYIV).strftime(fmt)


def tg(text):
    if TEST:
        text = "🧪 ТЕСТ\n" + text
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


# ---------------- Coinalyze (бережно, общий ключ) ----------------
_used = deque()


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
    for _ in range(8):
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
            log(f"Coinalyze просит подождать {wait:.0f} с")
            time.sleep(wait + 2)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Coinalyze {path} -> {r.status_code}: {r.text[:150]}")
        return r.json()
    raise RuntimeError(f"Coinalyze {path}: не удалось получить данные")


def cz_history(endpoint, czsyms, interval, frm_s, to_s, **extra):
    out = {}
    czsyms = list(czsyms)
    for i in range(0, len(czsyms), 20):
        b = czsyms[i:i + 20]
        try:
            data = cz(f"/{endpoint}", cost=len(b), symbols=",".join(b), interval=interval,
                      **{"from": int(frm_s), "to": int(to_s)}, **extra)
        except RuntimeError as e:
            log(e)
            continue
        for item in data:
            out[item["symbol"]] = sorted(item.get("history") or [], key=lambda x: x["t"])
    return out


# ---------------- состояние ----------------
def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def save_json(path, obj):
    if TEST:
        return
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


COLS = ["время (Киев)", "событие", "монета", "цена", "тейк", "стоп", "итог %", "сверху %капы", "снизу %капы",
        "L/S все", "капа $M", "слоты заняты", "BTC ниже MA50", "спот"]


def journal(row):
    if TEST:
        return
    new = not os.path.exists(SIGNALS_FILE)
    with open(SIGNALS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


# ---------------- монеты и капа (раз в сутки) ----------------
def universe():
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    u = load_json(UNI_FILE, {})
    if u.get("date") == today and u.get("coins") and not TEST:
        return u["coins"]
    if TEST and u.get("coins"):
        return u["coins"]
    caps = LC.caps_with_fallback()
    coins = {}
    for m in cz("/future-markets"):
        s = m.get("symbol", "")
        if not s.endswith(".A") or not m.get("is_perpetual") or m.get("quote_asset") != "USDT":
            continue
        b = (m.get("symbol_on_exchange") or s.split("_")[0]).upper()
        if not b.endswith("USDT") or b in TRADFI or b in LC.STOCKS:
            continue
        cap = caps.get(Hst.base_asset(b), 0)
        if CAP_MIN <= cap <= CAP_MAX:
            coins[b] = {"cz": s, "cap": cap}
    log(f"монет с капой ${CAP_MIN / 1e6:g}M–${CAP_MAX / 1e9:g}B: {len(coins)}")
    with open(UNI_FILE, "w") as f:
        json.dump({"date": today, "coins": coins}, f, indent=0)
    return coins


# ---------------- данные: архив Binance + хвост из Coinalyze ----------------
def archive_ls_yesterday(syms, st):
    """Последнее значение L/S всех аккаунтов из дневного архива (вчера, если нет — позавчера). Кэш на сутки."""
    today = dt.datetime.now(dt.timezone.utc).date()
    cache = st.get("ls_cache", {})
    if cache.get("date") == today.isoformat() and set(syms) <= set(cache.get("ls", {})):
        return cache["ls"]

    def one(sym):
        for back in (1, 2):
            m = PL.metrics(sym, {today - dt.timedelta(days=back)})
            if m is not None and len(m) and m.glob.notna().any():
                return sym, float(m.glob.dropna().iloc[-1])
        return sym, None

    with ThreadPoolExecutor(8) as ex:
        ls = dict(ex.map(one, syms))
    st["ls_cache"] = {"date": today.isoformat(), "ls": ls}
    return ls


def load_coin(sym, czsym, cz_ohlcv, cz_oi, cz_ls):
    """1h-свечи и метрики: архив, а за часы, которых в архиве ещё нет, — Coinalyze."""
    k = Hst.load_klines(sym, "1h")
    if k is None or not len(k):
        k = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "quote_volume"])
    k = k[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    last = k.open_time.max() if len(k) else -1
    tail = [{"open_time": b["t"] * 1000.0, "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"],
             "volume": b.get("v", 0) or 0, "quote_volume": (b.get("v", 0) or 0) * (b["h"] + b["l"] + b["c"]) / 3}
            for b in cz_ohlcv.get(czsym, []) if b["t"] * 1000 > last]
    if tail:
        k = pd.concat([k, pd.DataFrame(tail)], ignore_index=True)
    now_ms = time.time() * 1000
    k = k[k.open_time + H <= now_ms].sort_values("open_time").reset_index(drop=True)     # только закрытые свечи

    start = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=DAYS_BACK)
    days = {start + dt.timedelta(days=i) for i in range(DAYS_BACK)}
    met = PL.metrics(sym, days)
    if met is None:
        met = pd.DataFrame(columns=["t"] + PL.MCOLS)
    mlast = met.t.max() if len(met) else -1
    # значение бара Coinalyze t — это конец часа (совпадает по знаку изменений с архивом в 99% часов)
    oi = {b["t"]: b["c"] for b in cz_oi.get(czsym, [])}
    ls = {b["t"]: b.get("r") for b in cz_ls.get(czsym, [])}
    rows = [{"t": (t + 3300) * 1000.0, "oi": oi.get(t, np.nan), "glob": ls.get(t, np.nan)}
            for t in sorted(set(oi) | set(ls)) if (t + 3300) * 1000 > mlast]
    if rows:
        met = pd.concat([met, pd.DataFrame(rows)], ignore_index=True)
    met = met.sort_values("t").reset_index(drop=True)
    return k, met


def condition_series(k, met, cap, hours=28):
    """Карта «14д» по той же модели, что в тесте (liq_cap.coin): по каждому из последних часов —
    шорты в пределах +10% и лонги в пределах −10% в % капы, L/S всех аккаунтов, цена."""
    if len(k) < 24 * 20 or met is None or len(met) < 100:
        return []
    o, h, l, c = (k[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    v, qv, t = k.volume.to_numpy(float), k.quote_volume.to_numpy(float), k.open_time.to_numpy(float)
    n, c_now = len(k), c[-1]
    OI, LSA = PL.AsOf(met, "oi"), PL.AsOf(met, "glob")
    lo_g = math.log(np.nanmin(l[l > 0]) * 0.3)
    G = int((math.log(np.nanmax(h) * 3) - lo_g) / LM.STEP) + 2
    longs, shorts = np.zeros(G), np.zeros(G)
    dec = 0.5 ** (1 / (HL_DAYS * 24))
    lvl = np.exp(lo_g + (np.arange(G) + 0.5) * LM.STEP)
    gi = lambda p: min(G - 1, max(0, int((math.log(p) - lo_g) / LM.STEP)))
    prev, out = np.nan, []
    for i in range(n):
        if not (h[i] > 0 and l[i] > 0):
            continue
        oi = OI.at(t[i] + H, 2 * H)
        ih, il = gi(h[i]), gi(l[i])
        add = (oi - prev) if (np.isfinite(oi) and np.isfinite(prev) and oi > prev and v[i] > 0) else 0.0
        p = qv[i] / v[i] if v[i] > 0 else c[i]
        longs *= dec
        shorts *= dec
        shorts[:ih + 1] = 0
        longs[il:] = 0
        if add > 0:
            for L, wgt in LM.LEV.items():
                longs[gi(p * (1 - 1 / L + LM.MMR))] += add * wgt
                shorts[gi(p * (1 + 1 / L - LM.MMR))] += add * wgt
        if np.isfinite(oi) and oi > 0:
            for arr in (longs, shorts):
                s_ = arr.sum()
                if s_ > oi:
                    arr *= oi / s_
        if np.isfinite(oi):
            prev = oi
        if i < n - hours:
            continue
        px = c[i]
        mcap = cap * px / c_now
        ip = gi(px)
        up = (shorts * lvl)[ip + 1:gi(px * 1.10) + 1].sum() / mcap * 100
        dn = (longs * lvl)[gi(px * 0.90):ip].sum() / mcap * 100
        out.append({"t": t[i] + H, "px": px, "up": up, "dn": dn, "ls": LSA.at(t[i] + H, 2 * H), "mcap": mcap})
    return out


def is_on(r):
    return bool(np.isfinite(r["up"]) and np.isfinite(r["ls"]) and r["up"] >= UP_MIN and r["up"] > r["dn"]
                and r["ls"] < LS_MAX)


def btc_below_ma50():
    try:
        now = int(time.time())
        d = cz_history("ohlcv-history", ["BTCUSDT_PERP.A"], "daily", now - 70 * 86400, now).get("BTCUSDT_PERP.A", [])
        c = [b["c"] for b in d][:-1]                         # без текущего незакрытого дня
        if len(c) >= 50:
            return c[-1] < float(np.mean(c[-50:]))
    except Exception as e:  # noqa: BLE001
        log("BTC MA50:", e)
    return None


def has_spot(sym):
    try:
        b = Hst.base_asset(sym) + "USDT"
        r = Hst.get(Hst.S3, params={"prefix": f"data/spot/daily/klines/{b}/", "delimiter": "/", "max-keys": "1"})
        return r is not None and r.status_code == 200 and "<CommonPrefixes>" in r.text
    except Exception:  # noqa: BLE001
        return None


def fmt_price(x):
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    d = max(4, -int(math.floor(math.log10(x))) + 3)
    return f"{x:.{d}f}"


# ---------------- сопровождение открытых сделок ----------------
def follow(st, coins):
    opened = st.get("open", [])
    if not opened:
        return
    now = int(time.time())
    czs = [coins.get(p["sym"], {}).get("cz") or p.get("cz") for p in opened]
    bars = cz_history("ohlcv-history", [x for x in czs if x], "1hour", min(p["t"] for p in opened) / 1000 - 3600, now)
    keep, closed = [], []
    for p, czsym in zip(opened, czs):
        res, price = None, None
        for b in bars.get(czsym, []):
            bt = b["t"] * 1000
            if bt < p["t"] or bt + H > now * 1000:
                continue
            if b["l"] <= p["sl"]:
                res, price = "стоп", p["sl"]
            elif b["h"] >= p["tp"]:
                res, price = "тейк", p["tp"]
            if res:
                break
        if not res and now * 1000 >= p["t"] + HOLD_H * H:
            last = [b for b in bars.get(czsym, []) if b["t"] * 1000 + H <= now * 1000]
            if last:
                res, price = "время", last[-1]["c"]
        if not res:
            keep.append(p)
            continue
        closed.append((p, res, price))
    st["open"] = keep
    for p, res, price in closed:
        pnl = (price / p["px"] - 1) * 100 - 0.1
        icon = {"тейк": "✅", "стоп": "🛑", "время": "⏱"}[res]
        what = {"тейк": f"тейк +{TAKE:g}%", "стоп": f"стоп −{STOP:g}%", "время": "прошло 7 дней — закрыть по рынку"}[res]
        tg(f"{icon} L3 · {p['sym']} — {what}\n"
           f"Вход {kyiv(p['t'])} по {fmt_price(p['px'])} → {fmt_price(price)}: {pnl:+.1f}% (с комиссией)\n"
           f"Слот освободился. Открыто сделок: {len(keep)} из {SLOTS}")
        journal({"время (Киев)": kyiv(now * 1000), "событие": res, "монета": p["sym"], "цена": fmt_price(price),
                 "итог %": round(pnl, 2)})


# ---------------- главный проход ----------------
def main():
    st = load_json(STATE_FILE, {"open": [], "last_on": {}, "sent": []})
    st.setdefault("open", [])
    st.setdefault("last_on", {})
    st.setdefault("sent", [])
    coins = universe()
    follow(st, coins)

    ls_y = archive_ls_yesterday(sorted(coins), st)
    cand = [s for s, v in ls_y.items() if v is not None and v < PREFILTER_LS and s in coins]
    log(f"по вчерашнему архиву L/S < {PREFILTER_LS}: {len(cand)} монет")
    now = int(time.time())
    czc = [coins[s]["cz"] for s in cand]
    cz_ls = cz_history("long-short-ratio-history", czc, "1hour", now - 30 * 3600, now)
    fin = []
    for s in cand:
        h_ = [b for b in cz_ls.get(coins[s]["cz"], []) if b["t"] * 1000 + H <= now * 1000]
        if h_ and h_[-1].get("r") is not None and h_[-1]["r"] < LS_MAX:
            fin.append(s)
    log(f"сейчас L/S < {LS_MAX}: {len(fin)} монет")
    czf = [coins[s]["cz"] for s in fin]
    cz_ohlcv = cz_history("ohlcv-history", czf, "1hour", now - 3 * 86400, now)
    cz_oi = cz_history("open-interest-history", czf, "1hour", now - 3 * 86400, now, convert_to_usd="false")

    def work(s):
        try:
            k, met = load_coin(s, coins[s]["cz"], cz_ohlcv, cz_oi, cz_ls)
            return s, condition_series(k, met, coins[s]["cap"])
        except Exception as e:  # noqa: BLE001
            log(s, "ошибка:", repr(e))
            return s, []

    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(work, fin))

    busy = {p["sym"] for p in st["open"]}
    near, fired = [], []
    btc_low = None
    for s, ser in res.items():
        if not ser:
            continue
        last = ser[-1]
        near.append((s, last))
        # проверяем последние 4 закрытых часа: GitHub запускает по расписанию с опозданием до часа и иногда пропускает час
        for j in range(len(ser) - 4, len(ser)):
            if j < 24 or not is_on(ser[j]):
                continue
            if any(is_on(ser[x]) for x in range(j - 24, j)):
                continue
            key = f"{s}@{int(ser[j]['t'])}"
            if key in st["sent"]:
                continue
            fired.append((s, ser[j], len(ser) - 1 - j, key))
            break

    for s, r, late, key in sorted(fired, key=lambda x: -x[1]["up"]):
        if btc_low is None:
            btc_low = btc_below_ma50()
        spot = has_spot(s)
        px, tp, sl = r["px"], r["px"] * (1 + TAKE / 100), r["px"] * (1 - STOP / 100)
        taken = s not in busy and len(st["open"]) < SLOTS
        if s in busy:
            head, note = "⚪️", "По монете уже открыта сделка L3 — по правилам пропускаем."
        elif not taken:
            head, note = "⚪️", f"Все {SLOTS} слота заняты — по правилам пропускаем (для информации)."
        else:
            head, note = "🟢", f"Слот {len(st['open']) + 1} из {SLOTS}: на сделку ⅓ депозита."
        lines = [f"{head} L3 ЛОНГ · {s}" + (f" (С ОПОЗДАНИЕМ — сигнал {late} ч назад, проверь цену)" if late else ""),
                 f"Свеча закрылась: {kyiv(r['t'])} (Киев)",
                 f"Цена закрытия: {fmt_price(px)} — вход по рынку",
                 f"Тейк +{TAKE:g}%: {fmt_price(tp)}",
                 f"Стоп −{STOP:g}%: {fmt_price(sl)}",
                 f"Не сработало до {kyiv(r['t'] + HOLD_H * H)} — закрыть по рынку",
                 "(если вошла по другой цене — тейк и стоп считай от своей: +20% / −10%)",
                 "",
                 f"Шортовых ликвидаций до +10%: {r['up']:.1f}% капы, лонговых до −10%: {r['dn']:.1f}%",
                 f"Толпа (все аккаунты) L/S: {r['ls']:.2f} · капа ~${r['mcap'] / 1e6:.0f}M",
                 f"Спот на Binance: {'есть' if spot else 'нет' if spot is not None else '?'}"]
        if btc_low is not None:
            lines.append("BTC ниже MA50 — по истории сигнал сильнее" if btc_low else "BTC выше MA50")
        lines += ["", note]
        if tg("\n".join(lines)):
            st["sent"] = (st["sent"] + [key])[-500:]
            if taken:
                st["open"].append({"sym": s, "cz": coins[s]["cz"], "t": int(r["t"]), "px": px, "tp": tp, "sl": sl})
                busy.add(s)
            journal({"время (Киев)": kyiv(r["t"]), "событие": "сигнал" if taken else "пропуск", "монета": s,
                     "цена": fmt_price(px), "тейк": fmt_price(tp), "стоп": fmt_price(sl),
                     "сверху %капы": round(r["up"], 2), "снизу %капы": round(r["dn"], 2), "L/S все": round(r["ls"], 3),
                     "капа $M": round(r["mcap"] / 1e6, 1), "слоты заняты": len(st["open"]),
                     "BTC ниже MA50": btc_low, "спот": spot})

    log(f"сигналов: {len(fired)}; открыто сделок: {len(st['open'])}")
    if TEST:
        near.sort(key=lambda x: -x[1]["up"])
        txt = [f"L3 бот работает. Проверено монет: {len(coins)}, с толпой в шортах (L/S < {LS_MAX}) сейчас: {len(fin)}.",
               f"Новых сигналов: {len(fired)}. Открытых сделок по журналу: {len(st['open'])}.", "",
               "Ближе всего к сигналу (нужно сверху ≥3% капы и больше, чем снизу):"]
        for s, r in near[:8]:
            txt.append(f"• {s}: сверху {r['up']:.1f}%, снизу {r['dn']:.1f}%, L/S {r['ls']:.2f}")
        tg("\n".join(txt))
    st["last_run"] = kyiv(now * 1000, "%d.%m.%Y %H:%M")
    save_json(STATE_FILE, st)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        tg("⚠️ L3 бот: ошибка запуска\n" + traceback.format_exc()[-800:])
        sys.exit(1)
