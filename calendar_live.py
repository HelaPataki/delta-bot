"""
Живой бот «Календарь» — шорты по листингам и разблокировкам (отдельный от live.py и live_l3.py,
свои файлы calendar_state.json / calendar_signals.csv). Правила — как в истории (cycle_grid / cycle_trades):

ЛИСТИНГ  — шорт на 30-й день после запуска фьючерса на Binance (время листинга = первая часовая свеча в архиве
           Binance, как в тесте). Вход в тот же час, но если он попадает на ночь (23:00–07:00 по Киеву) — в 07:00
           по Киеву того же дня. Только если оборот за 24 ч перед входом ≥ $50M. Стоп +30%, тейк −15%;
           после −8% стоп переносится на вход; закрыть по рынку через 14 дней.
РАЗБЛОКИРОВКА — инсайдеры (команда, инвесторы, советники, private/seed…) разблокируют за день ≥ 3% от уже
           разблокированного (данные DefiLlama). Шорт в 07:00 по Киеву за 5 дней до даты, стоп +15%, тейк −10%,
           закрыть по рынку в 07:00 через день после даты.
Плечо x2, на сделку — один из 4 общих слотов.

Сообщения: в 20:00 — что будет в ближайшие сутки (листинги — с текущим оборотом, разблокировки — «завтра в 07:00»)
и предупреждения о разблокировках за 7–8 дней (сверить дату на Tokenomist); за 5 минут до входа в листинг —
напоминание; во время входа — сигнал (или «отмена», если оборот мал); дальше — «перенеси стоп на вход»,
«тейк / стоп сработал», «закрой по времени». Если DefiLlama сдвинула дату разблокировки — отдельное сообщение.
OBSERVE=1 — режим наблюдения: всё то же, но с пометкой «НАБЛЮДЕНИЕ — не торговать».
MODE=test — обновить календарь и прислать список ближайших входов (пометка ТЕСТ), состояние не сохраняется.
"""

import csv
import datetime as dt
import io
import json
import os
import re
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests

import live as L          # Telegram, Coinalyze с учётом лимита, общие мелочи
from tradfi import TRADFI

# ---------------- правила ----------------
LEV = 2
LIST_DAYS, LIST_TURN, LIST_STOP, LIST_TAKE, LIST_BE, LIST_HOLD_D = 30, 50e6, 30.0, 15.0, 8.0, 14
NIGHT_FROM, NIGHT_TO, MORNING = 23, 7, 7          # ночь по Киеву: [23:00, 07:00) → вход в 07:00
UNL_MIN, UNL_DEDUP_MIN, UNL_DEDUP_D = 3.0, 1.0, 14
UNL_BEFORE, UNL_AFTER, UNL_STOP, UNL_TAKE, UNL_HOUR = 5, 1, 15.0, 10.0, 7
HEADSUP_D = 8                                       # предупреждать о разблокировке, когда до неё ≤ 8 дней
DIGEST_HOUR = 20
OLD_DAYS = 45                                       # листинги старше — не интересны

# ---------------- техническое ----------------
KYIV = ZoneInfo("Europe/Kyiv")
H, DAY = 3600, 86400
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE = "https://data.binance.vision"
LLAMA = "https://defillama-datasets.llama.fi"
STATE_FILE, JOURNAL_FILE = "calendar_state.json", "calendar_signals.csv"
INSIDER = re.compile(r"team|investor|advis|private|seed|founder|contributor|backer|strategic|core|insider|employee|vc|series",
                     re.I)
MODE = os.environ.get("MODE", "live")
OBSERVE = os.environ.get("OBSERVE", "1") == "1"
log, kyiv = L.log, L.kyiv


def head():
    if MODE == "test":
        return "🧪 ТЕСТ — НЕ ТОРГОВАТЬ 🧪\n"
    return "👀 НАБЛЮДЕНИЕ — сделку не открывать 👀\n" if OBSERVE else ""


def tg(text):
    return L.tg(head() + text)


def kyiv_at(day, hour):
    """Время hour:00 по Киеву в дату day (date) → unix-секунды."""
    return int(dt.datetime(day.year, day.month, day.day, hour, tzinfo=KYIV).timestamp())


def http(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}, **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 + 5 * k)
    return None


# ---------------- монеты: Binance-перпетуалы в Coinalyze ----------------
def markets():
    """{символ Binance: символ Coinalyze} для USDT-перпетуалов Binance."""
    out = {}
    for m in L.cz("/future-markets"):
        s = m.get("symbol", "")
        if not s.endswith(".A") or not m.get("is_perpetual") or m.get("quote_asset") != "USDT":
            continue
        b = (m.get("symbol_on_exchange") or s.split("_")[0]).upper()
        if b.endswith("USDT"):
            out[b] = s
    return out


def bars_1h(cs, frm, to):
    return L.history("ohlcv-history", [cs], "1hour", frm, to).get(cs, [])


def turnover_24h(cs, at):
    bars = [b for b in bars_1h(cs, at - DAY - H, at) if at - DAY <= b["t"] < at]
    return sum((b["o"] + b["h"] + b["l"] + b["c"]) / 4 * (b.get("v") or 0) for b in bars), len(bars)


def looks_tradfi(sym, cs, at):
    """Акции/фонды/сырьё: список TRADFI или объём в выходные в 4+ раза ниже, чем в будни (у крипты — в 1–2,5)."""
    if sym in TRADFI:
        return True
    if not cs:
        return False
    bars = L.history("ohlcv-history", [cs], "daily", at - 29 * DAY, at).get(cs, [])
    wd, we = [], []
    for b in bars:
        qv = (b["o"] + b["h"] + b["l"] + b["c"]) / 4 * (b.get("v") or 0)
        (we if dt.datetime.fromtimestamp(b["t"], dt.timezone.utc).weekday() >= 5 else wd).append(qv)
    if len(wd) < 8 or len(we) < 3:
        return False
    a, b = sum(wd) / len(wd), sum(we) / len(we)
    return b <= 0 or a / b >= 4


def price_now(cs):
    now = int(time.time())
    b = L.history("ohlcv-history", [cs], "1min", now - 600, now).get(cs, [])
    return b[-1]["c"] if b else None


# ---------------- листинги: время первой часовой свечи в архиве Binance ----------------
def first_candle(sym):
    """Время открытия первой часовой свечи фьючерса (unix-сек) или None. По архиву Binance, как в тесте."""
    prefix = f"data/futures/um/daily/klines/{sym}/1h/"
    r = http(S3, params={"prefix": prefix, "max-keys": "5"})
    if r is None:
        return None
    root = ElementTree.fromstring(r.content)
    ns = {"s": root.tag.split("}")[0].strip("{")}
    keys = [k.text for k in root.findall("s:Contents/s:Key", ns) if k.text.endswith(".zip")]
    if not keys:
        return None
    z = http(f"{ARCHIVE}/{keys[0]}")
    if z is None:
        return None
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        rows = list(csv.reader(io.TextIOWrapper(zf.open(zf.namelist()[0]))))
    times = [int(r[0]) for r in rows if r and r[0].isdigit()]
    if not times:
        return None
    t = min(times)
    return t // 1000000 if t > 1e14 else t // 1000


def listing_entry(listed):
    """Время входа: листинг + 30 дней; если ночь по Киеву — 07:00 того же дня (как «смешанный» вариант в тесте)."""
    base = listed + LIST_DAYS * DAY
    local = dt.datetime.fromtimestamp(base, KYIV)
    if local.hour >= NIGHT_FROM or local.hour < NIGHT_TO:
        return kyiv_at(local.date(), MORNING), True
    return base, False


def update_listings(st, mk):
    reg = st.setdefault("registry", {})
    todo = [s for s in mk if s not in reg]
    log(f"новых монет для проверки листинга: {len(todo)}")

    def one(s):
        try:
            return s, first_candle(s)
        except Exception as e:  # noqa: BLE001
            log("листинг", s, e)
            return s, "ошибка"

    with ThreadPoolExecutor(6) as ex:
        for s, t in ex.map(one, todo):
            if t == "ошибка":
                continue
            reg[s] = t          # None — в архиве ещё нет (появится на следующий день), проверим снова
    for s in [s for s, t in reg.items() if t is None]:
        del reg[s]              # перепроверить в следующий раз
    now = time.time()
    added = 0
    for s, t in reg.items():
        if not t or now - t > OLD_DAYS * DAY:
            continue
        if s in TRADFI:
            continue
        entry, night = listing_entry(t)
        pid = f"L:{s}:{t}"
        if entry < now - 2 * H or any(p["id"] == pid for p in st["planned"]) or pid in st["done"]:
            continue
        st["planned"].append({"id": pid, "kind": "листинг", "sym": s, "entry": entry, "listed": t, "night": night})
        added += 1
    log(f"листингов в плане: +{added}")


# ---------------- разблокировки (DefiLlama) ----------------
def gecko_symbols():
    for _ in range(6):
        r = http("https://api.coingecko.com/api/v3/coins/list", tries=1)
        if r is not None:
            return {x["id"]: x["symbol"] for x in r.json()}
        time.sleep(20)
    return {}


def unlock_events(doc):
    """[(дата, размер %)] — дни, когда инсайдеры разблокируют ≥ 3% (после фильтра повторов, как в тесте)."""
    cats = None
    for path in (("documentedData", "data"), ("data",)):
        x = doc
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if isinstance(x, list) and x and isinstance(x[0], dict) and "data" in x[0]:
            cats = x
            break
    if not cats:
        return []
    tot, ins = {}, {}
    for cat in cats:
        is_ins = bool(INSIDER.search(str(cat.get("label", ""))))
        for p in cat.get("data", []):
            if not isinstance(p, dict) or "timestamp" not in p:
                continue
            d = dt.datetime.fromtimestamp(int(p["timestamp"]), dt.timezone.utc).date()
            u = float(p.get("unlocked") or 0)
            tot[d] = tot.get(d, 0) + u
            if is_ins:
                ins[d] = ins.get(d, 0) + u
    days = sorted(tot)
    out, last = [], None
    for a, b in zip(days, days[1:]):
        if tot[a] <= 0:
            continue
        size = (ins.get(b, 0) - ins.get(a, 0)) / tot[a] * 100
        if size < UNL_DEDUP_MIN or b < dt.date(2021, 1, 1):
            continue
        if last is not None and (b - last).days < UNL_DEDUP_D:
            continue
        last = b
        if size >= UNL_MIN:
            out.append((b, size))
    return out


def update_unlocks(st, mk):
    r = http(f"{LLAMA}/emissionsProtocolsList")
    names = r.json() if r is not None else []
    gk = gecko_symbols()
    log(f"DefiLlama: протоколов {len(names)}, CoinGecko монет {len(gk)}")
    today = dt.datetime.now(KYIV).date()

    def one(n):
        r = http(f"{LLAMA}/emissions/{n}", tries=3)
        return n, (r.json() if r is not None else None)

    found, fetched = {}, set()
    with ThreadPoolExecutor(6) as ex:
        for n, doc in ex.map(one, names):
            if not doc:
                continue
            fetched.add(n)
            tk = str(gk.get(str(doc.get("gecko_id")), "") or (doc.get("metadata") or {}).get("ticker") or "").upper()
            sym = next((s for s in (f"{tk}USDT", f"1000{tk}USDT") if tk and s in mk), None)
            if not sym:
                continue
            for d, size in unlock_events(doc):
                if today - dt.timedelta(days=2) <= d <= today + dt.timedelta(days=60):
                    found.setdefault(sym, []).append((d, size, n))
    log(f"разблокировок впереди (≥{UNL_MIN:g}%, 60 дней): {sum(len(v) for v in found.values())}")
    if not found:
        log("разблокировки: данных нет — план не меняю")
        return
    # сдвиги дат: событие по монете из плана пропало, а рядом появилось другое
    now = time.time()
    moved = []
    for p in list(st["planned"]):
        if p["kind"] != "разблокировка" or p["entry"] < now or p.get("protocol") not in fetched:
            continue
        d_old = dt.date.fromisoformat(p["date"])
        new = [x for x in found.get(p["sym"], [])]
        if any(x[0] == d_old for x in new):
            continue
        near = [x for x in new if abs((x[0] - d_old).days) <= 10]
        st["planned"].remove(p)
        if near:
            moved.append((p["sym"], d_old, near[0][0]))
        else:
            moved.append((p["sym"], d_old, None))
    for sym, items in found.items():
        for d, size, proto in items:
            pid = f"U:{sym}:{d.isoformat()}"
            entry = kyiv_at(d - dt.timedelta(days=UNL_BEFORE), UNL_HOUR)
            if pid in st["done"] or any(p["id"] == pid for p in st["planned"]) or entry < now - 2 * H:
                continue
            st["planned"].append({"id": pid, "kind": "разблокировка", "sym": sym, "entry": entry,
                                  "date": d.isoformat(), "size": round(size, 1), "protocol": proto})
    for sym, d_old, d_new in moved:
        if d_new:
            tg(f"📅 {sym}: разблокировка сдвинулась {d_old:%d.%m} → {d_new:%d.%m}\nВход и выход пересчитаны.")
        else:
            tg(f"📅 {sym}: разблокировка {d_old:%d.%m} пропала из графика DefiLlama\nВход отменён.")


# ---------------- тексты ----------------
def roi(x):
    return f"{x * LEV:.0f} %"


def entry_text(p, price, late=""):
    if p["kind"] == "листинг":
        stop, take = LIST_STOP, LIST_TAKE
        why = f"Листинг {kyiv(p['listed'], '%d.%m %H:%M')} → 30-й день"
        hold = f"4️⃣ После −8% (ROI +{LIST_BE * LEV:.0f}%) — стоп на вход (бот напомнит)\n" \
               f"5️⃣ Закрыть до: {kyiv(p['entry'] + LIST_HOLD_D * DAY)}"
    else:
        stop, take = UNL_STOP, UNL_TAKE
        why = f"Разблокировка инсайдеров {dt.date.fromisoformat(p['date']):%d.%m}: {p['size']:.1f}% предложения"
        hold = f"4️⃣ Закрыть до: {kyiv(p['exit'])}"
    lines = [f"🔴🔴🔴 ШОРТ 🔴🔴🔴 {p['kind'].upper()}", p["sym"]] + ([late] if late else []) + [
        why, "━━━━━━━━━━━━━━", "1️⃣ Рыночный → Продать/Шорт 🔴",
        f"2️⃣ Стоп-лосс ROI:   {roi(stop)}", f"3️⃣ Тейк-профит ROI: {roi(take)}", hold, "━━━━━━━━━━━━━━",
        f"Цена сейчас: {price:g}" if price else "Цена: нет данных",
        f"Плечо: {LEV}x | Один свободный слот (¼ депозита)", "Все 4 слота заняты — пропустить"]
    return "\n".join(lines)


# ---------------- журнал ----------------
COLS = ["время (Киев)", "событие", "вид", "монета", "цена", "результат % цены", "подробности"]


def journal(event, p, price="", res="", note=""):
    new = not os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        w.writerow([kyiv(time.time(), "%d.%m.%Y %H:%M"), event, p["kind"], p["sym"], price,
                    "" if res == "" else round(res, 2), note])


# ---------------- входы, выходы, сопровождение ----------------
def do_entry(st, p, mk, now):
    cs = mk.get(p["sym"])
    late = ""
    if now - p["entry"] > 15 * 60:
        late = f"⚠️ С ОПОЗДАНИЕМ: вход был в {kyiv(p['entry'], '%H:%M')}"
    if p["kind"] == "листинг" and looks_tradfi(p["sym"], cs, int(now)):
        tg(f"⚪️ {p['sym']}: это не крипта (акция / фонд / сырьё) — шорт листинга не берём")
        journal("отмена", p, note="TradFi")
        return
    if p["kind"] == "листинг":
        turn, n = turnover_24h(cs, p["entry"]) if cs else (0, 0)
        if n < 20 or turn < LIST_TURN:
            tg(f"⚪️ {p['sym']}: шорт листинга ОТМЕНА\nОборот за 24 ч ${turn / 1e6:.1f}M (нужно от ${LIST_TURN / 1e6:.0f}M)"
               + ("" if n >= 20 else f"\nДанных мало: {n} ч из 24"))
            journal("отмена", p, note=f"оборот ${turn / 1e6:.1f}M")
            return
        p["turnover"] = turn
    else:
        p["exit"] = kyiv_at(dt.date.fromisoformat(p["date"]) + dt.timedelta(days=UNL_AFTER), UNL_HOUR)
    price = price_now(cs) if cs else None
    tg(entry_text(p, price, late))
    if p["kind"] == "листинг":
        p["exit"] = p["entry"] + LIST_HOLD_D * DAY
    p.update({"price": price, "opened": now, "be": False, "be_sent": False})
    st["open"].append(p)
    journal("вход", p, price or "", note=late or (f"оборот ${p.get('turnover', 0) / 1e6:.0f}M" if p["kind"] == "листинг"
                                                   else f"{p['size']}% {p['date']}"))


def close(st, p, why, res, price=None):
    st["open"] = [x for x in st["open"] if x["id"] != p["id"]]
    journal(why, p, price or "", res)


def monitor(st, mk, now):
    """По часовым свечам с момента входа: тейк / стоп / безубыток. Правило как в тесте: стоп и тейк в одной свече — стоп."""
    for p in list(st["open"]):
        cs, e = mk.get(p["sym"]), p.get("price")
        if not cs or not e:
            continue
        stop_pct = LIST_STOP if p["kind"] == "листинг" else UNL_STOP
        take_pct = LIST_TAKE if p["kind"] == "листинг" else UNL_TAKE
        done_to = p.get("checked_to", 0)
        bars = [b for b in bars_1h(cs, p["opened"] - H, now)
                if b["t"] + H > p["opened"] and b["t"] > done_to and b["t"] + H <= now]
        for b in bars:
            p["checked_to"] = b["t"]
            stop_lvl = e if p["be"] else e * (1 + stop_pct / 100)
            if b["h"] >= stop_lvl:
                if p["be"]:
                    tg(f"⚪️ {p['sym']}: цена вернулась к входу — сработал стоп в безубытке (≈0%)\nСлот свободен.")
                    close(st, p, "безубыток", 0.0, stop_lvl)
                else:
                    tg(f"🛑 {p['sym']}: сработал стоп (ROI −{stop_pct * LEV:.0f}%)\nПроверь, что позиция закрыта. Слот свободен.")
                    close(st, p, "стоп", -stop_pct, stop_lvl)
                break
            if b["l"] <= e * (1 - take_pct / 100):
                tg(f"🎯 {p['sym']}: тейк взят (ROI +{take_pct * LEV:.0f}%)\nПроверь, что позиция закрыта. Слот свободен.")
                close(st, p, "тейк", take_pct, e * (1 - take_pct / 100))
                break
            if p["kind"] == "листинг" and not p["be"] and b["l"] <= e * (1 - LIST_BE / 100):
                p["be"] = True
                if not p["be_sent"]:
                    tg(f"🔁 {p['sym']}: цена ушла на −{LIST_BE:g}% (ROI +{LIST_BE * LEV:.0f}%)\n"
                       f"👉 Перенеси стоп-лосс на цену входа {e:g}\nТейк оставь на месте.")
                    p["be_sent"] = True
                    journal("стоп на вход", p, e)


def do_exit(st, p):
    cs = st.get("_mk", {}).get(p["sym"])
    price = price_now(cs) if cs else None
    res = (1 - price / p["price"]) * 100 if price and p.get("price") else ""
    tg(f"⏰🔴 {p['sym']}: время вышло ({'14 дней' if p['kind'] == 'листинг' else 'день после разблокировки'})\n"
       f"👉 Закрой ШОРТ по рынку, если ещё открыт" + (f"\nСейчас {res:+.1f}% цены (ROI {res * LEV:+.0f}%)"
                                                     if res != "" else ""))
    close(st, p, "время", res, price)


# ---------------- вечерняя сводка ----------------
def digest(st, mk, now):
    lines = []
    soon = sorted([p for p in st["planned"] if now < p["entry"] <= now + DAY], key=lambda p: p["entry"])
    for p in soon:
        when = kyiv(p["entry"], "%d.%m в %H:%M")
        if p["kind"] == "листинг":
            cs = mk.get(p["sym"])
            if looks_tradfi(p["sym"], cs, int(time.time())):
                continue
            turn, _ = turnover_24h(cs, int(time.time())) if cs else (0, 0)
            ok = "✅" if turn >= LIST_TURN else "⚠️ пока мало"
            note = " (листинг ночью → утром)" if p.get("night") else ""
            lines.append(f"🔴 {when} — шорт ЛИСТИНГА {p['sym']}{note}\n   оборот сейчас ${turn / 1e6:.0f}M {ok} "
                         f"(нужно от ${LIST_TURN / 1e6:.0f}M, проверю в момент входа)")
        else:
            lines.append(f"🔴 {when} — шорт РАЗБЛОКИРОВКИ {p['sym']}\n   разблокировка "
                         f"{dt.date.fromisoformat(p['date']):%d.%m}, {p['size']:.1f}% — сверь дату на Tokenomist")
    heads = [p for p in st["planned"] if p["kind"] == "разблокировка" and p["entry"] > now + DAY
             and dt.date.fromisoformat(p["date"]) <= dt.datetime.now(KYIV).date() + dt.timedelta(days=HEADSUP_D)
             and p["id"] not in st["headsup"]]
    for p in sorted(heads, key=lambda p: p["entry"]):
        lines.append(f"📅 {p['sym']}: разблокировка {dt.date.fromisoformat(p['date']):%d.%m} ({p['size']:.1f}%)\n"
                     f"   вход {kyiv(p['entry'], '%d.%m в %H:%M')} — сверь дату на Tokenomist")
        st["headsup"].append(p["id"])
    if lines:
        tg("🗓 КАЛЕНДАРЬ на ближайшие дни\n━━━━━━━━━━━━━━\n" + "\n".join(lines))
    log(f"сводка: {len(lines)} строк")


# ---------------- главный цикл ----------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except Exception:  # noqa: BLE001
        st = {}
    for k, v in (("registry", {}), ("planned", []), ("open", []), ("done", []), ("headsup", []), ("last_daily", ""),
                 ("last_digest", ""), ("reminded", [])):
        st.setdefault(k, v)
    return st


def save_state(st):
    st.pop("_mk", None)
    st["done"] = st["done"][-3000:]
    st["headsup"] = st["headsup"][-1000:]
    st["reminded"] = st["reminded"][-1000:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)


def main():
    st = load_state()
    mk = markets()
    st["_mk"] = mk
    now = time.time()
    today = dt.datetime.now(KYIV)
    log(f"монет Binance в Coinalyze: {len(mk)}")
    if MODE == "test" or (st["last_daily"] != today.date().isoformat() and today.hour >= DIGEST_HOUR - 2):
        update_listings(st, mk)
        update_unlocks(st, mk)
        st["last_daily"] = today.date().isoformat()
    if MODE == "test":
        up = sorted(st["planned"], key=lambda p: p["entry"])[:15]
        txt = "\n".join(f"{kyiv(p['entry'], '%d.%m %H:%M')} {p['kind']} {p['sym']}"
                        + (f" (разблок. {p['date']}, {p['size']}%)" if p["kind"] == "разблокировка" else "")
                        for p in up)
        tg(f"Бот «Календарь» работает ✅\nМонет: {len(mk)}, в плане входов: {len(st['planned'])}\n"
           f"Ближайшие:\n{txt or 'пока нет'}")
        return
    monitor(st, mk, now)
    # события ближайшего часа: напоминание за 5 мин (листинг), вход, выход по времени, сводка в 20:00
    end = now + 65 * 60
    tasks = []
    for p in st["planned"]:
        if p["entry"] - 300 >= now - 1800 and p["entry"] - 300 < end and p["kind"] == "листинг" \
                and p["id"] not in st["reminded"]:
            tasks.append((p["entry"] - 300, "напоминание", p))
        if p["entry"] < end:
            tasks.append((p["entry"], "вход", p))
    for p in st["open"]:
        if p.get("exit") and p["exit"] < end:
            tasks.append((p["exit"], "выход", p))
    d_ts = kyiv_at(today.date(), DIGEST_HOUR)
    if st["last_digest"] != today.date().isoformat() and now - 3 * H < d_ts < end:
        tasks.append((d_ts, "сводка", None))
    for ts, what, p in sorted(tasks, key=lambda x: x[0]):
        if ts > time.time():
            L.sleep_until(ts)
        now = time.time()
        if what == "напоминание":
            if now >= p["entry"]:
                continue
            tg(f"⏳ Через 5 минут шорт ЛИСТИНГА {p['sym']} ({kyiv(p['entry'], '%H:%M')})")
            st["reminded"].append(p["id"])
        elif what == "вход":
            st["planned"] = [x for x in st["planned"] if x["id"] != p["id"]]
            st["done"].append(p["id"])
            if now - p["entry"] > 3 * H:
                tg(f"⚠️ {p['sym']}: бот пропустил вход ({kyiv(p['entry'], '%d.%m %H:%M')}) — сделку не открываем")
                journal("пропущен", p)
            else:
                do_entry(st, p, mk, now)
        elif what == "выход":
            monitor(st, mk, now)
            if any(x["id"] == p["id"] for x in st["open"]):
                do_exit(st, p)
        elif what == "сводка":
            digest(st, mk, now)
            st["last_digest"] = today.date().isoformat()
        save_state(st)
        st["_mk"] = mk
    monitor(st, mk, time.time())
    save_state(st)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        L.tg("⚠️ Бот «Календарь»: ошибка\n" + err[-600:])
        sys.exit(1)
