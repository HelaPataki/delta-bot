"""
Сверка живого бота L3 с архивом Binance (на нём считался тест). Живой бот не меняется.

Каждый день — за вчерашние сутки (UTC) и все пропущенные дни, для которых архив уже выложен:
  те же правила L3 (карта ликвидаций «14д», сверху ≥3% капы и больше, чем снизу, L/S всех < 0,8, свежесть 24 ч),
  но только на данных архива Binance (бот в последние часы берёт Coinalyze) → сравнение с журналом l3_signals.csv.
Монеты — список бота (l3_universe.json). Сначала по дневным метрикам отбираются монеты, у которых в этот день
L/S всех опускался ниже 0,8 (без этого сигнала быть не может), и только по ним считается карта.
Итог — сообщение в Telegram и строки в l3_reconcile_log.csv.
"""

import csv
import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import live_l3 as L3          # те же функции и параметры, что у живого бота (импорт его не запускает)
import pump_life as PL

STATE, LOG = "l3_reconcile_state.json", "l3_reconcile_log.csv"
FIRST_DAY = "2026-11-13"                      # первый день, за который у бота есть журнал
H = 3600 * 1000
L3.TEST = False


def bot_keys(day):
    """Сигналы и пропуски бота, у которых свеча закрылась в UTC-сутки day: {(монета, 'дд.мм, чч:мм')}."""
    if not os.path.exists(L3.SIGNALS_FILE):
        return set()
    d0 = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    keys = {}
    with open(L3.SIGNALS_FILE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("событие") not in ("сигнал", "пропуск"):
                continue
            t = r["время (Киев)"]
            for year in (day.year, day.year - 1):
                try:
                    ms = dt.datetime.strptime(f"{year}.{t}", "%Y.%d.%m, %H:%M").replace(tzinfo=L3.KYIV).timestamp() * 1000
                except ValueError:
                    continue
                if d0 < ms <= d0 + 24 * H:
                    keys[(r["монета"], t)] = r
    return keys


def ls_dipped(sym, day):
    m = PL.metrics(sym, {day})
    if m is None or not len(m) or m.glob.isna().all():
        return sym, None
    return sym, bool((m.glob < L3.LS_MAX).any())


def archive_hits(sym, cap, day):
    """Свежие срабатывания L3 по архиву со свечами, закрывшимися в сутки day."""
    d0 = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    end = d0 + 24 * H
    k, met = L3.archive(sym)
    c_last = float(k.close.iloc[-1]) if len(k) else np.nan
    k = k[k.open_time + H <= end].reset_index(drop=True)
    met = met[met.t <= end].reset_index(drop=True)
    if not len(k) or k.open_time.max() + H < end:
        return None                                   # свечей за этот день в архиве ещё нет
    # капа в списке бота — по сегодняшней цене; на день сверки пересчитываем по цене конца того дня
    cap_d = cap * float(k.close.iloc[-1]) / c_last if c_last > 0 else cap
    ser = L3.condition_series(k, met, cap_d, hours=48)
    by_t = {L3.kyiv(r["t"]): r for r in ser}
    out = []
    for j in range(24, len(ser)):
        r = ser[j]
        if not (d0 < r["t"] <= end) or not L3.is_on(r):
            continue
        if any(L3.is_on(ser[x]) for x in range(j - 24, j)):
            continue
        out.append((sym, L3.kyiv(r["t"]), r))
    return out, by_t


def check_day(day, coins):
    with ThreadPoolExecutor(8) as ex:
        dip = dict(ex.map(lambda s: ls_dipped(s, day), coins))
    nodata = [s for s, v in dip.items() if v is None]
    if len(nodata) > 0.05 * len(coins):
        print(f"{day}: метрик в архиве нет по {len(nodata)} монетам — проверим позже", flush=True)
        return None
    b_rows = bot_keys(day)
    b_keys = set(b_rows)
    cand = sorted({s for s, v in dip.items() if v} | ({m for m, _ in b_keys} & set(coins)))
    print(f"{day}: L/S < {L3.LS_MAX} хоть раз за день — {len(cand)} монет", flush=True)

    def work(s):
        try:
            return archive_hits(s, coins[s]["cap"], day)
        except Exception as e:  # noqa: BLE001
            print(s, "ошибка:", repr(e), flush=True)
            return [], {}
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(work, cand))
    if any(r is None for r in res):
        print(f"{day}: свечей за день в архиве ещё нет — проверим позже", flush=True)
        return None
    arch = {(s, t): r for x in res if x for s, t, r in x[0]}
    series = {s: x[1] for s, x in zip(cand, res) if x}
    a_keys = set(arch)
    ok = a_keys == b_keys
    lines = [f"🔎 Сверка L3 за {day:%d.%m} (архив Binance)",
             f"🟢 Бот {len(b_keys)} · архив {len(a_keys)} {'✅' if ok else '⚠️'}"]
    for m, t in sorted(a_keys - b_keys):
        r = arch[(m, t)]
        lines.append(f"   пропустил: {m} {t[7:]} (сверху {r['up']:.1f}% капы, L/S {r['ls']:.2f})")
    def why(m, t):
        r, b = series.get(m, {}).get(t), b_rows.get((m, t), {})
        txt = f"\n      бот: сверху {b.get('сверху %капы', '?')}%, снизу {b.get('снизу %капы', '?')}%, L/S {b.get('L/S все', '?')}"
        if r:
            txt += f"\n      архив: сверху {r['up']:.2f}%, снизу {r['dn']:.2f}%, L/S {r['ls']:.3f}"
        return txt
    for m, t in sorted(b_keys - a_keys):
        lines.append(f"   лишний: {m} {t[7:]}{why(m, t)}")
    rows = [{"дата": str(day), "монета": m, "свеча (Киев)": t,
             "статус": "совпал" if (m, t) in a_keys & b_keys else ("пропущен ботом" if (m, t) in a_keys else "лишний у бота"),
             "сверху %капы (архив)": round(series[m][t]["up"], 2) if series.get(m, {}).get(t) else "",
             "L/S (архив)": round(series.get(m, {}).get(t, {}).get("ls", np.nan), 3) if series.get(m, {}).get(t) else ""}
            for m, t in sorted(a_keys | b_keys)]
    return "\n".join(lines), rows


def write_log(rows):
    if not rows:
        return
    cols = ["дата", "монета", "свеча (Киев)", "статус", "сверху %капы (архив)", "L/S (архив)"]
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"last_day": str(dt.date.fromisoformat(FIRST_DAY) - dt.timedelta(days=1))}
    coins = L3.load_json(L3.UNI_FILE, {}).get("coins", {})
    if not coins:
        print("нет списка монет бота (l3_universe.json)")
        return
    print("монет бота:", len(coins), flush=True)
    today = dt.datetime.now(dt.timezone.utc).date()
    day = dt.date.fromisoformat(st["last_day"]) + dt.timedelta(days=1)
    texts = []
    while day < today:
        res = check_day(day, coins)
        if res is None:
            if (today - day).days >= 3:
                texts.append(f"⏳ Сверка L3 за {day:%d.%m}: Binance ещё не выложил архив — жду дальше")
            break
        text, rows = res
        texts.append(text)
        write_log(rows)
        st["last_day"] = str(day)
        day += dt.timedelta(days=1)
    if texts:
        L3.tg("\n\n".join(texts))
    json.dump(st, open(STATE, "w"), indent=1)


if __name__ == "__main__":
    main()
