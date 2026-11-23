"""
Диагностика расхождений живого бота L3 со сверкой (архив Binance). Живой бот не меняется.

По каждому случаю (монета + час закрытия свечи по Киеву) считаем условие L3 тремя способами:
  A  «архив»   — только архив Binance (так считает сверка и так считался тест);
  B  «как бот» — архив до начала суток сигнала (UTC) + часы этих суток из Coinalyze (так видит бот);
  C  «бот, L/S из архива» — как B, но L/S всех аккаунтов взят из архива (отделяет вклад L/S от вклада OI и свечей).
И рядом сырые значения за те же часы: цена закрытия, OI, L/S — архив против Coinalyze.
Для сигналов, которые бот пропустил, — чем закончилась бы сделка (тейк +20 / стоп −10 / 7 дней, по 1h-свечам архива).
Плюс почему бот мог пропустить: монета в его списке? L/S в архиве на конец прошлых суток (предфильтр < 1,0).
"""

import datetime as dt
import os

import numpy as np
import pandas as pd

import live_l3 as L3
import pump_life as PL

OUT = "results/l3_diag"
H = 3600 * 1000
KY = L3.KYIV

CASES = [  # (монета, свеча закрылась по Киеву, что показала сверка)
    ("BMTUSDT", "2026-11-13 17:00", "лишний"),
    ("INITUSDT", "2026-11-15 09:00", "лишний"),
    ("EVAAUSDT", "2026-11-15 11:00", "пропущен"),
    ("MEGAUSDT", "2026-11-15 20:00", "пропущен"),
    ("HUMAUSDT", "2026-11-16 17:00", "совпал (контроль)"),
    ("BLESSUSDT", "2026-11-17 03:00", "пропущен"),
    ("MEGAUSDT", "2026-11-18 07:00", "пропущен (бот на час позже)"),
    ("MEGAUSDT", "2026-11-18 08:00", "лишний (бот на час позже)"),
    ("MMTUSDT", "2026-11-18 18:00", "пропущен"),
    ("INITUSDT", "2026-11-19 11:00", "лишний"),
    ("RAREUSDT", "2026-11-19 23:00", "лишний"),
    ("GTCUSDT", "2026-11-20 08:00", "совпал (контроль)"),
    ("BMTUSDT", "2026-11-20 08:00", "пропущен (бот на час позже)"),
    ("BMTUSDT", "2026-11-20 09:00", "лишний (бот на час позже)"),
    ("EVAAUSDT", "2026-11-20 15:00", "лишний"),
]
if os.environ.get("CASES"):   # «МОНЕТА|ГГГГ-ММ-ДД ЧЧ:ММ|пометка; ...» — свои случаи из запуска вручную
    CASES = [tuple(x.strip() for x in c.split("|")) + ("",) * (3 - len(c.split("|")))
             for c in os.environ["CASES"].split(";") if c.strip()]
    CASES = [c[:3] for c in CASES]
BEFORE, AFTER = 4, 1           # сколько часов показать до и после свечи сигнала


def ms(kyiv_str):
    return dt.datetime.strptime(kyiv_str, "%Y-%m-%d %H:%M").replace(tzinfo=KY).timestamp() * 1000


def cz_data(czsym, frm_ms, to_ms):
    a, b = int(frm_ms / 1000), int(to_ms / 1000)
    oh = L3.cz_history("ohlcv-history", [czsym], "1hour", a, b)
    oi = L3.cz_history("open-interest-history", [czsym], "1hour", a, b, convert_to_usd="false")
    ls = L3.cz_history("long-short-ratio-history", [czsym], "1hour", a, b)
    return oh, oi, ls


def bot_like(k_full, met_full, czsym, d0, end, oh, oi, ls, ls_from_archive):
    """Данные так, как их видит бот: архив до d0, дальше Coinalyze (та же склейка, что в live_l3.load_coin)."""
    k = k_full[k_full.open_time < d0].copy()
    met = met_full[met_full.t < d0].copy()
    last = k.open_time.max() if len(k) else -1
    tail = [{"open_time": b_["t"] * 1000.0, "open": b_["o"], "high": b_["h"], "low": b_["l"], "close": b_["c"],
             "volume": b_.get("v", 0) or 0, "quote_volume": (b_.get("v", 0) or 0) * (b_["h"] + b_["l"] + b_["c"]) / 3}
            for b_ in oh.get(czsym, []) if b_["t"] * 1000 > last and b_["t"] * 1000 + H <= end]
    if tail:
        k = pd.concat([k, pd.DataFrame(tail)], ignore_index=True)
    k = k.sort_values("open_time").reset_index(drop=True)
    mlast = met.t.max() if len(met) else -1
    # как сейчас в боте (с 21.11): OI бара t — на конец часа, L/S бара t — на начало часа
    rows = [{"t": (b_["t"] + 3300) * 1000.0, "oi": b_["c"], "glob": np.nan}
            for b_ in oi.get(czsym, []) if mlast < (b_["t"] + 3300) * 1000 <= end]
    rows += [{"t": b_["t"] * 1000.0, "oi": np.nan, "glob": b_.get("r") if b_.get("r") is not None else np.nan}
             for b_ in ls.get(czsym, []) if mlast < b_["t"] * 1000 <= end]
    if rows:
        met = pd.concat([met, pd.DataFrame(rows)], ignore_index=True)
    met = met.sort_values("t").reset_index(drop=True)
    if ls_from_archive:
        arch = PL.AsOf(met_full, "glob")
        met["glob"] = [arch.at(t, 2 * H) for t in met.t]
    return k, met


def series(k, met, cap, end):
    k = k[k.open_time + H <= end].reset_index(drop=True)
    met = met[met.t <= end].reset_index(drop=True)
    return {int(r["t"]): r for r in L3.condition_series(k, met, cap, hours=BEFORE + AFTER + 26)}, k


def fresh(ser, t):
    """Свежий сигнал на часе t: условие верно сейчас и не было верно 24 ч до этого."""
    keys = sorted(ser)
    if t not in ser or not L3.is_on(ser[t]):
        return False
    i = keys.index(t)
    return not any(L3.is_on(ser[x]) for x in keys[max(0, i - 24):i])


def outcome(k_full, t_close):
    """Сделка по архиву: вход по закрытию сигнальной свечи, тейк +20 / стоп −10 / 7 дней."""
    k = k_full[k_full.open_time + H > t_close - 1].reset_index(drop=True)
    if not len(k):
        return "нет данных"
    px = float(k_full[k_full.open_time + H == t_close].close.iloc[0]) if (k_full.open_time + H == t_close).any() else None
    if px is None:
        return "нет данных"
    tp, sl = px * 1.2, px * 0.9
    for r in k.itertuples():
        if r.open_time < t_close:
            continue
        if r.low <= sl:
            return f"стоп −10% ({L3.kyiv(r.open_time + H)})"
        if r.high >= tp:
            return f"тейк +20% ({L3.kyiv(r.open_time + H)})"
        if r.open_time + H >= t_close + L3.HOLD_H * H:
            return f"7 дней: {(r.close / px - 1) * 100:+.1f}%"
    last = k.close.iloc[-1]
    return f"ещё открыта по архиву: сейчас {(last / px - 1) * 100:+.1f}%"


def f(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    coins = L3.load_json(L3.UNI_FILE, {}).get("coins", {})
    out = ["# Диагностика расхождений L3: бот (Coinalyze) против архива Binance", "",
           "A — только архив (как сверка и тест); B — как видит бот (архив до начала суток + Coinalyze); "
           "C — как B, но L/S из архива. «✔» — условие L3 выполнено на этом часе, «★» — свежий сигнал.",
           "Порог: сверху ≥ 3% капы и больше, чем снизу; L/S < 0,8.", ""]
    rows_csv = []
    for sym, kt, what in CASES:
        t = ms(kt)
        d0 = dt.datetime.fromtimestamp(t / 1000 - 1, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        d0 = d0.timestamp() * 1000
        end = t + AFTER * H
        czsym = coins.get(sym, {}).get("cz") or f"{sym}_PERP.A"
        cap = coins.get(sym, {}).get("cap")
        out.append(f"## {sym} · {kt[8:10]}.{kt[5:7]} {kt[11:]} — {what}")
        if not cap:
            out += ["Монеты нет в сегодняшнем списке бота (капа вне $10M–$2B или нет данных) — пропуск мог быть из-за этого.", ""]
            caps = L3.LC.caps_with_fallback()
            cap = caps.get(L3.Hst.base_asset(sym), 0)
            if not cap:
                out += ["Капы нет — пропускаю.", ""]
                continue
        try:
            k_full, met_full = L3.archive(sym)
            c_last = float(k_full.close.iloc[-1])
            c_t = float(k_full[k_full.open_time + H <= t].close.iloc[-1])
            cap_d = cap * c_t / c_last
            oh, oi, ls = cz_data(czsym, d0 - 2 * 86400e3, end + H)
            sA, kA = series(k_full, met_full, cap_d, end)
            kB, mB = bot_like(k_full, met_full, czsym, d0, end, oh, oi, ls, False)
            sB, _ = series(kB, mB, cap_d, end)
            kC, mC = bot_like(k_full, met_full, czsym, d0, end, oh, oi, ls, True)
            sC, _ = series(kC, mC, cap_d, end)
        except Exception as e:  # noqa: BLE001
            out += [f"ошибка: {e!r}", ""]
            continue
        prev = met_full[met_full.t < d0]
        ls_prev = float(prev.glob.dropna().iloc[-1]) if len(prev) and prev.glob.notna().any() else np.nan
        out.append(f"Предфильтр бота: L/S в архиве на конец прошлых суток = {f(ls_prev)} "
                   f"({'проходит' if ls_prev < L3.PREFILTER_LS else 'НЕ проходит — бот монету в эти сутки не считал'} < 1,0)")
        A_ = PL.AsOf(met_full, "oi")
        Al = PL.AsOf(met_full, "glob")
        czc = {b_["t"] * 1000 + H: b_["c"] for b_ in oh.get(czsym, [])}
        czo = {b_["t"] * 1000 + H: b_["c"] for b_ in oi.get(czsym, [])}
        czl = {b_["t"] * 1000 + H: b_.get("r") for b_ in ls.get(czsym, [])}
        out += ["", "| час (Киев) | A сверху / снизу / L/S | B сверху / снизу / L/S | C сверху / L/S | цена архив / CZ | OI архив / CZ | L/S архив / CZ |",
                "|---|---|---|---|---|---|---|"]
        for h in range(-BEFORE, AFTER + 1):
            tt = int(t + h * H)
            cell = []
            for s_ in (sA, sB):
                r = s_.get(tt)
                cell.append("—" if not r else f"{f(r['up'])} / {f(r['dn'])} / {f(r['ls'], 3)} "
                            f"{'★' if fresh(s_, tt) else ('✔' if L3.is_on(r) else '')}")
            r = sC.get(tt)
            cell.append("—" if not r else f"{f(r['up'])} / {f(r['ls'], 3)} {'★' if fresh(sC, tt) else ('✔' if L3.is_on(r) else '')}")
            pa = kA[kA.open_time + H == tt].close
            pa = float(pa.iloc[0]) if len(pa) else np.nan
            mark = " ◀" if h == 0 else ""
            out.append(f"| {L3.kyiv(tt)}{mark} | {cell[0]} | {cell[1]} | {cell[2]} | {f(pa, 5)} / {f(czc.get(tt, np.nan), 5)} | "
                       f"{f(A_.at(tt, 2 * H), 0)} / {f(czo.get(tt, np.nan), 0)} | {f(Al.at(tt, 2 * H), 3)} / {f(czl.get(tt, np.nan), 3)} |")
            rows_csv.append({"случай": f"{sym} {kt}", "час": L3.kyiv(tt),
                             **{f"{n} {x}": (s_.get(tt) or {}).get(x) for n, s_ in (("A", sA), ("B", sB), ("C", sC)) for x in ("up", "dn", "ls")},
                             "цена архив": pa, "цена CZ": czc.get(tt), "OI архив": A_.at(tt, 2 * H), "OI CZ": czo.get(tt),
                             "L/S архив": Al.at(tt, 2 * H), "L/S CZ": czl.get(tt)})
        if "пропущен" in what or "совпал" in what:
            out.append(f"\nСделка по архиву (вход {L3.kyiv(t)}): {outcome(k_full, t)}")
        out.append("")
    open(f"{OUT}/l3_diag_summary.md", "w").write("\n".join(out))
    pd.DataFrame(rows_csv).to_csv(f"{OUT}/l3_diag_hours.csv", index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
