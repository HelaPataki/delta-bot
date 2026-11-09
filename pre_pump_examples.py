"""
Два случайных примера сделки с сентября 2026 (одна в плюсе, одна в минусе) — максимально подробно.

Условие: л/ш аккаунтов ≤ 0.8, OI за 3д ≥ +10%, накопленная дельта 7д ≥ 0 (стало выполняться на этой свече,
пауза 72 ч между сигналами монеты). Вход на подтверждении: первая 1h-свеча в течение 72 ч, закрывшаяся
на +6% выше цены сигнала с объёмом >= x3. Выход: стоп 12%, тейк 10% (и для сравнения 20%), не дольше 48 ч.
Кандидаты — монеты, где с сентября был вход на подтверждении в прогоне pre_pump_trades (л/ш ≤ 1).
"""

import os
import random

os.environ.setdefault("INTERVAL", "1h")
os.environ.setdefault("START", "2026-04-01")
os.environ.setdefault("SIG_FROM", "2026-08-01")

import numpy as np
import pandas as pd

import history as Hst
import pre_pump as PP
import pre_pump_trades as TR

OUT = "results/pre_pump_examples"
FROM = pd.Timestamp("2026-09-01", tz=Hst.KYIV).timestamp() * 1000
H = 3600 * 1000
HOLD, WAIT = PP.C(48), PP.C(72)
STOP = 12.0


def kyiv(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(Hst.KYIV).strftime("%Y-%m-%d %H:%M")


def find(sym, cap, btc):
    f = PP.coin(sym, cap, btc, keep_all=True)
    if f is None:
        return []
    c, vx, t = f.close.to_numpy(), f.vol_x.to_numpy(), f.open_time.to_numpy()
    h, l = f.high.to_numpy(), f.low.to_numpy()
    cond = (f.ls_acc <= 0.8) & (f.oi_chg3d >= 10) & (f.cvd7d >= 0)
    out = []
    for s in TR.signals(cond, f.elig.to_numpy()):
        if t[s] < FROM:
            continue
        e = next((j for j in range(s + 1, min(s + 1 + WAIT, len(c))) if c[j] >= c[s] * 1.06 and vx[j] >= 3), None)
        if e is None or e + HOLD >= len(c):
            continue
        res = {}
        for tk in (10, 20):
            px = c[e]
            for k in range(1, HOLD + 1):
                if l[e + k] <= px * (1 - STOP / 100):
                    res[tk] = (-STOP, e + k, "стоп")
                    break
                if h[e + k] >= px * (1 + tk / 100):
                    res[tk] = (tk, e + k, "тейк")
                    break
            else:
                res[tk] = ((c[e + HOLD] / px - 1) * 100, e + HOLD, "по времени (48 ч)")
        out.append((f, s, e, res))
    return out


def describe(sym, f, s, e, res, cond_prev, main_tk=10, extra=""):
    r = f.iloc
    sig, ent = r[s], r[e]
    L = []
    L.append(f"## {sym} — {'ПЛЮС' if res[main_tk][0] > 0 else 'МИНУС'} (тейк {main_tk}%: {res[main_tk][0] - 0.1:+.2f}% после комиссии)\n")
    L.append("### 1. Сигнал — условие стало выполняться\n")
    L.append(f"Сигнальная свеча: {kyiv(sig.open_time)}–{kyiv(sig.open_time + H)[-5:]} по Киеву (часовая), "
             f"закрытие **{sig.close:.6g}**. Капа (оценка) ${sig.mcap / 1e6:.0f}M, оборот свечи ${sig.turnover / 1e3:.0f}K.\n")
    L.append("| показатель на закрытии | значение | порог | на свече раньше |\n|---|---|---|---|")
    p = r[s - 1]
    L.append(f"| лонг/шорт по аккаунтам | {sig.ls_acc:.3f} | ≤ 0.8 | {p.ls_acc:.3f} |")
    L.append(f"| OI за 3 дня | {sig.oi_chg3d:+.1f}% | ≥ +10% | {p.oi_chg3d:+.1f}% |")
    L.append(f"| накопленная дельта 7 дней | {sig.cvd7d:+.2f}% объёма | ≥ 0 | {p.cvd7d:+.2f}% |")
    L.append(f"\nНа свече раньше не выполнялось: {cond_prev}.\n")
    L.append("Фон на момент сигнала (для понимания, в условие не входит): "
             f"цена за 1д {sig.px1d:+.1f}%, за 3д {sig.px3d:+.1f}%, за 7д {sig.px7d:+.1f}%; "
             f"OI за 1д {sig.oi_chg1d:+.1f}%; фандинг 1д {sig.fund1d:+.4f}% за 8ч; "
             f"лонг/шорт топ-трейдеров {sig.ls_top:.2f}; макс. аномалия объёма за 3д x{sig.volx_max3d:.1f}; "
             f"ширина диапазона 7д {sig.range7d:.0f}%; BTC за 7д {sig.btc7d:+.1f}%.\n")
    L.append("### 2. Ожидание подтверждения (+6% к цене сигнала и объём ≥ x3)\n")
    lvl = sig.close * 1.06
    L.append(f"Уровень подтверждения: {sig.close:.6g} × 1.06 = **{lvl:.6g}**.\n")
    L.append("| час (закрытие) | закрытие | к цене сигнала | объём к норме | подтверждение? |\n|---|---|---|---|---|")
    for j in range(s + 1, e + 1):
        q = r[j]
        ok = "**да → вход**" if j == e else ("цена да, объём нет" if q.close >= lvl else "нет")
        if j == e or e - j < 8 or q.close >= lvl or q.vol_x >= 3:
            L.append(f"| {kyiv(q.open_time + H)} | {q.close:.6g} | {(q.close / sig.close - 1) * 100:+.1f}% | x{q.vol_x:.1f} | {ok} |")
    if e - s > 9:
        L.append(f"\n(показаны последние 8 часов и часы с ценой выше уровня или объёмом ≥ x3; всего ждали {e - s} ч)\n")
    L.append("\n### 3. Вход\n")
    px = ent.close
    L.append(f"Вход по закрытию свечи {kyiv(ent.open_time + H)}: **{px:.6g}**. "
             f"Стоп −12% = **{px * 0.88:.6g}**, тейк +10% = **{px * 1.10:.6g}** (вариант тейк +20% = {px * 1.2:.6g}). "
             f"Держать не дольше 48 ч, до {kyiv(ent.open_time + H + 48 * H)}.\n")
    L.append("При x2 и риске 2% депозита: позиция ≈ 17% депозита по номиналу, маржа ≈ 8%; "
             "стоп = −2% депозита, тейк 10% ≈ +1.7% депозита. В Binance (ROI при x2): стоп −24%, тейк +20%.\n")
    L.append("### 4. Что было после входа\n")
    L.append("| час (закрытие) | макс. | мин. | закрытие | макс. от входа | мин. от входа |\n|---|---|---|---|---|---|")
    end = max(res[10][1], res[20][1])
    for k in range(e + 1, end + 1):
        q = r[k]
        L.append(f"| {kyiv(q.open_time + H)} | {q.high:.6g} | {q.low:.6g} | {q.close:.6g} | "
                 f"{(q.high / px - 1) * 100:+.1f}% | {(q.low / px - 1) * 100:+.1f}% |")
    for tk in (10, 20):
        pct, k, kind = res[tk]
        L.append(f"\n**Тейк {tk}%:** {kind} на свече {kyiv(r[k].open_time + H)}, результат {pct - 0.1:+.2f}% цены "
                 f"(при x2: {2 * (pct - 0.1):+.1f}% ROI; при риске 2%: {(pct - 0.1) / STOP * 2:+.2f}% депозита).")
    mx = f.high.to_numpy()[e + 1:e + 1 + HOLD].max() / px - 1
    mn = f.low.to_numpy()[e + 1:e + 1 + HOLD].min() / px - 1
    if extra:
        L.append("\n" + extra)
    L.append(f"\nЗа все 48 ч после входа: максимум {mx * 100:+.1f}%, минимум {mn * 100:+.1f}% от цены входа.\n")
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv("results/pre_pump_trades/pre_pump_trades.csv")
    tr = tr[tr.strat.str.startswith("толпа в шортах + OI") & (tr.entry == "B подтверждение") & (tr.signal_time >= "2026-09-01")]
    syms = sorted(tr.symbol.unique())
    caps = Hst.coingecko_mcaps()
    b = Hst.load_klines("BTCUSDT", PP.INTERVAL)
    btc = pd.DataFrame({"open_time": b.open_time, "btc7d": (b.close / b.close.shift(PP.C(168)) - 1) * 100})
    found = []
    for sym in syms:
        cap = caps.get(Hst.base_asset(sym))
        if not cap:
            continue
        for f, s, e, res in find(sym, cap, btc):
            found.append((sym, f, s, e, res))
        print(sym, len(found), flush=True)
    rnd = random.Random(int(os.environ.get("SEED", "8")))
    wins = [x for x in found if x[4][10][0] > 0]
    losses = [x for x in found if x[4][10][0] <= 0]
    text = ["# Два случайных примера с сентября 2026\n",
            "Условие: лонг/шорт по аккаунтам ≤ 0.8, OI за 3 дня ≥ +10%, накопленная дельта 7 дней ≥ 0. "
            "Вход на подтверждении (+6% к цене сигнала, объём ≥ x3, в течение 72 ч). Стоп 12%, тейк 10%, до 48 ч. "
            "Время — по Киеву, время свечи = её закрытие. Комиссия 0.1%.\n",
            f"Всего таких сделок с сентября найдено: {len(found)} (в плюсе при тейке 10%: {len(wins)}, в минусе: {len(losses)}).\n"]
    for pool in (wins, losses):
        if not pool:
            continue
        sym, f, s, e, res = rnd.choice(pool)
        p = f.iloc[s - 1]
        prev = []
        if not p.ls_acc <= 0.8:
            prev.append(f"лонг/шорт был {p.ls_acc:.3f}")
        if not p.oi_chg3d >= 10:
            prev.append(f"OI за 3д был {p.oi_chg3d:+.1f}%")
        if not p.cvd7d >= 0:
            prev.append(f"дельта 7д была {p.cvd7d:+.2f}%")
        text.append(describe(sym, f, s, e, res, ", ".join(prev) or "—"))
    rows = [{"монета": x[0], "сигнал": kyiv(x[1].open_time.iloc[x[2]] + H), "вход": kyiv(x[1].open_time.iloc[x[3]] + H),
             "тейк 10%": f"{x[4][10][2]} {x[4][10][0] - 0.1:+.1f}%", "тейк 20%": f"{x[4][20][2]} {x[4][20][0] - 0.1:+.1f}%"}
            for x in sorted(found, key=lambda x: x[1].open_time.iloc[x[3]])]
    text.append("## Все сделки с сентября\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    open(f"{OUT}/examples.md", "w").write("\n".join(text))
    print("\n".join(text)[:3000])


if __name__ == "__main__":
    main()
