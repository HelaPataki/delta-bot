"""
ВРЕМЕННО (удалить вместе с replay_check.py). Гипотеза: прогнозная ставка Coinalyze — «мгновенная»,
а Binance платит СРЕДНЮЮ за интервал. Проверка на 42 свечах (16 настоящих + 26 лишних):
по каждой выплате архива сравниваем
  • «точка» — прогноз в момент выплаты (как сейчас в новой схеме);
  • «среднее» — средний прогноз за весь интервал перед выплатой.
Затем фандинг сигнальной и прошлой свечи каждым способом и совпадение решений (порог −0,2% и −0,03%) с архивом.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

import live as L
import replay_check as RC  # noqa: F401  — включает паузу на время работы живого бота
from replay_extras import EXTRAS, funding

KYIV = ZoneInfo("Europe/Kyiv")
H = 3600
REAL = """CELRUSDT 20.09 19:00|KERNELUSDT 22.09 07:00|RLCUSDT 05.10 19:00|NMRUSDT 06.10 19:00|OGNUSDT 08.10 15:00|
CTSIUSDT 08.10 19:00|KAIAUSDT 09.10 19:00|RSRUSDT 12.10 15:00|GTCUSDT 12.10 23:00|CTKUSDT 13.10 15:00|BIRBUSDT 21.10 07:00|
MEWUSDT 21.10 07:00|ORDERUSDT 24.10 15:00|TUSDT 25.10 06:00|POLYXUSDT 25.10 14:00|BLURUSDT 29.10 06:00"""


def parse(s):
    out = []
    for item in s.replace("\n", "").split("|"):
        sym, d, hm = item.split()
        out.append((sym, f"{d} {hm}", int(dt.datetime.strptime(f"2026.{d} {hm}", "%Y.%d.%m %H:%M").replace(tzinfo=KYIV).timestamp())))
    return out


def main():
    cases = [("✅", *c) for c in parse(REAL)] + [("➕", *c) for c in parse(EXTRAS)]
    lines, err_p, err_a = [], [], []
    agree = {"точка": [0, 0], "среднее": [0, 0]}
    for kind, sym, label, T in cases:
        f = funding(sym)
        cs = sym + "_PERP.A"
        pr = {b["t"] + H: b["c"] for b in L.history("predicted-funding-rate-history", [cs], "1hour", T - 40 * H, T + 60).get(cs, [])
              if b.get("c") is not None}
        if f is None or not pr:
            lines.append(f"{kind} {sym} {label}: нет данных")
            continue
        w = f[(f.t > (T - 28 * H) * 1000) & (f.t <= T * 1000)]
        rows = []
        for r in w.itertuples():
            s, h = int(r.t // 1000 // H * H), r.hours
            pt = pr.get(s)
            av = [pr[x] for x in range(s - int(h) * H + H, s + H, H) if x in pr]
            act = r.rate * 8 / h * 100
            p8 = pt * 8 / h * 100 if pt is not None else np.nan
            a8 = float(np.mean(av)) * 8 / h * 100 if av else np.nan
            rows.append((s, act, p8, a8))
            if not np.isnan(p8):
                err_p.append(abs(p8 - act))
            if not np.isnan(a8):
                err_a.append(abs(a8 - act))

        def f24(X, col):
            v = [x[col] for x in rows if X - 24 * H < x[0] <= X]
            return float(np.nanmean(v)) if v else np.nan
        arch, arch_p = f24(T, 1), f24(T - 4 * H, 1)
        res = {}
        for name, col in (("точка", 2), ("среднее", 3)):
            sig, prv = f24(T, col), f24(T - 4 * H, col)
            res[name] = (sig, prv)
            agree[name][0] += (sig <= -0.2) == (arch <= -0.2)
            agree[name][1] += (prv <= -0.03) == (arch_p <= -0.03)
        lines.append(f"{kind} {sym} {label}: архив {arch:.2f} / прошлая {arch_p:.2f} · точка {res['точка'][0]:.2f} / {res['точка'][1]:.2f}"
                     f" · среднее {res['среднее'][0]:.2f} / {res['среднее'][1]:.2f}")
        print(lines[-1], flush=True)
    n = len(cases)
    head = ["🧪 Фандинг: прогноз «в точке» против «среднего за интервал» (архив = правда)",
            f"Ошибка одной выплаты, медиана: точка {np.median(err_p):.3f}%, среднее {np.median(err_a):.3f}%",
            f"Совпало решение «сигнальная свеча ≤ −0,2%»: точка {agree['точка'][0]}/{n}, среднее {agree['среднее'][0]}/{n}",
            f"Совпало решение «прошлая свеча ≤ −0,03%»: точка {agree['точка'][1]}/{n}, среднее {agree['среднее'][1]}/{n}",
            "Строки: архив / прошлая · точка · среднее (в % на 8 ч)"]
    print("\n".join(head + lines))
    chunk = ""
    for ln in head + lines:                      # Telegram режет после 4000 знаков — шлём частями
        if len(chunk) + len(ln) > 3800:
            L.tg(chunk)
            chunk = ""
        chunk += ln + "\n"
    L.tg(chunk)


if __name__ == "__main__":
    main()
