"""
Угасание после пампа (шорт). Только 4h-свечи, без минуток.

Памп (объём >= x5 к медиане 12 свечей до пампа; капа $10M–$400M; оборот >= $50K):
  «1 свеча»  — рост от открытия до максимума одной свечи >= P%
  «до 3 свечей» — рост от минимума до максимума за окно до 3 свечей (12 ч) >= P%, максимум — на последней свече окна
  P = 10 / 15 / 20 / 30 %.
Цена пампа, где «запертые покупатели» набирали, = VWAP пампа (средняя цена всех сделок пампа = оборот / объём).

Наблюдение после пампа — 7 дней (42 свечи):
  отмена: цена обновила максимум пампа (памп продолжается);
  угасание (сигнал): первая свеча, где объём <= 1.5 x норма до пампа И закрытие ниже VWAP пампа.
  День сигнала: 1..7 после пампа.
Для сравнения («все свечи после пампа»): все свечи наблюдения до отмены, без условия угасания.

Результат от закрытия сигнальной свечи, 10 дней:
  что раньше −4% или +4%, как часто −10% / +10%;
  сделка: шорт по закрытию, стоп за максимумом пампа +0.5%, тейк = 1.8 x риск
  (если стоп и тейк в одной 4h-свече — считается стоп). Безубыточно = 0R.
"""

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.environ.get("OUT_DIR", "results/pump_fade")
SIZES = [10, 15, 20, 30]
WATCH = 42            # 7 дней
HOR = 60              # 10 дней
RR = 1.8
STOP_BUF = 0.5


def outcome(h, l, c, j, stop_px):
    """Направление и сделка от закрытия свечи j."""
    e = c[j]
    hw, lw = h[j + 1:j + 1 + HOR], l[j + 1:j + 1 + HOR]
    complete = len(hw) == HOR
    up, dn = hw / e - 1, lw / e - 1

    def first(x):
        return int(np.argmax(x)) if x.any() else 10 ** 6

    u4, d4 = first(up >= 0.04), first(dn <= -0.04)
    risk = stop_px / e - 1
    take_px = e * (1 - RR * risk)
    s_i, t_i = first(hw >= stop_px), first(lw <= take_px)
    if s_i == t_i == 10 ** 6:
        res, R = "по времени", ((e - c[min(j + HOR, len(c) - 1)]) / (stop_px - e)) if complete else np.nan
    elif s_i <= t_i:
        res, R = "стоп", -1.0
    else:
        res, R = "тейк", RR
    return {"complete": complete, "first_dn": d4 < u4, "first_up": u4 < d4,
            "dn10": bool((dn <= -0.10).any()), "up10": bool((up >= 0.10).any()),
            "risk_pct": risk * 100, "result": res, "R": R}


def coin(sym, df, mcap_now):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    v, qv = df.volume.to_numpy(), df.quote_volume.to_numpy()
    t = df.open_time.to_numpy()
    n = len(df)
    norm = pd.Series(qv).shift(1).rolling(12, min_periods=12).median().to_numpy()
    mcap = mcap_now * c / c[-1]
    start = pd.Timestamp(Hst.START, tz="UTC").timestamp() * 1000
    rows = []
    for form in ("1 свеча", "до 3 свечей"):
        busy_until = -1
        for i in range(13, n - 1):
            if i <= busy_until or t[i] < start:
                continue
            if form == "1 свеча":
                k0 = i
                rise = h[i] / o[i] - 1
            else:
                k0 = max(i - 2, 0)
                rise = h[i] / l[k0:i + 1].min() - 1
                if h[i] < h[k0:i + 1].max():
                    continue
            pre = norm[k0]
            if not (pre > 0) or qv[k0:i + 1].max() < 5 * pre or rise < SIZES[0] / 100:
                continue
            if not (Hst.MCAP_MIN <= mcap[i] <= Hst.MCAP_MAX) or qv[i] < Hst.MIN_TURNOVER:
                continue
            p_high = h[k0:i + 1].max()
            vwap = qv[k0:i + 1].sum() / v[k0:i + 1].sum()
            stop_px = p_high * (1 + STOP_BUF / 100)
            base = {"symbol": sym, "form": form, "pump_open_time": t[k0], "pump_rise_pct": rise * 100,
                    "pump_vol_x": qv[k0:i + 1].max() / pre, "pump_high": p_high, "pump_vwap": vwap, "mcap": mcap[i]}
            faded = False
            last = i
            for j in range(i + 1, min(i + 1 + WATCH, n)):
                last = j
                if h[j] > p_high:
                    break                                   # памп продолжается — отмена
                day = (4 * (j - i) + 23) // 24
                is_fade = (not faded) and qv[j] <= 1.5 * pre and c[j] < vwap
                rows.append({**base, "candle_time": t[j], "day": day, "signal": is_fade,
                             "close": c[j], **outcome(h, l, c, j, stop_px)})
                if is_fade:
                    faded = True
            busy_until = last                               # один памп — одно наблюдение
    return pd.DataFrame(rows)


def stats(g):
    gc = g[g.complete]
    if len(gc) == 0:
        return None
    closed = gc[gc.result.isin(["тейк", "стоп", "по времени"])]
    tk, st = (closed.result == "тейк").sum(), (closed.result == "стоп").sum()
    return {"свечей": len(gc),
            "сначала −4%": f"{gc.first_dn.mean() * 100:.0f}%", "сначала +4%": f"{gc.first_up.mean() * 100:.0f}%",
            "−10% за 10д": f"{gc.dn10.mean() * 100:.0f}%", "+10% за 10д": f"{gc.up10.mean() * 100:.0f}%",
            "сделка: винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
            "в среднем R": f"{closed.R.mean():+.2f}" if len(closed) else "-",
            "средний стоп": f"{closed.risk_pct.mean():.1f}%" if len(closed) else "-"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    with_cap = [s for s in syms if Hst.base_asset(s) in caps]

    def work(sym):
        try:
            df = Hst.load_klines(sym)
            if df is None or len(df) < 100:
                return None
            return coin(sym, df, caps[Hst.base_asset(sym)])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(8) as ex:
        a = pd.concat([p for p in ex.map(work, with_cap) if p is not None and len(p)], ignore_index=True)
    months = (a.candle_time.max() - a.candle_time.min()) / 86_400_000 / 30.4

    with open(f"{OUT_DIR}/pump_fade_summary.md", "w") as f:
        f.write("# Угасание после пампа (шорт)\n\n")
        f.write("Памп: объём ≥ x5, рост ≥ P% (одной свечой или до 3 свечей подряд). Угасание: объём ≤ 1,5 нормы до пампа "
                "и закрытие ниже VWAP пампа, нового максимума нет. День — сколько дней прошло после пампа.\n"
                f"Сделка: шорт по закрытию, стоп за максимумом пампа +{STOP_BUF:g}%, тейк = {RR:g} × риск "
                f"(безубыточный винрейт {1 / (1 + RR) * 100:.1f}%, безубыточно = 0R). Комиссии не учтены.\n\n")
        for form in ("1 свеча", "до 3 свечей"):
            for p in SIZES:
                g = a[(a.form == form) & (a.pump_rise_pct >= p)]
                if len(g) == 0:
                    continue
                pumps = g.drop_duplicates(["symbol", "pump_open_time"])
                f.write(f"## Памп {form}, рост ≥ +{p}% — {len(pumps)} пампов ({len(pumps) / months:.1f} в месяц)\n\n")
                rows = []
                for d in range(1, 8):
                    for name, m in (("угасание", g.signal), ("все свечи этого дня", g.day.notna())):
                        r = stats(g[m & (g.day == d)])
                        if r:
                            rows.append({"день": d, "группа": name, **r})
                for name, m in (("угасание, любой день", g.signal), ("все свечи после пампа", g.day.notna())):
                    r = stats(g[m])
                    if r:
                        rows.append({"день": "1–7", "группа": name, **r})
                f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    out = a.copy()
    for col in ("pump_open_time", "candle_time"):
        out[col] = (pd.to_datetime(out[col], unit="ms", utc=True).dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M"))
    out[out.signal].round(4).to_csv(f"{OUT_DIR}/pump_fade_signals.csv", index=False)
    print(open(f"{OUT_DIR}/pump_fade_summary.md").read())


if __name__ == "__main__":
    main()
