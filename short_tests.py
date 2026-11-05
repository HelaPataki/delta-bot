"""
Поиск шорта: другие стопы и вход после подтверждения.

Сигналы: шортовые свечи из results/variants/variants_trades.csv
(объём >= x5, POC в верхних 30% свечи, закрытие ниже POC; POC уже посчитан).
Варианты дельты в сводке: >= +17% / > 0 / любая.

Сценарии (везде тейк = 1,8 x расстояние от входа до стопа, если не сказано иначе):
  S0  вход POC − 5 тиков, стоп +4,3% от входа (как раньше)
  S1  вход POC − 5 тиков, стоп = максимум свечи + 0,5%
  S2  вход POC − 5 тиков, стоп = максимум свечи + 1%
  S3  вход POC − 5 тиков, стоп = максимум свечи + 2%
  C1  подтверждение, вход по рынку, стоп +4,3% от входа
  C2  подтверждение, вход по рынку, стоп = максимум свечи + 1%
  C3  подтверждение, ретест минимума сигнальной свечи, стоп +4,3% от входа
  C4  подтверждение, ретест минимума сигнальной свечи, стоп = максимум свечи + 1%
Подтверждение = следующая 4h-свеча закрылась ниже минимума сигнальной.
  «По рынку» = вход по цене закрытия подтверждающей свечи.
  «Ретест» = условный ордер на минимуме сигнальной свечи после подтверждения.
Условные ордера срабатывают при касании уровня с любой стороны; ожидание 1 свеча 4h и 24 часа.
Если в одной минутке задеты и стоп, и тейк — считается стоп. Держим до 10 дней. Комиссии не учтены.
Результат в R: тейк = +1,8R, стоп = −1R.
"""

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as B
from poc import kyiv_to_utc

IN_FILE = os.environ.get("IN_FILE", "results/variants/variants_trades.csv")
OUT_DIR = os.environ.get("OUT_DIR", "results/short_tests")
RR = 1.8
FIX_STOP = 4.3
TICKS = 5
H4 = 4 * 3600 * 1000
DAY = 24 * 3600 * 1000
HOLD = 10 * DAY
WAITS = {"1 свеча 4h": H4, "24 часа": DAY}
THREADS = 16

SCEN = {
    "S0": "вход POC−5т, стоп +4,3%",
    "S1": "вход POC−5т, стоп за максимумом +0,5%",
    "S2": "вход POC−5т, стоп за максимумом +1%",
    "S3": "вход POC−5т, стоп за максимумом +2%",
    "C1": "подтверждение → по рынку, стоп +4,3%",
    "C2": "подтверждение → по рынку, стоп за максимумом +1%",
    "C3": "подтверждение → ретест минимума, стоп +4,3%",
    "C4": "подтверждение → ретест минимума, стоп за максимумом +1%",
}


class Mins:
    """Минутки монеты, подгружаются по дням по мере надобности."""

    def __init__(self, sym, start_ms):
        self.sym = sym
        self.day = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).date()
        self.parts, self.loaded_to, self.days = [], start_ms - start_ms % DAY, 0

    def until(self, t_end):
        while self.loaded_to < t_end and self.days < 23:
            a = B.day_minutes(self.sym, self.day)
            if a is not None:
                self.parts.append(a)
            self.day += dt.timedelta(days=1)
            self.loaded_to += DAY
            self.days += 1

    def rows(self, t_from, t_to):
        self.until(t_to)
        for a in self.parts:
            m = (a[:, 0] >= t_from) & (a[:, 0] < t_to)
            for r in a[m]:
                yield r


def trade(mins, t_start, prev_price, entry_level, wait, stop_of, take_of, immediate=False):
    """Шорт. Возвращает (результат, R, ожидание входа в часах)."""
    if immediate:
        entry, fill_t = prev_price, t_start
    else:
        entry, fill_t, prev = entry_level, None, prev_price
        for t, o, h, l, c in mins.rows(t_start, t_start + wait):
            if (l <= entry <= h) or (prev < entry < l) or (h < entry < prev):
                fill_t = t
                break
            prev = c
        if fill_t is None:
            return "не налило", np.nan, np.nan
    stop, take = stop_of(entry), take_of(entry)
    if not (stop > entry > take):
        return "некорректно", np.nan, np.nan
    risk = stop - entry
    first = True
    last_c = None
    for t, o, h, l, c in mins.rows(fill_t, fill_t + HOLD):
        last_c = c
        if h >= stop:
            return "стоп", -1.0, (fill_t - t_start) / 3.6e6
        if not first and l <= take:
            return "тейк", RR, (fill_t - t_start) / 3.6e6
        first = False
    if last_c is None:
        return "нет данных", np.nan, np.nan
    return "по времени", (entry - last_c) / risk, (fill_t - t_start) / 3.6e6


def run_signal(row):
    out = []
    t_open = kyiv_to_utc(row["candle_open_kyiv"]).timestamp() * 1000
    t0 = t_open + H4
    mins = Mins(row["symbol"], t_open)
    sig = [r for r in mins.rows(t_open, t0)]
    if not sig:
        return out
    sig = np.array(sig)
    hi, lo = sig[:, 2].max(), sig[:, 3].min()
    tick = B.infer_tick(mins.parts[0])
    poc_entry = row["poc"] - TICKS * tick
    close = row["close"]

    fix_stop = lambda e: e * (1 + FIX_STOP / 100)
    fix_take = lambda e: e * (1 - FIX_STOP * RR / 100)

    def hi_stop(buf):
        return lambda e: hi * (1 + buf / 100)

    def hi_take(buf):
        return lambda e: e - RR * (hi * (1 + buf / 100) - e)

    base = {"symbol": row["symbol"], "candle_open_kyiv": row["candle_open_kyiv"], "delta_pct": row["delta_pct"],
            "vol_ratio": row["vol_ratio"]}

    for w_name, w in WAITS.items():
        for code, st, tk in (("S0", fix_stop, fix_take), ("S1", hi_stop(0.5), hi_take(0.5)),
                             ("S2", hi_stop(1), hi_take(1)), ("S3", hi_stop(2), hi_take(2))):
            res, r, wh = trade(mins, t0, close, poc_entry, w, st, tk)
            out.append({**base, "сценарий": code, "ожидание": w_name, "result": res, "R": r, "fill_wait_h": wh,
                        "stop_dist_pct": (st(poc_entry) / poc_entry - 1) * 100})

    # подтверждение: следующая свеча закрылась ниже минимума сигнальной
    nxt = [r for r in mins.rows(t0, t0 + H4)]
    confirmed = bool(nxt) and nxt[-1][4] < lo
    c_close = nxt[-1][4] if nxt else np.nan
    t1 = t0 + H4
    for code in ("C1", "C2", "C3", "C4"):
        for w_name, w in WAITS.items():
            if code in ("C1", "C2") and w_name != "1 свеча 4h":
                continue                                     # вход по рынку — ожидание не нужно
            w_label = w_name if code in ("C3", "C4") else "—"
            if not confirmed:
                out.append({**base, "сценарий": code, "ожидание": w_label, "result": "нет подтверждения",
                            "R": np.nan, "fill_wait_h": np.nan, "stop_dist_pct": np.nan})
                continue
            st, tk = (fix_stop, fix_take) if code in ("C1", "C3") else (hi_stop(1), hi_take(1))
            if code in ("C1", "C2"):
                res, r, wh = trade(mins, t1, c_close, None, 0, st, tk, immediate=True)
                e = c_close
            else:
                res, r, wh = trade(mins, t1, c_close, lo, w, st, tk)
                e = lo
            out.append({**base, "сценарий": code, "ожидание": w_label,
                        "result": res, "R": r, "fill_wait_h": wh, "stop_dist_pct": (st(e) / e - 1) * 100})
    return out


def stats(g):
    closed = g[g.result.isin(["тейк", "стоп", "по времени"])]
    tk, st = (closed.result == "тейк").sum(), (closed.result == "стоп").sum()
    return {
        "сигналов": len(g),
        "сделок": len(closed),
        "тейк": tk, "стоп": st,
        "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
        "в среднем R": f"{closed.R.mean():+.2f}" if len(closed) else "-",
        "сумма R": f"{closed.R.sum():+.1f}",
        "средний стоп": f"{closed.stop_dist_pct.mean():.1f}%" if len(closed) else "-",
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    v = pd.read_csv(IN_FILE)
    s = v[v.side == "шорт"].drop_duplicates(["symbol", "candle_open_kyiv"])
    print(f"шортовых сигналов: {len(s)}")
    with ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(run_signal, s.to_dict("records")))
    t = pd.DataFrame([x for r in res for x in r])
    t["R"] = t.R.round(3)
    t["stop_dist_pct"] = t.stop_dist_pct.round(2)
    t["fill_wait_h"] = t.fill_wait_h.round(1)
    t.to_csv(f"{OUT_DIR}/short_tests_trades.csv", index=False)

    variants = [("дельта ≥ +17%", t.delta_pct >= 17), ("дельта > 0", t.delta_pct > 0), ("дельта любая", t.delta_pct.notna())]
    with open(f"{OUT_DIR}/short_tests_summary.md", "w") as f:
        f.write("# Поиск шорта: стоп за максимумом и вход после подтверждения\n\n")
        f.write("Сигналы: объём ≥ x5, POC в верхних 30%, закрытие ниже POC. Тейк = 1,8 × риск. "
                "Безубыточный винрейт 35,7%, безубыточно = 0R. Комиссии не учтены.\n\n")
        f.write("Сценарии:\n\n" + "\n".join(f"- **{k}** — {v}" for k, v in SCEN.items()) + "\n\n")
        for vname, mask in variants:
            f.write(f"## {vname}\n\n")
            rows = []
            for code in SCEN:
                for w_name in ["1 свеча 4h", "24 часа", "—"]:
                    g = t[mask & (t["сценарий"] == code) & (t["ожидание"] == w_name)]
                    if len(g) == 0:
                        continue
                    rows.append({"сценарий": code, "ожидание": w_name, **stats(g)})
            f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT_DIR}/short_tests_summary.md").read())


if __name__ == "__main__":
    main()
