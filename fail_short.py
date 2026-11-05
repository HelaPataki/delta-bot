"""
Шорт на провале лонгового сетапа. По минуткам.

Лонговый сетап (results/poc_signals.csv): дельта <= −17%, POC в нижних 30%, закрытие выше POC, объём >= x3.
Лонг: условный ордер на POC + 5 тиков, ждёт 1 свечу 4h; стоп −4,3%, тейк +7,74%.
Провал = после входа в лонг цена дошла до лонгового стопа раньше тейка (в пределах 10 дней).
Уровень стопа лонга = L.

Вход в шорт:
  F1 — условный ордер на уровне L: вход в момент вылета, по цене L;
  F2 — по закрытию 4h-свечи, в которой был вылет, если закрытие ниже L (иначе сделки нет);
  F3 — после вылета условный ордер на ретест L снизу, ждём до 24 ч.
Стоп шорта:
  A — над уровнем входа лонга (POC + 5 тиков);
  B — +4,3% от входа шорта.
Тейк = 1,8 x риск. Если стоп и тейк в одной минутке — стоп. Держим до 10 дней. Комиссии не учтены. Результат в R.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as B
import short_tests as ST
from poc import kyiv_to_utc

OUT_DIR = os.environ.get("OUT_DIR", "results/fail_short")
H4 = 4 * 3600 * 1000
DAY = 24 * 3600 * 1000
LSTOP, LTAKE, TICKS = 4.3, 7.74, 5
ST.RR = 1.8


def run(row):
    base = {k: row[k] for k in ("symbol", "candle_open_kyiv", "delta_pct", "vol_ratio")}
    t_open = kyiv_to_utc(row["candle_open_kyiv"]).timestamp() * 1000
    t0 = t_open + H4
    mins = ST.Mins(row["symbol"], t_open)
    list(mins.rows(t_open, t0))
    if not mins.parts:
        return [{**base, "long": "нет данных"}]
    tick = B.infer_tick(mins.parts[0])
    le = row["poc"] + TICKS * tick                     # вход лонга
    L, LT = le * (1 - LSTOP / 100), le * (1 + LTAKE / 100)

    # 1) лонг: налив за 1 свечу
    fill_t, prev = None, row["close"]
    for t, o, h, l, c in mins.rows(t0, t0 + H4):
        if (l <= le <= h) or (prev < le < l) or (h < le < prev):
            fill_t = t
            break
        prev = c
    if fill_t is None:
        return [{**base, "long": "не налило"}]
    # 2) лонг: что раньше — стоп или тейк
    hit_t, hit_c, first = None, None, True
    for t, o, h, l, c in mins.rows(fill_t, fill_t + 10 * DAY):
        if l <= L:
            hit_t, hit_c = t, c
            break
        if not first and h >= LT:
            return [{**base, "long": "тейк"}]
        first = False
    if hit_t is None:
        return [{**base, "long": "по времени"}]

    out = []
    stops = {"A": lambda e: le, "B": lambda e: e * (1 + LSTOP / 100)}
    takes = {"A": lambda e: e - ST.RR * (le - e), "B": lambda e: e * (1 - LSTOP * ST.RR / 100)}
    # F1: вход по L в момент вылета
    for sk in ("A", "B"):
        res, R, _ = ST.trade(mins, hit_t, L, None, 0, stops[sk], takes[sk], immediate=True)
        out.append({**base, "long": "стоп", "вход": "F1", "стоп_шорта": sk, "result": res, "R": R,
                    "stop_pct": (stops[sk](L) / L - 1) * 100, "wait_h": 0.0})
    # F2: закрытие 4h-свечи вылета
    c_end = hit_t - (hit_t % H4) + H4
    rows = list(mins.rows(hit_t, c_end))
    c_close = rows[-1][4] if rows else hit_c
    for sk in ("A", "B"):
        if c_close >= L:
            out.append({**base, "long": "стоп", "вход": "F2", "стоп_шорта": sk, "result": "закрылась выше уровня",
                        "R": np.nan, "stop_pct": np.nan, "wait_h": np.nan})
            continue
        res, R, _ = ST.trade(mins, c_end, c_close, None, 0, stops[sk], takes[sk], immediate=True)
        out.append({**base, "long": "стоп", "вход": "F2", "стоп_шорта": sk, "result": res, "R": R,
                    "stop_pct": (stops[sk](c_close) / c_close - 1) * 100, "wait_h": (c_end - hit_t) / 3.6e6})
    # F3: ретест L снизу, до 24 ч
    for sk in ("A", "B"):
        res, R, wh = ST.trade(mins, hit_t + 60_000, hit_c, L, DAY, stops[sk], takes[sk])
        out.append({**base, "long": "стоп", "вход": "F3", "стоп_шорта": sk, "result": res, "R": R,
                    "stop_pct": (stops[sk](L) / L - 1) * 100, "wait_h": wh})
    return out


def stats(g):
    closed = g[g.result.isin(["тейк", "стоп", "по времени"])]
    tk, st = (closed.result == "тейк").sum(), (closed.result == "стоп").sum()
    return {"сделок": len(closed), "тейк": tk, "стоп": st, "винрейт": f"{tk / max(tk + st, 1) * 100:.0f}%",
            "в среднем R": f"{closed.R.mean():+.2f}" if len(closed) else "-",
            "сумма R": f"{closed.R.sum():+.1f}", "средний стоп": f"{closed.stop_pct.mean():.1f}%" if len(closed) else "-"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p = pd.read_csv("results/poc_signals.csv")
    s = p[(p.delta_sign == "−") & (p.vol_ratio >= 3) & (p.poc_pos <= 30) & (p.close_below_poc == "нет") & p.poc.notna()]
    print(f"лонговых сигналов: {len(s)}")
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, s.to_dict("records")))
    t = pd.DataFrame([x for r in res for x in r])
    t.to_csv(f"{OUT_DIR}/fail_short_trades.csv", index=False)

    lg = t.drop_duplicates(["symbol", "candle_open_kyiv"])
    tr = t[t["вход"].notna()]
    with open(f"{OUT_DIR}/fail_short_summary.md", "w") as f:
        f.write("# Шорт на провале лонгового сетапа\n\n")
        f.write("Провал = лонг (вход POC + 5 тиков) выбило по стопу −4,3%. L — уровень этого стопа.\n\n"
                "- **F1** — шорт условным ордером на L в момент вылета\n"
                "- **F2** — шорт по закрытию 4h-свечи вылета, если она закрылась ниже L\n"
                "- **F3** — шорт на ретесте L снизу (ждём до 24 ч)\n"
                "- стоп **A** — над входом лонга; стоп **B** — +4,3% от входа шорта; тейк = 1,8 × риск\n\n"
                "Безубыточный винрейт 35,7%, безубыточно = 0R. Комиссии не учтены.\n\n")
        for name, lo, hi in (("x3–x5", 3, 5), ("x5+", 5, 10 ** 6), ("x10+", 10, 10 ** 6), ("все x3+", 3, 10 ** 6)):
            g = lg[(lg.vol_ratio >= lo) & (lg.vol_ratio < hi)]
            vc = g["long"].value_counts()
            f.write(f"## {name}: сигналов {len(g)}, лонг: налило {len(g) - vc.get('не налило', 0) - vc.get('нет данных', 0)}, "
                    f"тейк {vc.get('тейк', 0)}, стоп (провал) {vc.get('стоп', 0)}, по времени {vc.get('по времени', 0)}\n\n")
            rows = []
            for e in ("F1", "F2", "F3"):
                for sk in ("A", "B"):
                    gg = tr[(tr.vol_ratio >= lo) & (tr.vol_ratio < hi) & (tr["вход"] == e) & (tr["стоп_шорта"] == sk)]
                    rows.append({"вход": e, "стоп": sk, **stats(gg)})
            f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT_DIR}/fail_short_summary.md").read())


if __name__ == "__main__":
    main()
