"""
Шорт: вход условным ордером внутри большой сигнальной свечи (идея Насти). Старые скрипты не меняются.

Сигналы — финальный шорт (фандинг ≤ −0,2% + OI ≥ +60% + цена стоит), 2025–2026 и для проверки 2021–2024
(с фильтром капитализации $10M–$400M).
Размер сигнальной свечи с тенями = (макс / мин − 1) · 100.
  ≤ 11%  — как раньше: вход по рынку через 8 мин после закрытия.
  > 11%  — условный ордер ровно в середине свечи с тенями: (макс + мин) / 2.
Ордер активен с 8-й минуты после закрытия свечи: 4 ч, 8 ч или 24 ч. Срабатывает при касании уровня с любой стороны,
вход ровно по уровню. Не сработал — либо пропуск, либо вход по рынку в конце ожидания.
Отдельно смотрим, где была цена в момент постановки ордера: ниже уровня (ждём отскок вверх, как POLYX) или выше.
Стоп +4,3%, тейк −7,74% от входа, выход через 24 ч после входа; комиссия 0,1%, фандинг учтён.
Итог — статистика по большим свечам и симуляция $100 (шорт + лонг, 2 слота) для каждого варианта.
"""

import datetime as dt
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import short_tests as ST
import combo_sim as CS

OUT = "results/short_mid"
H, MIN = 3600 * 1000, 60_000
H4 = 4 * H
DELAY = 8 * MIN
STOP, TAKE = 4.3, 4.3 * 1.8
BIG = 11.0
WAITS = [4, 8, 24]

_fund = {}


def funding_series(sym):
    if sym in _fund:
        return _fund[sym]
    frames = []
    for m in Hst.months(dt.date(2020, 12, 1), dt.datetime.now(dt.timezone.utc).date()):
        r = Hst.get(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if r is None or r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "last_funding_rate" not in df.columns:
            df = df.iloc[:, :3]
            df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        frames.append(df[["calc_time", "last_funding_rate"]].astype(float))
    f = pd.concat(frames).sort_values("calc_time") if frames else pd.DataFrame(columns=["calc_time", "last_funding_rate"])
    _fund[sym] = (f.calc_time.to_numpy(), f.last_funding_rate.to_numpy())
    return _fund[sym]


def run_short(mins, sym, t_in, entry):
    """Шорт с входом в минуту t_in по цене entry: стоп/тейк/24 ч."""
    stop, take = entry * (1 + STOP / 100), entry * (1 - TAKE / 100)
    res, t_out, px, first = None, None, None, True
    for t, o, h, l, c in mins.rows(t_in, t_in + 24 * H):
        px, t_out = c, t + MIN
        if h >= stop:
            res, px = "стоп", stop
            break
        if not first and l <= take:
            res, px = "тейк", take
            break
        first = False
    if px is None:
        return None
    ft, fr = funding_series(sym)
    sel = (ft > t_in) & (ft <= t_out)
    return {"entry_time": int(t_in), "exit_time": int(t_out), "result": res or "по времени", "entry": entry, "exit": px,
            "pnl_pct": (entry - px) / entry * 100, "funding_pct": fr[sel].sum() * 100}


def one(sig):
    sym, t_open, period = sig["symbol"], int(sig["open_time"]), sig["period"]
    T = t_open + H4
    mins = ST.Mins(sym, t_open)
    candle = list(mins.rows(t_open, T))
    if not candle:
        return []
    hi, lo = max(r[2] for r in candle), min(r[3] for r in candle)
    size = (hi / lo - 1) * 100
    base = {"period": period, "symbol": sym, "open_time": t_open, "size": size}
    start = list(mins.rows(T + DELAY, T + DELAY + MIN))
    if not start:
        return []
    p0 = start[0][1]
    out = []
    # базовый вариант: по рынку через 8 мин
    tr = run_short(mins, sym, T + DELAY, p0)
    if tr:
        out.append({**base, "variant": "рынок 8 мин", "side": "", "filled": True, **tr})
    if size <= BIG:
        return out
    lvl = (hi + lo) / 2                         # ровно середина свечи с тенями
    side = "ниже уровня" if p0 < lvl else "выше уровня"
    prev, t_fill = p0, None
    for t, o, h, l, c in mins.rows(T + DELAY, T + DELAY + max(WAITS) * H):
        if (l <= lvl <= h) or (prev < lvl < l) or (h < lvl < prev):
            t_fill = t
            break
        prev = c
    for w in WAITS:
        end = T + DELAY + w * H
        if t_fill is not None and t_fill < end:
            tr = run_short(mins, sym, t_fill, lvl)
            if tr:
                for mode in ("пропуск", "рынок в конце"):
                    out.append({**base, "variant": f"ордер {w} ч / {mode}", "side": side, "filled": True, "level": lvl, **tr})
        else:
            out.append({**base, "variant": f"ордер {w} ч / пропуск", "side": side, "filled": False, "level": lvl})
            m = list(mins.rows(end, end + MIN))
            if m:
                tr = run_short(mins, sym, end, m[0][1])
                if tr:
                    out.append({**base, "variant": f"ордер {w} ч / рынок в конце", "side": side, "filled": False, "level": lvl, **tr})
    return out


def signals():
    a = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz")
    a = a[(a.rr == 1.8) & (a.hold_h == 24)][["symbol", "open_time"]].assign(period="2025–26")
    b = pd.read_csv("results/y2021/short_trades_all.csv.gz")
    b = b[b.mcap.between(10e6, 400e6)][["symbol", "open_time"]].assign(period="2021–24")
    return pd.concat([a, b]).drop_duplicates(["symbol", "open_time"])


def net(d):
    return d.pnl_pct + d.funding_pct - 0.1


def sim(sh, lg):
    ev = sorted(CS.events(sh, "шорт") + CS.events(lg, "лонг"), key=lambda e: e[0])
    res = {}
    for lev in (1, 2):
        r = [CS.simulate(ev, lev, np.random.default_rng(i)) for i in range(CS.N_RUNS)]
        fin = np.array([x[0] for x in r])
        res[f"x{lev} средний"] = f"${np.median(fin):,.0f}"
        res[f"x{lev} плохой (5%)"] = f"${np.percentile(fin, 5):,.0f}"
        res[f"x{lev} просадка"] = f"{np.percentile([x[1] for x in r], 5) * 100:.0f}%"
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    sig = signals()
    print("сигналов:", len(sig), flush=True)
    with ThreadPoolExecutor(12) as ex:
        rows = [x for r in ex.map(one, sig.to_dict("records")) for x in r]
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/short_mid_trades.csv.gz", index=False)
    t["net"] = net(t.fillna({"pnl_pct": 0, "funding_pct": 0}))
    big = t[t["size"] > BIG]
    out = ["# Шорт: условный ордер в середине большой сигнальной свечи\n",
           "Свеча > 11% (с тенями): ордер ровно в середине свечи (макс + мин) / 2, активен с 8-й минуты. "
           "Стоп 4,3% / тейк 7,74% от входа, выход через 24 ч, комиссия 0,1%, фандинг учтён.\n"]
    for period in ("2025–26", "2021–24"):
        p = t[t.period == period]
        nsig = p.drop_duplicates(["symbol", "open_time"])
        nbig = nsig[nsig["size"] > BIG]
        out.append(f"## {period}\n")
        out.append(f"Сигналов: {len(nsig)}, из них свеча > 11%: **{len(nbig)}** "
                   f"(ниже уровня в момент постановки: {(p[(p['size'] > BIG) & (p.side == 'ниже уровня')].drop_duplicates(['symbol', 'open_time']).shape[0])}).\n")
        rows = []
        b = big[big.period == period]
        for v in ["рынок 8 мин"] + [f"ордер {w} ч / {m}" for w in WAITS for m in ("пропуск", "рынок в конце")]:
            for side in ("все", "ниже уровня", "выше уровня"):
                d = b[b.variant == v]
                if side != "все":
                    d = d[d.side == side] if v != "рынок 8 мин" else d[d.set_index(["symbol", "open_time"]).index.isin(
                        b[(b.variant != "рынок 8 мин") & (b.side == side)].set_index(["symbol", "open_time"]).index)]
                tr = d[d.result.notna()] if "result" in d else d
                if not len(d):
                    continue
                rows.append({"вариант": v, "цена при постановке": side, "сигналов": d.drop_duplicates(["symbol", "open_time"]).shape[0],
                             "сделок": len(tr), "сработал ордер": f"{d.filled.mean() * 100:.0f}%" if v != "рынок 8 мин" else "—",
                             "винрейт": f"{(tr.result == 'тейк').mean() * 100:.0f}%" if len(tr) else "—",
                             "% на сделку": f"{tr.net.mean():+.2f}%" if len(tr) else "—",
                             "сумма %": f"{tr.net.sum():+.0f}%" if len(tr) else "—"})
        out.append("### Только большие свечи (> 11%)\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")

    # симуляция всей стратегии: маленькие свечи — как было, большие — по варианту
    lg = {"2025–26": pd.read_csv("results/delay8/long/oi_sim_trades.csv.gz").query("rr == 4 and hold_h == 240"),
          "2021–24": pd.read_csv("results/y2021/long_trades_all.csv.gz").query("mcap >= 10e6 and mcap <= 400e6")}
    srows = []
    for period in ("2025–26", "2021–24"):
        p = t[(t.period == period) & t.result.notna()]
        baseline = p[p.variant == "рынок 8 мин"]
        small = baseline[baseline["size"] <= BIG]
        for v in ["рынок 8 мин"] + [f"ордер {w} ч / {m}" for w in WAITS for m in ("пропуск", "рынок в конце")]:
            for rule in ("все большие", "только ниже уровня"):
                if v == "рынок 8 мин" and rule != "все большие":
                    continue
                if v == "рынок 8 мин":
                    sh = baseline
                else:
                    bv = p[(p.variant == v) & (p["size"] > BIG)]
                    if rule == "только ниже уровня":
                        # сторону берём по ВСЕМ большим сигналам (и несработавшим), иначе в выборку попадает заглядывание в будущее
                        allb = t[(t.period == period) & (t.variant == v) & (t["size"] > BIG)]
                        keys = allb[allb.side == "выше уровня"].set_index(["symbol", "open_time"]).index
                        above = baseline[baseline.set_index(["symbol", "open_time"]).index.isin(keys)]
                        sh = pd.concat([small, bv[bv.side == "ниже уровня"], above])
                    else:
                        sh = pd.concat([small, bv])
                row = {"период": period, "вариант больших свечей": v, "к каким": rule if v != "рынок 8 мин" else "—",
                       "шортов": len(sh), "% на шорт": f"{net(sh).mean():+.2f}%"}
                row.update(sim(sh, lg[period]))
                srows.append(row)
                print(row, flush=True)
    out.append("## Симуляция $100: шорт + лонг, 2 слота (маленькие свечи всегда по рынку)\n")
    out.append(pd.DataFrame(srows).to_markdown(index=False) + "\n")
    open(f"{OUT}/short_mid_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
