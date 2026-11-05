"""
Поиск шорта: необычная для монеты дельта и накопленная дельта. Без сделок — только направление.

По всем 4h-свечам USDT-перпетуалов Binance с 2025-01-01 (оборот >= $50K, капа $10M–$400M).

A. Необычная дельта: дельта свечи сравнивается с нормой этой монеты за N дней до неё
   (медиана и типичный разброс — MAD, чтобы редкие выбросы не раздували норму).
   «В K раз необычнее» = (дельта − медиана) / разброс >= K.
   Условие шорта: необычно сильные покупки, но свеча выросла меньше чем на 1%.
B. Накопленная дельта: сумма дельт за окно / весь объём окна (24 ч или 3 дня), и цена за то же окно.
   Условие шорта: покупатели перевешивают (>= X%), а цена за окно не выросла.
   Берётся только первая свеча, где условие стало выполняться (чтобы не считать одну ситуацию много раз).
Каждый вариант — с всплеском объёма (>= x3 к медиане 12 свечей) и без.

Для каждой группы: что раньше (−4% или +4%), как часто −10% / +10% за 10 дней, и главное для входа —
насколько цена сначала поднималась перед падением на −10% и через сколько часов падение случалось.
"""

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.environ.get("OUT_DIR", "results/short_search")
H = 60   # 10 дней по 4h


def mad_z(d, n):
    s = pd.Series(d).shift(1)
    med = s.rolling(n, min_periods=n // 2).median()
    mad = (s - med).abs().rolling(n, min_periods=n // 2).median() * 1.4826
    return ((pd.Series(d) - med) / mad.replace(0, np.nan)).to_numpy()


def coin(sym, df, mcap_now):
    n = len(df)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    v, qv, tbv = df.volume.to_numpy(), df.quote_volume.to_numpy(), df.taker_buy_volume.to_numpy()
    dv = 2 * tbv - v
    delta = dv / v * 100
    chg = (c / o - 1) * 100
    norm = pd.Series(qv).shift(1).rolling(12, min_periods=12).median().to_numpy()
    vol_ratio = qv / norm
    mcap = mcap_now * c / c[-1]

    feat = {"z14": mad_z(delta, 84), "z30": mad_z(delta, 180)}
    for name, w in (("24h", 6), ("3d", 18)):
        feat[f"cum_{name}"] = (pd.Series(dv).rolling(w).sum() / pd.Series(v).rolling(w).sum() * 100).to_numpy()
        feat[f"px_{name}"] = (c / pd.Series(c).shift(w).to_numpy() - 1) * 100

    Hw, Lw = Hst.window(h, n), Hst.window(l, n)
    tracked = np.sum(~np.isnan(Hw), axis=1)
    up, dn = Hw / c[:, None] - 1, Lw / c[:, None] - 1
    res = {}
    for lev in (4, 10):
        for nm, hit in ((f"up{lev}", up >= lev / 100), (f"dn{lev}", dn <= -lev / 100)):
            a = hit.any(axis=1)
            res[nm] = np.where(a, np.argmax(hit, axis=1), np.nan)
    # что было до падения на −10%: максимум роста от закрытия до свечи, где случилось −10% (включительно)
    k = res["dn10"]
    rise = np.full(n, np.nan)
    for i in np.where(~np.isnan(k))[0]:
        rise[i] = (np.nanmax(Hw[i, : int(k[i]) + 1]) / c[i] - 1) * 100
    out = pd.DataFrame({
        "symbol": sym, "open_time": df.open_time.to_numpy(), "close": c, "change_pct": chg, "delta_pct": delta,
        "vol_ratio": vol_ratio, "turnover": qv, "mcap": mcap, "complete": tracked == H,
        "first_dn": np.where(np.isnan(res["dn4"]), 999, res["dn4"]) < np.where(np.isnan(res["up4"]), 999, res["up4"]),
        "first_up": np.where(np.isnan(res["up4"]), 999, res["up4"]) < np.where(np.isnan(res["dn4"]), 999, res["dn4"]),
        "dn10": ~np.isnan(res["dn10"]), "up10": ~np.isnan(res["up10"]),
        "dn10_hours": (res["dn10"] + 1) * 4, "rise_before_dn10": rise,
        "max_up": np.nanmax(up, axis=1) * 100, "max_dn": np.nanmin(dn, axis=1) * 100,
        **feat,
    })
    keep = (qv >= Hst.MIN_TURNOVER) & (mcap >= Hst.MCAP_MIN) & (mcap <= Hst.MCAP_MAX) & (v > 0)
    keep &= df.open_time.to_numpy() >= pd.Timestamp(Hst.START, tz="UTC").timestamp() * 1000
    return out[keep]


def first_only(df, mask):
    """Только первая свеча подряд идущей серии, где условие выполняется (по каждой монете)."""
    m = mask.astype(int)
    prev = m.groupby(df.symbol).shift(1).fillna(0)
    return mask & (prev == 0)


def stats(g):
    g = g[g.complete]
    if len(g) == 0:
        return None
    d = g[g.dn10]
    return {
        "сигналов": len(g),
        "сначала −4%": f"{g.first_dn.mean() * 100:.0f}%",
        "сначала +4%": f"{g.first_up.mean() * 100:.0f}%",
        "−10% за 10д": f"{g.dn10.mean() * 100:.0f}%",
        "+10% за 10д": f"{g.up10.mean() * 100:.0f}%",
        "рост перед −10%: медиана": f"{d.rise_before_dn10.median():.1f}%" if len(d) else "-",
        "рост перед −10%: 75% случаев до": f"{d.rise_before_dn10.quantile(0.75):.1f}%" if len(d) else "-",
        "часов до −10% (медиана)": f"{d.dn10_hours.median():.0f}" if len(d) else "-",
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    with_cap = [s for s in syms if Hst.base_asset(s) in caps]
    print(f"монет: {len(with_cap)}")

    def work(sym):
        try:
            df = Hst.load_klines(sym)
            if df is None or len(df) < 200:
                return None
            return coin(sym, df, caps[Hst.base_asset(sym)])
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return None

    parts = []
    with ThreadPoolExecutor(8) as ex:
        for i, p in enumerate(ex.map(work, with_cap), 1):
            if p is not None and len(p):
                parts.append(p)
            if i % 100 == 0:
                print(f"{i}/{len(with_cap)}")
    a = pd.concat(parts, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    days = (a.open_time.max() - a.open_time.min()) / 86_400_000

    spike = a.vol_ratio >= 3
    calm = a.change_pct < 1
    groups = [("все свечи (для сравнения)", a.delta_pct.notna())]
    for win in ("30", "14"):
        for k in (2, 3):
            m = (a[f"z{win}"] >= k) & calm
            groups.append((f"A: дельта ≥ {k}× необычнее нормы ({win} дн.), свеча < +1%", m))
            groups.append((f"A: то же + объём ≥ x3", m & spike))
    for win, wname in (("24h", "24 ч"), ("3d", "3 дня")):
        for x in (5, 10):
            m = first_only(a, (a[f"cum_{win}"] >= x) & (a[f"px_{win}"] <= 0))
            groups.append((f"B: накопл. дельта {wname} ≥ +{x}%, цена за {wname} ≤ 0", m))
            groups.append((f"B: то же + объём ≥ x3", m & spike))

    rows, flags = [], pd.DataFrame(index=a.index)
    for i, (name, m) in enumerate(groups):
        r = stats(a[m])
        if r:
            rows.append({"группа": name, "в день": f"{m.sum() / days:.1f}", **r})
        if i:
            flags[f"g{i}"] = m
    tbl = pd.DataFrame(rows)
    with open(f"{OUT_DIR}/short_search_summary.md", "w") as f:
        f.write("# Поиск шорта: необычная дельта и накопленная дельта\n\n")
        f.write("Только направление, без сделок. От цены закрытия сигнальной свечи, 10 дней вперёд по 4h high/low.\n")
        f.write("«Рост перед −10%» — насколько цена успевала подняться от закрытия, прежде чем упасть на 10% "
                "(только для случаев, где −10% было). Это подсказка, где ставить вход и стоп.\n\n")
        f.write(tbl.to_markdown(index=False) + "\n")
    sel = flags.any(axis=1)
    out = a[sel].copy()
    out["группы"] = flags[sel].apply(lambda r: ",".join(c for c in flags.columns if r[c]), axis=1)
    out.insert(1, "candle_open_kyiv", pd.to_datetime(out.open_time, unit="ms", utc=True)
               .dt.tz_convert(Hst.KYIV).dt.strftime("%Y-%m-%d %H:%M"))
    out.drop(columns=["open_time"]).round(3).to_csv(f"{OUT_DIR}/short_search_signals.csv", index=False)
    with open(f"{OUT_DIR}/short_search_summary.md", "a") as f:
        f.write("\nКоды групп в файле сигналов:\n\n" + "\n".join(f"- g{i}: {n}" for i, (n, _) in enumerate(groups) if i) + "\n")
    print(open(f"{OUT_DIR}/short_search_summary.md").read())


if __name__ == "__main__":
    main()
