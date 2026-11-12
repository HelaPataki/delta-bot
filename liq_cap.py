"""
Идея Анастасии: ликвидации «притягивают» цену, только если их много относительно капитализации.
(пример: капа $50M, шортовых ликвидаций сверху ~$5M = 10% капы — серьёзно; капа $1B и $7M — цена туда не идёт)
Живого бота и другие скрипты не трогает.

Карта — та же модель, что в liq_map.py (по OI архива Binance), суммы в долларах.
Модель завышает суммы примерно в 10–13 раз против реальных ликвидаций Binance (liq_check), поэтому
кроме «сырых» процентов от капы показываем пересчёт ÷13 — примерно в масштабе реальных ликвидаций.
Капа — оценка: текущая капа CoinGecko × цена тогда / цена сейчас.

Монеты: все USDT-перпетуалы Binance с текущей капой $10M–$2B. Период: с 2025-01-01, наблюдение на закрытии 4h-свечи.
Как у Анастасии на Coinglass: не одна полка, а ОБЩАЯ сумма ликвидаций выше цены (шорты) и ниже цены (лонги),
в пределах ±10% и ±20% от цены, в % от капы. Пороги — от 1% капы.
Три варианта карты по «возрасту» позиций (полураспад 3 / 14 / 45 дней) — похоже на окна Coinglass 7д / 30д / 90–180д.
Вопрос: если сверху жирнее (≥ X% капы и больше, чем снизу) — чаще ли цена сначала идёт вверх на 5% (10%),
чем вниз на 5% (10%), за 7 дней? И наоборот.
"""

import datetime as dt
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-12-01")

import numpy as np
import pandas as pd

import history as Hst
import liq_map as LM
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_cap")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = PL.H
FROM = pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000
WARM = FROM - 30 * 24 * H
FWD = 168
SCALE = 13.0


HL = {"3д": 3, "14д": 14, "45д": 45}      # полураспад старых позиций: примерно как окна Coinglass 7 дней / 30 дней / 90–180 дней
RANGES = [5, 10, 20]                     # сумма ликвидаций от цены до +R% (вверх) и до −R% (вниз)


def coin(sym, cap):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < 500:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    v, qv, t = k.volume.to_numpy(), k.quote_volume.to_numpy(), k.open_time.to_numpy()
    n, c_now = len(k), k.close.iloc[-1]
    days, d = set(), pd.Timestamp(max(WARM - 60 * 24 * H, t[0]), unit="ms", tz="UTC").date()
    while d < dt.datetime.now(dt.timezone.utc).date():
        days.add(d)
        d += dt.timedelta(days=1)
    met = PL.metrics(sym, days)
    if met is None or len(met) < 500:
        return None
    OI, OIU = PL.AsOf(met, "oi"), PL.AsOf(met, "oi_usd")
    lo_g = math.log(np.nanmin(l[l > 0]) * 0.3)
    G = int((math.log(np.nanmax(h) * 3) - lo_g) / LM.STEP) + 2
    maps = {name: [np.zeros(G), np.zeros(G), 0.5 ** (1 / (hl * 24))] for name, hl in HL.items()}
    lvl = np.exp(lo_g + (np.arange(G) + 0.5) * LM.STEP)
    gi = lambda p: min(G - 1, max(0, int((math.log(p) - lo_g) / LM.STEP)))
    prev, rows = np.nan, []
    for i in range(int(np.searchsorted(t, WARM - 60 * 24 * H)), n):
        if not (h[i] > 0 and l[i] > 0):
            continue
        oi = OI.at(t[i] + H, 2 * H)
        ih, il = gi(h[i]), gi(l[i])
        add = (oi - prev) if (np.isfinite(oi) and np.isfinite(prev) and oi > prev and v[i] > 0) else 0.0
        p = qv[i] / v[i] if v[i] > 0 else c[i]
        for longs, shorts, dec in maps.values():
            longs *= dec
            shorts *= dec
            shorts[:ih + 1] = 0
            longs[il:] = 0
            if add > 0:
                for L, wgt in LM.LEV.items():
                    longs[gi(p * (1 - 1 / L + LM.MMR))] += add * wgt
                    shorts[gi(p * (1 + 1 / L - LM.MMR))] += add * wgt
            if np.isfinite(oi) and oi > 0:
                for arr in (longs, shorts):
                    s_ = arr.sum()
                    if s_ > oi:
                        arr *= oi / s_
        if np.isfinite(oi):
            prev = oi
        if ((t[i] + H) % (4 * H)) != 0 or t[i] < FROM or i + FWD >= n:
            continue
        px = c[i]
        mcap = cap * px / c_now
        ip = gi(px)
        hh, ll = h[i + 1:i + 1 + FWD], l[i + 1:i + 1 + FWD]
        row = {"symbol": sym, "t": int((t[i] + H) // 1000), "капа $M": mcap / 1e6,
               "OI$/капа %": OIU.at(t[i] + H, 2 * H) / mcap * 100,
               "оборот 24ч / капа %": qv[max(0, i - 23):i + 1].sum() / mcap * 100}
        for name, (longs, shorts, _) in maps.items():
            su, lu = shorts * lvl, longs * lvl
            for R in RANGES:
                row[f"{name} вверх{R} %капы"] = su[ip + 1:gi(px * (1 + R / 100)) + 1].sum() / mcap * 100
                row[f"{name} вниз{R} %капы"] = lu[gi(px * (1 - R / 100)):ip].sum() / mcap * 100
        for d_ in (3, 5, 10):
            iu = np.nonzero(hh >= px * (1 + d_ / 100))[0]
            idn = np.nonzero(ll <= px * (1 - d_ / 100))[0]
            iu = iu[0] if len(iu) else 10 ** 6
            idn = idn[0] if len(idn) else 10 ** 6
            row[f"±{d_}%"] = "вверх" if iu < idn else "вниз" if idn < iu else ("никуда" if iu == 10 ** 6 else "обе")
        row["ret72"] = (c[i + 72] / px - 1) * 100
        rows.append(row)
    # снимок «сейчас» — для сверки с Coinglass (суммы в $)
    px, ip = c[-1], gi(c[-1])
    snap = {"symbol": sym, "t": int((t[-1] + H) // 1000), "снимок": True, "цена": px, "капа $M": cap / 1e6}
    for name, (longs, shorts, _) in maps.items():
        su, lu = shorts * lvl, longs * lvl
        for R in (10, 20, 30, 50):
            snap[f"{name} вверх{R} $"] = su[ip + 1:gi(px * (1 + R / 100)) + 1].sum()
            snap[f"{name} вниз{R} $"] = lu[gi(px * (1 - R / 100)):ip].sum()
    rows.append(snap)
    return pd.DataFrame(rows)


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    uni = {s: caps.get(Hst.base_asset(s)) for s in syms}
    uni = {s: v for s, v in uni.items() if v and 10e6 <= v <= 2e9}
    json.dump(uni, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(uni))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return coin(s, uni[s])
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(2) as ex:
        parts = [p for p in ex.map(work, mine) if p is not None and len(p)]
    (pd.concat(parts) if parts else pd.DataFrame()).to_csv(f"{OUT}/parts/obs{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(parts)} из {len(mine)}")


def outcome_row(g, label):
    r = {"группа": label, "наблюдений": len(g)}
    for d_ in (5, 10):
        up, dn = (g[f"±{d_}%"] == "вверх").mean(), (g[f"±{d_}%"] == "вниз").mean()
        r[f"±{d_}%: вверх"] = f"{up * 100:.0f}%"
        r[f"±{d_}%: вниз"] = f"{dn * 100:.0f}%"
        r[f"±{d_}%: перевес вверх, п.п."] = f"{(up - dn) * 100:+.0f}"
    for y, gy in g.groupby("год"):
        if len(gy) >= 50:
            r[f"{y} (±5%)"] = f"{((gy['±5%'] == 'вверх').mean() - (gy['±5%'] == 'вниз').mean()) * 100:+.0f}"
    return r


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    a = pd.concat(parts, ignore_index=True)
    snap = a[a.get("снимок", pd.Series(False, index=a.index)).fillna(False).astype(bool)]
    snap.dropna(axis=1, how="all").to_csv(f"{OUT}/snapshot_now.csv", index=False)
    a = a[~a.index.isin(snap.index)].copy()
    a["год"] = pd.to_datetime(a.t, unit="s").dt.year
    a.to_csv(f"{OUT}/observations.csv.gz", index=False)
    w = open(f"{OUT}/liq_cap_summary.md", "w").write
    w("# Сумма ликвидаций выше/ниже цены относительно капы: куда пойдёт цена\n\n" + __doc__.split("\n", 2)[2] + "\n\n")
    w(f"Монет: {a.symbol.nunique()}, наблюдений (4h-свечей): {len(a)}.\n\n")
    w("База: " + "; ".join(f"±{d_}%: вверх {(a[f'±{d_}%'] == 'вверх').mean() * 100:.0f}%, вниз {(a[f'±{d_}%'] == 'вниз').mean() * 100:.0f}%"
                          for d_ in (5, 10)) + ". «Перевес вверх» = вверх − вниз (п.п.); сравнивать с базой.\n\n")
    edges = [0, 0.5, 1, 2, 3, 5, 10, 1e9]
    labs = ["<0,5%", "0,5–1%", "1–2%", "2–3%", "3–5%", "5–10%", "10%+"]
    capb = pd.cut(a["капа $M"], [0, 50, 100, 300, 1000, 1e9], labels=["<$50M", "$50–100M", "$100–300M", "$300M–1B", ">$1B"])
    for scale, sname in ((1.0, "сырая модель"), (SCALE, f"÷{SCALE:g}, масштаб реальных ликвидаций")):
        w(f"# Шкала: {sname}\n\n")
        for hl in HL:
            for R in (10, 20):
                up, dn = a[f"{hl} вверх{R} %капы"] / scale, a[f"{hl} вниз{R} %капы"] / scale
                w(f"## Карта «{hl}», сумма в пределах ±{R}% от цены\n\n")
                rows = []
                # 1) жирнее сверху: сверху ≥ X% капы и сверху больше, чем снизу
                for x0, x1, lab in zip(edges[:-1], edges[1:], labs):
                    m = (up >= x0) & (up < x1) & (up > dn)
                    if m.sum() >= 100:
                        rows.append(outcome_row(a[m], f"сверху {lab} капы, сверху > снизу"))
                for x0, x1, lab in zip(edges[:-1], edges[1:], labs):
                    m = (dn >= x0) & (dn < x1) & (dn > up)
                    if m.sum() >= 100:
                        rows.append(outcome_row(a[m], f"снизу {lab} капы, снизу > сверху"))
                # 2) насколько сверху жирнее, чем снизу, при сверху ≥ 1% капы
                ratio = up / dn.where(dn > 0)
                for r0, r1 in ((1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 1e9)):
                    m = (up >= 1) & (ratio >= r0) & (ratio < r1)
                    if m.sum() >= 100:
                        rows.append(outcome_row(a[m], f"сверху ≥1% капы, сверху в {r0}–{r1 if r1 < 1e9 else '∞'} раз больше"))
                    m = (dn >= 1) & (1 / ratio >= r0) & (1 / ratio < r1)
                    if m.sum() >= 100:
                        rows.append(outcome_row(a[m], f"снизу ≥1% капы, снизу в {r0}–{r1 if r1 < 1e9 else '∞'} раз больше"))
                w(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
        # по капе: сверху ≥ 1% и в 2+ раза больше
        hl = "14д"
        up, dn = a[f"{hl} вверх20 %капы"] / scale, a[f"{hl} вниз20 %капы"] / scale
        rows = []
        for cb in capb.cat.categories:
            m = (capb == cb) & (up >= 1) & (up >= 2 * dn)
            if m.sum() >= 50:
                rows.append(outcome_row(a[m], f"{cb}: сверху ≥1% капы и в 2+ раза больше"))
            m = (capb == cb) & (dn >= 1) & (dn >= 2 * up)
            if m.sum() >= 50:
                rows.append(outcome_row(a[m], f"{cb}: снизу ≥1% капы и в 2+ раза больше"))
        w(f"## По размеру капы (карта «14д», ±20%)\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/liq_cap_summary.md").read())


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
