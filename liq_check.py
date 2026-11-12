"""
Честная сверка: мои модельные ликвидации (liq_map.py) против реальных ликвидаций Binance (Coinalyze).
Живого бота и liq_map.py не трогает.

Модель — та же, что в liq_map.py (плечи, сгорание, ограничение реальным OI, полураспад 14 дней).
Сгоревшие позиции переводятся в доллары: контракты × цена ликвидации.
Реальные данные: Coinalyze liquidation-history, только Binance (символ XXXUSDT_PERP.A), в долларах.
  • по дням — с 2025-01-01;  • по часам — сколько Coinalyze хранит (примерно последние 2–3 месяца).
Для честности есть «тупая» замена модели: OI$ × (насколько цена ушла вниз/вверх от открытия за период).
Если модель не лучше тупой замены — модель ничего не добавляет.

Важно: Binance с 2021 года публикует не все ликвидации (в потоке — не больше одной в секунду на монету),
поэтому реальные цифры занижены. Поэтому смотрим не только на суммы, но и на согласованность (корреляцию),
и на то, совпадает ли выбор монет: где сегодня ликвидаций больше всего.
"""

import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-11-01")

import numpy as np
import pandas as pd
import requests

import history as Hst
import liq_map as LM
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_check")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
KEY = os.environ.get("COINALYZE_KEY", "")
H = PL.H
FROM = pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000
WARM = FROM - 45 * 24 * H


# ---------------- Coinalyze (реальные ликвидации) ----------------
def cz_liq(syms, interval, frm, to):
    out = {}
    for i in range(0, len(syms), 10):
        batch = syms[i:i + 10]
        for _ in range(8):
            time.sleep(3)                                  # бережём общий ключ (живой бот)
            r = requests.get("https://api.coinalyze.net/v1/liquidation-history", headers={"api_key": KEY}, timeout=60,
                             params={"symbols": ",".join(f"{s}_PERP.A" for s in batch), "interval": interval,
                                     "from": int(frm), "to": int(to), "convert_to_usd": "true"})
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)) + 2)
                continue
            if r.status_code != 200:
                print("Coinalyze", r.status_code, r.text[:200])
                break
            for item in r.json():
                h = pd.DataFrame(item.get("history") or [])
                if len(h):
                    out[item["symbol"].replace("_PERP.A", "")] = h.rename(columns={"l": "real_long", "s": "real_short"})
            break
    return out


def fetch_real():
    os.makedirs(OUT, exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    now = int(time.time())
    rows = []
    for interval, frm in (("daily", FROM / 1000), ("1hour", now - 120 * 86400)):
        got = cz_liq(syms, interval, frm, now)
        for s, h in got.items():
            h["symbol"], h["interval"] = s, interval
            rows.append(h)
        print(interval, "монет с данными:", len(got))
    pd.concat(rows).to_csv(f"{OUT}/real.csv.gz", index=False)


# ---------------- модель по часам ----------------
def model(sym):
    k = Hst.load_klines(sym, "1h")
    if k is None or len(k) < 1000:
        return None
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    v, qv, t = k.volume.to_numpy(), k.quote_volume.to_numpy(), k.open_time.to_numpy()
    n = len(k)
    days, d = set(), pd.Timestamp(WARM, unit="ms", tz="UTC").date()
    while d < dt.datetime.now(dt.timezone.utc).date():
        days.add(d)
        d += dt.timedelta(days=1)
    met = PL.metrics(sym, days)
    if met is None or len(met) < 500:
        return None
    OI, OIU = PL.AsOf(met, "oi"), PL.AsOf(met, "oi_usd")
    lo_g = math.log(np.nanmin(l[l > 0]) * 0.3)
    G = int((math.log(np.nanmax(h) * 3) - lo_g) / LM.STEP) + 2
    longs, shorts = np.zeros(G), np.zeros(G)
    lvl = np.exp(lo_g + (np.arange(G) + 0.5) * LM.STEP)          # цена уровня
    gi = lambda p: min(G - 1, max(0, int((math.log(p) - lo_g) / LM.STEP)))
    prev, rows = np.nan, []
    for i in range(int(np.searchsorted(t, WARM)), n):
        if not (h[i] > 0 and l[i] > 0):
            continue
        oi = OI.at(t[i] + H, 2 * H)
        longs *= LM.DECAY
        shorts *= LM.DECAY
        ih, il = gi(h[i]), gi(l[i])
        bs = float((shorts[:ih + 1] * lvl[:ih + 1]).sum())
        bl = float((longs[il:] * lvl[il:]).sum())
        shorts[:ih + 1] = 0
        longs[il:] = 0
        if np.isfinite(oi) and np.isfinite(prev) and oi > prev and v[i] > 0:
            p = qv[i] / v[i]
            for L, wgt in LM.LEV.items():
                longs[gi(p * (1 - 1 / L + LM.MMR))] += (oi - prev) * wgt
                shorts[gi(p * (1 + 1 / L - LM.MMR))] += (oi - prev) * wgt
        if np.isfinite(oi) and oi > 0:
            for arr in (longs, shorts):
                s = arr.sum()
                if s > oi:
                    arr *= oi / s
            prev = oi
        if t[i] >= FROM:
            oiu = OIU.at(t[i], 2 * H)
            rows.append({"symbol": sym, "t": int(t[i] // 1000), "model_long": bl, "model_short": bs,
                         "naive_long": oiu * max(0.0, 1 - l[i] / o[i]) if np.isfinite(oiu) else np.nan,
                         "naive_short": oiu * max(0.0, h[i] / o[i] - 1) if np.isfinite(oiu) else np.nan,
                         "oi_usd": oiu, "qv": qv[i]})
    return pd.DataFrame(rows)


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(syms)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            return model(s)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(2) as ex:
        parts = [p for p in ex.map(work, mine) if p is not None and len(p)]
    (pd.concat(parts) if parts else pd.DataFrame()).to_csv(f"{OUT}/parts/model{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(parts)} из {len(mine)}")


# ---------------- сверка ----------------
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()) if m.sum() >= 20 else np.nan   # Спирмен без scipy


def compare(m, r, label):
    """m — модель, r — реальные, одинаковая сетка времени (t в секундах)."""
    a = m.merge(r[["symbol", "t", "real_long", "real_short"]], on=["symbol", "t"], how="inner")
    a = a.dropna(subset=["real_long", "real_short"])
    for x in ("model", "naive", "real"):
        a[f"{x}_tot"] = a[f"{x}_long"] + a[f"{x}_short"]
        a[f"{x}_lshare"] = a[f"{x}_long"] / a[f"{x}_tot"].where(a[f"{x}_tot"] > 0)
    rows = []
    for s, g in a.groupby("symbol"):
        r_ = {"монета": s, "точек": len(g),
              "реально $M": g.real_tot.sum() / 1e6, "модель $M": g.model_tot.sum() / 1e6}
        for x in ("model", "naive"):
            r_[f"{x}: связь по сумме"] = corr(g[f"{x}_tot"].to_numpy(), g.real_tot.to_numpy())
            r_[f"{x}: связь по стороне"] = corr(g[f"{x}_lshare"].to_numpy(), g.real_lshare.to_numpy())
        rows.append(r_)
    per = pd.DataFrame(rows)
    # выбор монет: в каждый момент — ранжируем монеты, совпадает ли топ
    picks = []
    for t, g in a.groupby("t"):
        if len(g) < 20:
            continue
        g = g.assign(real_rel=g.real_tot / g.oi_usd, model_rel=g.model_tot / g.oi_usd, naive_rel=g.naive_tot / g.oi_usd)
        top_real = set(g.nlargest(10, "real_tot").symbol)
        top_real_rel = set(g.nlargest(10, "real_rel").symbol)
        picks.append({"t": t,
                      "модель: связь по $": corr(g.model_tot.to_numpy(), g.real_tot.to_numpy()),
                      "тупая: связь по $": corr(g.naive_tot.to_numpy(), g.real_tot.to_numpy()),
                      "модель: топ-10 по $ совпал": len(top_real & set(g.nlargest(10, "model_tot").symbol)) / 10,
                      "тупая: топ-10 по $ совпал": len(top_real & set(g.nlargest(10, "naive_tot").symbol)) / 10,
                      "модель: топ-10 по доле OI совпал": len(top_real_rel & set(g.nlargest(10, "model_rel").symbol)) / 10,
                      "тупая: топ-10 по доле OI совпал": len(top_real_rel & set(g.nlargest(10, "naive_rel").symbol)) / 10})
    picks = pd.DataFrame(picks)
    # крупные события: реальные топ-5% по монете — поймала ли модель (её топ-10%)
    big = []
    for s, g in a.groupby("symbol"):
        if len(g) < 50:
            continue
        rb = g.real_tot >= g.real_tot.quantile(0.95)
        for x in ("model", "naive"):
            mb = g[f"{x}_tot"] >= g[f"{x}_tot"].quantile(0.90)
            big.append({"монета": s, "кто": x, "поймано": (rb & mb).sum() / max(1, rb.sum()),
                        "ложных": (mb & ~rb).sum() / max(1, mb.sum())})
    big = pd.DataFrame(big)
    w = open(f"{OUT}/liq_check_summary.md", "a").write
    w(f"## {label}: {a.symbol.nunique()} монет, {len(a)} точек\n\n")
    w(f"Итого реальных ликвидаций (Coinalyze, Binance): ${a.real_tot.sum() / 1e9:.2f} млрд; "
      f"модель: ${a.model_tot.sum() / 1e9:.2f} млрд (модель / реальность = {a.model_tot.sum() / a.real_tot.sum():.1f}x).\n\n")
    w("Связь (ранговая корреляция, медиана по монетам; 1 = идеально, 0 = никакой):\n\n")
    w(f"| что | модель | тупая замена |\n|---|---|---|\n"
      f"| сколько ликвидаций в период | {per['model: связь по сумме'].median():.2f} | {per['naive: связь по сумме'].median():.2f} |\n"
      f"| какая сторона (лонги или шорты) | {per['model: связь по стороне'].median():.2f} | {per['naive: связь по стороне'].median():.2f} |\n\n")
    if len(picks):
        w("Выбор монет (в каждый момент сравниваем все монеты между собой):\n\n")
        w("| что | модель | тупая замена |\n|---|---|---|\n")
        for lbl in ("связь по $", "топ-10 по $ совпал", "топ-10 по доле OI совпал"):
            w(f"| {lbl} | {picks[f'модель: {lbl}'].mean():.2f} | {picks[f'тупая: {lbl}'].mean():.2f} |\n")
        w("\n(для «топ-10 совпал»: доля совпавших монет; случайный выбор дал бы ≈ 10 / число монет)\n\n")
    if len(big):
        bb = big.groupby("кто")[["поймано", "ложных"]].mean()
        w("Крупные реальные выносы (топ-5% по монете): сколько из них модель отметила как свои топ-10%:\n\n")
        w(f"| | поймано | ложных тревог |\n|---|---|---|\n"
          f"| модель | {bb.loc['model', 'поймано'] * 100:.0f}% | {bb.loc['model', 'ложных'] * 100:.0f}% |\n"
          f"| тупая замена | {bb.loc['naive', 'поймано'] * 100:.0f}% | {bb.loc['naive', 'ложных'] * 100:.0f}% |\n\n")
    w("По монетам:\n\n" + per.round(2).sort_values("реально $M", ascending=False).to_markdown(index=False) + "\n\n")
    return a


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    m = pd.concat(parts, ignore_index=True)
    r = pd.read_csv(f"{OUT}/real.csv.gz")
    open(f"{OUT}/liq_check_summary.md", "w").write("# Сверка модели ликвидаций с реальными (Coinalyze, Binance)\n\n"
                                                   + __doc__.split("\n", 2)[2] + "\n\n")
    rh = r[r.interval == "1hour"].copy()
    if len(rh):
        hourly = compare(m, rh, f"По часам ({pd.Timestamp(rh.t.min(), unit='s'):%Y-%m-%d} — {pd.Timestamp(rh.t.max(), unit='s'):%Y-%m-%d})")
        hourly.to_csv(f"{OUT}/hourly.csv.gz", index=False)
    rd = r[r.interval == "daily"].copy()
    md = m.assign(t=(m.t // 86400) * 86400).groupby(["symbol", "t"], as_index=False)[
        ["model_long", "model_short", "naive_long", "naive_short", "qv"]].sum()
    md = md.merge(m.assign(t=(m.t // 86400) * 86400).groupby(["symbol", "t"], as_index=False).oi_usd.mean(), on=["symbol", "t"])
    if len(rd):
        rd["t"] = (rd.t // 86400) * 86400
        daily = compare(md, rd, "По дням (с 2025-01-01)")
        daily.to_csv(f"{OUT}/daily.csv.gz", index=False)
    print(open(f"{OUT}/liq_check_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        os.makedirs(OUT, exist_ok=True)
        uni = json.load(open("results/liq_map/universe.json")) if os.path.exists("results/liq_map/universe.json") else None
        if uni is None:
            LM.OUT = OUT
            LM.prep()
        else:
            json.dump(uni, open(f"{OUT}/universe.json", "w"))
        fetch_real()
    elif mode == "combine":
        combine()
    else:
        run_shard()
