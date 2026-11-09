"""
Эксперимент (не бот): что отличает удачные сигналы «коробка → выход 4h» (+30% за 30 дней) от неудачных.

Сигналы и их итоги — из results/box4h/box4h_signals.csv (прогон Box 4h), только коробки шириной ≤ 25%.
Для каждого сигнала признаки на момент закрытия сигнальной 4h-свечи (известно до входа):
  OI (архив Binance metrics): за коробку (10 дней), за 3 дня, за сигнальную свечу; OI в $ к капе.
  Лонг/шорт по аккаунтам (толпа) и топ-трейдеров по позициям: на сигнале и изменение за коробку.
  Фандинг: средний за коробку и за последние сутки.
  Объём: последние 3 дня коробки к первым 7; максимальный всплеск внутри коробки; коробка к 20 дням до неё;
         объём и дельта сигнальной свечи; накопленная дельта за коробку и за последние 3 дня.
  До коробки (20 дней): изменение цены, ширина диапазона, ход за 3 дня перед коробкой.
  Коробка: сколько свечей касались верха / низа (тело в пределах 2%), где была цена перед пробоем.
Сравнение: доля дошедших до +30% по десяткам каждого признака + медианы у удачных и неудачных.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2024-09-01")
os.environ.setdefault("SIG_FROM", "2025-01-01")

import numpy as np
import pandas as pd

import history as Hst
import oi_funding as OF
import pre_pump as PP

OUT = "results/box4h_feat"
SRC = "results/box4h/box4h_signals.csv"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = 3600 * 1000
H4 = 4 * H
BOX = 60
WMAX = 25
SEED = 21


def kyiv_ms(s):
    return pd.Timestamp(s, tz=Hst.KYIV).tz_convert("UTC").timestamp() * 1000


def signals():
    a = pd.read_csv(SRC)
    a = a[(a["тип"] == "сигнал") & (a["ширина коробки, %"] <= WMAX) & a["30 дней прошло"]].copy()
    loc = pd.to_datetime(a["сигнальная свеча (TV 4Ч)"]).dt.tz_localize(Hst.KYIV, ambiguous="NaT", nonexistent="NaT")
    a = a[loc.notna()].copy()                     # пара свечей в часы перевода времени — пропускаем
    a["t"] = [round(x.timestamp() * 1000) * 1.0 for x in loc[loc.notna()]]
    return a


def coin(sym, ts, btc):
    k = Hst.load_klines(sym, "4h")
    if k is None:
        return []
    o, h, l, c = (k[x].to_numpy() for x in ("open", "high", "low", "close"))
    t, qv, tbq = k.open_time.to_numpy(), k.quote_volume.to_numpy(), k.taker_buy_quote_volume.to_numpy()
    m = PP.metrics(sym, min(ts) - 12 * 86400 * 1000)
    fu = OF.funding(sym)
    rows = []
    for ti in ts:
        hit = np.flatnonzero(t == ti)
        if not len(hit):
            continue
        i = int(hit[0])
        if i < BOX + 120:
            continue
        b = slice(i - BOX, i)
        pre = slice(i - BOX - 120, i - BOX)
        top, bot = np.maximum(o[b], c[b]).max(), np.minimum(o[b], c[b]).min()
        T, T0 = ti + H4, t[i - BOX]
        qb = qv[b]
        r = {"монета": sym, "t": ti,
             "объём: последние 3д коробки к первым 7д": np.median(qb[-18:]) / max(np.median(qb[:42]), 1e-9),
             "объём: макс. всплеск в коробке (к медиане)": qb.max() / max(np.median(qb), 1e-9),
             "объём: коробка к 20 дням до неё": np.median(qb) / max(np.median(qv[pre]), 1e-9),
             "объём сигнальной свечи к медиане коробки": qv[i] / max(np.median(qb), 1e-9),
             "дельта сигнальной свечи, %": (2 * tbq[i] - qv[i]) / qv[i] * 100,
             "накопл. дельта за коробку, %": (2 * tbq[b] - qb).sum() / qb.sum() * 100,
             "накопл. дельта за последние 3д, %": (2 * tbq[i - 18:i] - qv[i - 18:i]).sum() / qv[i - 18:i].sum() * 100,
             "цена за 20 дней до коробки, %": (c[i - BOX - 1] / c[i - BOX - 120] - 1) * 100,
             "диапазон 20 дней до коробки, %": (h[pre].max() / l[pre].min() - 1) * 100,
             "ход за 3 дня перед коробкой, %": (c[i - BOX - 1] / c[i - BOX - 19] - 1) * 100,
             "касаний верха (тела)": int((np.maximum(o[b], c[b]) >= top * 0.98).sum()),
             "касаний низа (тела)": int((np.minimum(o[b], c[b]) <= bot * 1.02).sum()),
             "где цена перед пробоем (0 низ, 100 верх)": (c[i - 1] - bot) / (top - bot) * 100 if top > bot else np.nan,
             "тени: выход за коробку тенями, %": (h[b].max() / l[b].min() - 1) * 100 - (top / bot - 1) * 100}
        if m is not None and len(m):
            mt = m.t.to_numpy()

            def at(col, when, fresh=60 * 60 * 1000):
                v = OF.asof(mt, m[col].to_numpy(), np.array([when], dtype=float))[0]
                tt = OF.asof(mt, mt, np.array([when], dtype=float))[0]
                return v if when - tt <= fresh else np.nan

            oi_T, oi_0, oi_3, oi_s = at("oi", T), at("oi", T0), at("oi", T - 72 * H), at("oi", ti)
            r["OI за коробку, %"] = (oi_T / oi_0 - 1) * 100
            r["OI за 3 дня, %"] = (oi_T / oi_3 - 1) * 100
            r["OI за сигнальную свечу, %"] = (oi_T / oi_s - 1) * 100
            r["OI в $ к капе, %"] = at("oi_usd", T)
            r["толпа л/ш на сигнале"] = at("ls_acc", T)
            r["толпа л/ш: изменение за коробку"] = at("ls_acc", T) - at("ls_acc", T0)
            r["топы л/ш (позиции) на сигнале"] = at("ls_top", T)
            r["топы л/ш: изменение за коробку"] = at("ls_top", T) - at("ls_top", T0)
        if fu is not None and len(fu):
            ft, fr = fu.t.to_numpy(), fu.rate8.to_numpy()
            r["фандинг средний за коробку, %"] = fr[(ft > T0) & (ft <= T)].mean() * 100 if ((ft > T0) & (ft <= T)).any() else np.nan
            r["фандинг за сутки, %"] = fr[(ft > T - 24 * H) & (ft <= T)].mean() * 100 if ((ft > T - 24 * H) & (ft <= T)).any() else np.nan
        bt = btc.get(ti)
        r["BTC за 7 дней, %"] = bt
        rows.append(r)
    return rows


def prep():
    os.makedirs(OUT, exist_ok=True)
    a = signals()
    json.dump({s: sorted(g.t.astype(float).tolist()) for s, g in a.groupby("монета")}, open(f"{OUT}/todo.json", "w"))
    print("сигналов:", len(a), "монет:", a["монета"].nunique())


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    todo = json.load(open(f"{OUT}/todo.json"))
    mine = [s for i, s in enumerate(sorted(todo)) if i % NSHARDS == SHARD]
    b = Hst.load_klines("BTCUSDT", "4h")
    btc = dict(zip(b.open_time.astype(float), (b.close / b.close.shift(42) - 1) * 100))

    def work(sym):
        try:
            return coin(sym, todo[sym], btc)
        except Exception as e:
            print(f"{sym}: {e}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(work, mine), 1):
            rows += r
            if i % 10 == 0:
                print(f"{i}/{len(mine)}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/part{SHARD}.csv.gz", index=False)


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    f = pd.concat(parts, ignore_index=True)
    a = signals().merge(f, left_on=["монета", "t"], right_on=["монета", "t"], how="inner")
    a = a.replace([np.inf, -np.inf], np.nan)
    a["успех"] = a["тейк 30%"].astype(bool)
    base = a["успех"].mean()
    feats = [c for c in f.columns if c not in ("монета", "t")] + ["ширина коробки, %", "доля тела за коробкой, %",
                                                                 "вход над верхом, %", "капа, $M"]
    w = open(f"{OUT}/box4h_feat_summary.md", "w").write
    w("# Что отличает удачные пробои коробки (+30% за 30 дней) от неудачных\n\n"
      f"Сигналы прогона Box 4h, коробка ≤ {WMAX}%, с 2025 года. Всего: {len(a)}, дошли до +30%: {base * 100:.0f}%. "
      "Все признаки — на момент закрытия сигнальной свечи.\n\n")
    rows = []
    for col in feats:
        x = a[col]
        win, lose = x[a["успех"]].median(), x[~a["успех"]].median()
        ok = x.notna()
        best = worst = None
        try:
            bins = pd.qcut(x[ok], 10, duplicates="drop")
            g = a[ok].groupby(bins, observed=True)["успех"].agg(["mean", "size"])
            best, worst = g["mean"].max(), g["mean"].min()
            spread = best - worst
        except ValueError:
            spread = np.nan
        rows.append({"признак": col, "медиана у удачных": round(win, 2) if pd.notna(win) else None,
                     "медиана у неудачных": round(lose, 2) if pd.notna(lose) else None,
                     "лучший десяток: доля +30%": f"{best * 100:.0f}%" if best is not None else "-",
                     "худший десяток": f"{worst * 100:.0f}%" if worst is not None else "-",
                     "_spread": spread, "заполнено": f"{ok.mean() * 100:.0f}%"})
    t = pd.DataFrame(rows).sort_values("_spread", ascending=False)
    w(f"## Признаки по силе (разница между лучшим и худшим десятком; в среднем дошли {base * 100:.0f}%)\n\n"
      + t.drop(columns="_spread").to_markdown(index=False) + "\n\n")
    for col in t["признак"].head(8):
        x = a[col]
        ok = x.notna()
        try:
            bins = pd.qcut(x[ok], 10, duplicates="drop")
        except ValueError:
            continue
        g = a[ok].groupby(bins, observed=True).agg(сигналов=("успех", "size"), дошли=("успех", "mean"),
                                                   **{"2025": ("успех", lambda s: s[a.loc[s.index, "год"] == 2025].mean()),
                                                      "2026": ("успех", lambda s: s[a.loc[s.index, "год"] == 2026].mean())})
        for cc in ("дошли", "2025", "2026"):
            g[cc] = (g[cc] * 100).round(0).astype("Int64").astype(str) + "%"
        g.index = [f"{iv.left:.3g} … {iv.right:.3g}" for iv in g.index]
        w(f"### {col}\n\n" + g.to_markdown() + "\n\n")
    # 5 неудачных (те же, что в чате) и 5 удачных
    chat = [("ENSUSDT", "2026-06-15 11:00"), ("AUCTIONUSDT", "2026-05-07 11:00"), ("HANAUSDT", "2026-03-07 06:00"),
            ("TRBUSDT", "2025-01-03 18:00"), ("SAFEUSDT", "2026-05-22 07:00")]
    fail = pd.concat([a[(a["монета"] == s_) & (a["сигнальная свеча (TV 4Ч)"] == tt)] for s_, tt in chat])
    good = a[a["успех"]].sample(5, random_state=SEED)
    show = ["монета", "сигнальная свеча (TV 4Ч)", "ширина коробки, %"] + [c for c in feats if c in a.columns][:28]
    for name, d in (("Неудачные", fail), ("Удачные", good)):
        w(f"## {name} (5 случайных)\n\n" + d[show].set_index("монета").T.round(2).to_markdown() + "\n\n")
    a.round(4).to_csv(f"{OUT}/box4h_feat_all.csv.gz", index=False)
    print(open(f"{OUT}/box4h_feat_summary.md").read()[:3000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    prep() if mode == "prep" else combine() if mode == "combine" else run_shard()
