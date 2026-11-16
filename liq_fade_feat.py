"""
Лонг после выноса лонгов (часы): что общего у удачных сделок. Только история, живых ботов не трогает.

Берём сигналы из liq_fade (часовая красная свеча, лонгов вынесло ≥ 3× больше шортов и ≥ $20k,
вынос ≥ 0,5% OI; одна монета — не чаще раза в 24 ч), вход по открытию следующего часа.
Удачная сделка = тейк/стоп 10/10 закрылся в плюс. Для каждого сигнала считаем признаки — только по данным
ДО входа (без заглядывания в будущее):
  размер выноса: % OI, x нормы монеты, $;
  свеча: падение, где закрылась внутри свечи (0 = на минимуме, 1 = на максимуме — «выкупили хвост»), размах;
  объём: оборот свечи / средний часовой за 24 ч до, оборот в $;
  что было до: изменение цены за 24 ч / 7 дней / 30 дней, расстояние от максимума и минимума 30 дней;
  OI: изменение OI за саму свечу (сколько позиций закрыто), изменение OI за 24 ч до;
  толпа: доля аккаунтов в лонгах (L/S все), позиции топ-трейдеров (L/S топы), тейкеры (покупки/продажи);
  фандинг: последний перед сигналом, % за 8 ч;
  рынок: BTC за этот час и за 24 ч; сколько монет в этот же час тоже выносило в лонгах (общий слив рынка);
  монета: капитализация (оценка: текущая CoinGecko × цена тогда / цена сейчас), есть ли спот на Binance,
          возраст фьючерса, сколько похожих выносов было по монете за 7 дней до, час суток (Киев), день недели.
Проверка без самообмана: признаки подбираем на июне–сентябре, проверяем на октябре–ноябре (их «не видели»).
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree

import numpy as np
import pandas as pd

import history as Hst
import liq_fade as F
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/liq_fade_feat")
SRC = "results/liq_fade"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H = F.H
MIN_OI = 0.5
SPLIT = pd.Timestamp("2026-10-01", tz="UTC").value // 10 ** 6       # до — подбор, после — проверка
RES = "10/10"


def signals():
    e = pd.read_csv(f"{SRC}/events.csv.gz")
    e = e[(e.tf == "h") & (e.side == 1) & (e["% OI"] >= MIN_OI)].sort_values("t")
    return F.cooldown(e, 24).reset_index(drop=True)


def spot_bases():
    prefix, out, marker = "data/spot/monthly/klines/", set(), ""
    while True:
        r = Hst.get(Hst.S3, params={"prefix": prefix, "delimiter": "/", "marker": marker})
        root = ElementTree.fromstring(r.content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        ps = root.findall("s:CommonPrefixes/s:Prefix", ns)
        for p in ps:
            s = p.text[len(prefix):].strip("/")
            if s.endswith("USDT"):
                out.add(s[:-4])
        if root.find("s:IsTruncated", ns).text != "true" or not ps:
            break
        marker = ps[-1].text
    return out


def prep():
    os.makedirs(OUT, exist_ok=True)
    caps = Hst.coingecko_mcaps()
    spot = spot_bases()
    syms = sorted(signals().symbol.unique())
    meta = {s: {"cap": caps.get(Hst.base_asset(s), np.nan), "spot": Hst.base_asset(s) in spot} for s in syms}
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print("монет:", len(syms), "с капой:", sum(np.isfinite(m["cap"]) for m in meta.values()),
          "со спотом:", sum(m["spot"] for m in meta.values()))


# ---------------- признаки ----------------
D0 = dt.date(2026, 4, 1)
_btc = {}


def btc():
    if "k" not in _btc:
        k = PL.klines_any("um", "BTCUSDT", D0, dt.date.today())
        _btc["t"], _btc["c"], _btc["o"] = k.open_time.to_numpy().astype("int64"), k.close.to_numpy(), k.open.to_numpy()
        _btc["k"] = True
    return _btc


def coin(sym, sig, meta, mkt):
    k = PL.klines_any("um", sym, D0, dt.date.today())
    if k is None or len(k) < 200:
        return []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c, qv = (k[x].to_numpy() for x in ("open", "high", "low", "close", "quote_volume"))
    days = set()
    for ts in sig.t:
        d = pd.Timestamp(ts, unit="ms", tz="UTC").date()
        days |= {d - dt.timedelta(days=1), d, d + dt.timedelta(days=1)}
    met = PL.metrics(sym, days)
    OI, GL, TP, TK = (PL.AsOf(met, x) for x in ("oi_usd", "glob", "top_pos", "taker"))
    fu = PL.funding(sym, dt.date(2026, 5, 1))
    FU = PL.AsOf(fu, "rate8") if fu is not None else PL.AsOf(None, "rate8")
    B = btc()
    allsig = mkt["own"].get(sym, np.array([], dtype="int64"))
    m = meta.get(sym, {})
    rows = []
    for _, s in sig.iterrows():
        i = int(np.searchsorted(t, s.t))
        if i >= len(t) or t[i] != s.t or i < 168:
            continue
        before = slice(i - 24, i)
        rng = h[i] - l[i]
        oi0, oi1, oi24 = OI.at(t[i], 2 * H), OI.at(t[i] + H, 2 * H), OI.at(t[i] - 24 * H, 2 * H)
        bi = int(np.searchsorted(B["t"], t[i]))
        bok = bi < len(B["t"]) and B["t"][bi] == t[i] and bi >= 24
        j30 = max(0, i - 720)
        hi30, lo30 = h[j30:i].max(), l[j30:i].min()
        full30 = i >= 720
        rows.append({
            "symbol": sym, "t": int(s.t),
            "% OI": s["% OI"], "x нормы": s["x нормы"], "вынос $": s["liq $"], "OI $": s["OI $"],
            "падение свечи, %": (c[i] / o[i] - 1) * 100,
            "закрытие в свече (0=низ)": (c[i] - l[i]) / rng if rng > 0 else np.nan,
            "размах свечи, %": rng / o[i] * 100,
            "объём x 24ч": qv[i] / qv[before].mean() if qv[before].mean() > 0 else np.nan,
            "оборот свечи $": qv[i],
            "цена 24ч до, %": (o[i] / o[i - 24] - 1) * 100,
            "цена 7д до, %": (o[i] / o[i - 168] - 1) * 100,
            "цена 30д до, %": (o[i] / o[i - 720] - 1) * 100 if full30 else np.nan,
            "от хая 30д, %": (c[i] / hi30 - 1) * 100 if full30 else np.nan,
            "от лоя 30д, %": (c[i] / lo30 - 1) * 100 if full30 else np.nan,
            "дней торгов (с апреля)": (t[i] - t[0]) / (24 * H),
            "OI за свечу, %": (oi1 / oi0 - 1) * 100 if oi0 and oi1 else np.nan,
            "OI за 24ч до, %": (oi0 / oi24 - 1) * 100 if oi0 and oi24 else np.nan,
            "L/S все": GL.at(t[i] + H, 2 * H),
            "L/S топы": TP.at(t[i] + H, 2 * H),
            "тейкеры покуп/прод": TK.at(t[i] + H, 2 * H),
            "фандинг %/8ч": FU.at(t[i] + H, 9 * H),
            "BTC за час, %": (B["c"][bi] / B["o"][bi] - 1) * 100 if bok else np.nan,
            "BTC за 24ч, %": (B["c"][bi] / B["o"][bi - 23] - 1) * 100 if bok else np.nan,
            "монет с выносом лонгов в этот час": mkt["count"].get(int(s.t), 0),
            "вынос лонгов по рынку $M": mkt["total"].get(int(s.t), 0) / 1e6,
            "капа $M": m.get("cap", np.nan) * c[i] / c[-1] / 1e6 if np.isfinite(m.get("cap", np.nan)) else np.nan,
            "спот на Binance": float(m.get("spot", False)),
            "выносов у монеты за 7д": int(((allsig < s.t) & (allsig >= s.t - 168 * H)).sum()),
            "час (Киев)": pd.Timestamp(s.t, unit="ms", tz="UTC").tz_convert(Hst.KYIV).hour,
            "день недели (0=пн)": pd.Timestamp(s.t, unit="ms", tz="UTC").tz_convert(Hst.KYIV).weekday(),
            **{x: s[x] for x in ("10/10", "20/10", "10/20", "24ч", "72ч", "против, %")},
        })
    return rows


def market():
    lh = pd.read_csv(f"{SRC}/liq_h.csv.gz")
    lh["t"] = lh.t.astype("int64") * 1000
    lh["dom"] = (lh.liq_long >= 20000) & (lh.liq_long >= 3 * lh.liq_short)
    count = lh[lh.dom].groupby("t").size().to_dict()
    total = lh.groupby("t").liq_long.sum().to_dict()
    own = {s: g.t.to_numpy() for s, g in lh[lh.dom].groupby("symbol")}
    return {"count": count, "total": total, "own": own}


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    sig = signals()
    meta = json.load(open(f"{OUT}/meta.json"))
    mkt = market()
    syms = sorted(sig.symbol.unique())
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    btc()

    def work(s):
        try:
            return coin(s, sig[sig.symbol == s], meta, mkt)
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return []

    rows = []
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(work, mine):
            rows += r
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/f{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(rows)} сигналов")


# ---------------- анализ ----------------
OUTC = ["10/10", "20/10", "10/20", "24ч", "72ч", "против, %"]


def summ(g):
    r = g[RES]
    return {"сигналов": len(g), "винрейт 10/10, %": (r > 0).mean() * 100, "средняя 10/10, %": r.mean(),
            "средняя 72ч, %": g["72ч"].mean()}


def combine():
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        try:
            parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
        except pd.errors.EmptyDataError:
            pass
    d = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    d["время (Киев)"] = d.t.map(PL.kyiv)
    d.to_csv(f"{OUT}/features.csv.gz", index=False)
    feats = [x for x in d.columns if x not in OUTC + ["symbol", "t", "время (Киев)"]]
    A, B = d[d.t < SPLIT], d[d.t >= SPLIT]
    w = open(f"{OUT}/liq_fade_feat_summary.md", "w").write
    w("# Лонг после выноса лонгов: что общего у удачных сделок\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w(f"Сигналов: {len(d)} (подбор, июнь–сентябрь: {len(A)}; проверка, октябрь–ноябрь: {len(B)}). Монет: {d.symbol.nunique()}.\n\n")
    w("Все сигналы: " + ", ".join(f"{k} {v:.2f}" for k, v in summ(d).items()) + "\n\n")

    # 1. удачные против неудачных
    win, loss = d[d[RES] > 0], d[d[RES] <= 0]
    rows = []
    for f in feats:
        rows.append({"признак": f, "удачные (медиана)": win[f].median(), "неудачные (медиана)": loss[f].median(),
                     "заполнено, %": d[f].notna().mean() * 100})
    w("## 1. Удачные (10/10 в плюс) против неудачных — медианы признаков\n\n")
    w(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n\n")

    # 2. по пятым частям: подбор и проверка
    w("## 2. Каждый признак по пятым частям (от меньших значений к большим)\n\n"
      "Слева — подбор (июнь–сентябрь), справа — проверка (октябрь–ноябрь, те же границы). "
      "Признак полезен, если лучшая часть хороша И там, И там.\n\n")
    score = []
    for f in feats:
        x = A[f].dropna()
        if x.nunique() < 2:
            continue
        if x.nunique() <= 6:
            ga, gb = A.groupby(f), B.groupby(f)
        else:
            qs = np.unique(np.nanquantile(x, [0, .2, .4, .6, .8, 1]))
            qs[0], qs[-1] = -np.inf, np.inf
            ca, cb = pd.cut(A[f], qs), pd.cut(B[f], qs)
            ga, gb = A.groupby(ca, observed=True), B.groupby(cb, observed=True)
        ta = {k: summ(g) for k, g in ga if len(g) >= 10}
        tb = {k: summ(g) for k, g in gb if len(g) >= 5}
        rows = []
        for k in ta:
            rows.append({"значения": str(k), "подбор: сигналов": ta[k]["сигналов"], "подбор: винрейт": ta[k]["винрейт 10/10, %"],
                         "подбор: средняя": ta[k]["средняя 10/10, %"],
                         "проверка: сигналов": tb.get(k, {}).get("сигналов", 0),
                         "проверка: винрейт": tb.get(k, {}).get("винрейт 10/10, %", np.nan),
                         "проверка: средняя": tb.get(k, {}).get("средняя 10/10, %", np.nan)})
        t_ = pd.DataFrame(rows)
        if not len(t_):
            continue
        best = t_.loc[t_["подбор: средняя"].idxmax()]
        score.append({"признак": f, "лучшая часть (подбор)": best.значения, "подбор: средняя": best["подбор: средняя"],
                      "проверка: средняя": best["проверка: средняя"], "проверка: сигналов": best["проверка: сигналов"],
                      "разброс между частями (подбор)": t_["подбор: средняя"].max() - t_["подбор: средняя"].min()})
        w(f"### {f}\n\n" + t_.round(2).to_markdown(index=False) + "\n\n")
    sc = pd.DataFrame(score).sort_values("разброс между частями (подбор)", ascending=False)
    w("## 3. Сводка: у каких признаков есть сила (лучшая часть на подборе → что она дала на проверке)\n\n")
    w(f"Для сравнения — все сигналы проверки: средняя {B[RES].mean():.2f}%, винрейт {(B[RES] > 0).mean() * 100:.0f}%.\n\n")
    w(sc.round(2).to_markdown(index=False) + "\n\n")

    # 4. пары условий: каждое условие — «выше/ниже медианы подбора», выбираем на подборе, смотрим на проверке
    conds = []
    for f in feats:
        x = A[f].dropna()
        if x.nunique() < 2:
            continue
        med = x.median()
        if x.nunique() == 2:
            conds += [(f, f"{f} = {v:g}", (lambda df, f=f, v=v: df[f] == v)) for v in sorted(x.unique())]
        else:
            conds += [(f, f"{f} > {med:.3g}", (lambda df, f=f, m=med: df[f] > m)),
                      (f, f"{f} ≤ {med:.3g}", (lambda df, f=f, m=med: df[f] <= m))]
    pairs = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            (f1, n1, c1), (f2, n2, c2) = conds[i], conds[j]
            if f1 == f2:
                continue
            ga = A[c1(A) & c2(A)]
            if len(ga) < 60:
                continue
            gb = B[c1(B) & c2(B)]
            pairs.append({"условия": f"{n1}  И  {n2}", "подбор: сигналов": len(ga),
                          "подбор: винрейт": (ga[RES] > 0).mean() * 100, "подбор: средняя": ga[RES].mean(),
                          "проверка: сигналов": len(gb), "проверка: винрейт": (gb[RES] > 0).mean() * 100 if len(gb) else np.nan,
                          "проверка: средняя": gb[RES].mean() if len(gb) else np.nan})
    pp = pd.DataFrame(pairs).sort_values("подбор: средняя", ascending=False)
    w("## 4. Пары условий: 20 лучших на подборе и как они прошли проверку\n\n")
    w(pp.head(20).round(2).to_markdown(index=False) + "\n\n")
    pp.to_csv(f"{OUT}/pairs.csv", index=False)

    # 5. примеры для глаз
    cols = ["время (Киев)", "symbol", "% OI", "падение свечи, %", "закрытие в свече (0=низ)", "объём x 24ч",
            "цена 7д до, %", "от хая 30д, %", "капа $M", "10/10", "72ч", "против, %"]
    w("## 5. Примеры, чтобы посмотреть на графике (вход — открытие следующего часа после указанного)\n\n")
    w("15 лучших (по 72ч):\n\n" + d.nlargest(15, "72ч")[cols].round(2).to_markdown(index=False) + "\n\n")
    w("15 худших (по 72ч):\n\n" + d.nsmallest(15, "72ч")[cols].round(2).to_markdown(index=False) + "\n\n")
    print(open(f"{OUT}/liq_fade_feat_summary.md").read()[:6000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
