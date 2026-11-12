"""
Идея Анастасии: монеты одного нарратива идут одна за другой — одна выстрелила, за ней подтягиваются остальные
из её сектора/экосистемы. Проверяем на истории, для лонгов и шортов. Живого бота не трогает.

Сектора — категории CoinGecko (AI, мемы, экосистемы BNB/Solana/Base/Polkadot, Binance Alpha, RWA, DePIN, L1, L2 ...).
Внимание: состав категорий — сегодняшний (как CoinGecko размечает сейчас).

Событие «лидер пошёл» (лонг): монета сектора выросла ≥ +20% за 24 часа (на закрытии 4h-свечи), а за прошлые 72 часа
у неё такого не было. Остальные монеты сектора, которые ещё НЕ выросли (< +10% за 24ч) — «догоняющие».
Сравниваем догоняющих с «контролем» — монетами НЕ из этого сектора в тот же момент (тоже < +10% за 24ч),
чтобы убрать общий рынок. Если догоняющие чаще пампятся / лучше идут, чем контроль — эффект есть.
Шорты — зеркально: лидер упал ≥ 20% за 24ч, догоняющие (> −10%) — падают ли за ним.
Сделка: вход по закрытию 4h-свечи, тейк/стоп 10/10 и 20/10, выход по времени через 7 дней, комиссия 0.1%.
Если тейк и стоп в одной 4h-свече — считаем стоп (строго).

MODE=prep — монеты и категории; MODE=(пусто) — свечи части SHARD; MODE=combine — анализ.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("START", "2022-12-01")

import numpy as np
import pandas as pd

import history as Hst

OUT = os.environ.get("OUT_DIR", "results/narrative")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
FROM = pd.Timestamp(os.environ.get("FROM", "2023-01-01"), tz="UTC")
MONTH = os.environ.get("MONTH", "2026-10")
LOSERS = os.environ.get("LOSERS", "JCT,MOVR,MUBARAK,ARX,AIN").split(",")
H4 = 4 * 3600 * 1000
D24, D72, D7 = 6, 18, 42          # в 4h-свечах
PUMP, LAG, MINVOL, FEE = 0.20, 0.10, 1e6, 0.001

STOCKS = {"SPYUSDT", "QQQUSDT", "NVDAUSDT", "METAUSDT", "CRCLUSDT", "SPCXUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT",
          "AMZNUSDT", "GOOGLUSDT", "MSTRUSDT", "COINUSDT", "HOODUSDT", "PLTRUSDT", "AMDUSDT", "INTCUSDT", "TSMUSDT",
          "PAXGUSDT", "XAUUSDT", "XAGUSDT", "XAUTUSDT"}

# (название в таблице, id категории CoinGecko, ключевые слова для поиска, если id не нашёлся)
WANT = [
    ("AI", "artificial-intelligence", ["artificial intelligence"]),
    ("AI-агенты", "ai-agents", ["ai agents"]),
    ("Мемы", "meme-token", ["meme"]),
    ("Мемы Solana", "solana-meme-coins", ["solana", "meme"]),
    ("Мемы BNB", "bnb-chain-meme-coins", ["bnb", "meme"]),
    ("Экосистема BNB", "binance-smart-chain", ["bnb chain ecosystem"]),
    ("Binance Alpha", "binance-alpha-spotlight", ["binance alpha"]),
    ("Binance Launchpool", "binance-launchpool", ["launchpool"]),
    ("Binance HODLer", "binance-hodler-airdrops", ["hodler"]),
    ("Экосистема Solana", "solana-ecosystem", ["solana ecosystem"]),
    ("Экосистема Base", "base-ecosystem", ["base ecosystem"]),
    ("Экосистема Polkadot", "dot-ecosystem", ["polkadot"]),
    ("Экосистема Kusama", "kusama-ecosystem", ["kusama"]),
    ("Экосистема Sui", "sui-ecosystem", ["sui ecosystem"]),
    ("Экосистема TON", "ton-ecosystem", ["ton ecosystem"]),
    ("Экосистема Bitcoin", "bitcoin-ecosystem", ["bitcoin ecosystem"]),
    ("Layer 1", "layer-1", ["layer 1"]),
    ("Layer 2", "layer-2", ["layer 2"]),
    ("DeFi", "decentralized-finance-defi", ["decentralized finance"]),
    ("DEX", "decentralized-exchange", ["decentralized exchange"]),
    ("Перпы", "perpetuals", ["perpetuals"]),
    ("Игры", "gaming", ["gaming"]),
    ("RWA", "real-world-assets-rwa", ["real world assets"]),
    ("DePIN", "depin", ["depin"]),
    ("ZK", "zero-knowledge-zk", ["zero knowledge"]),
    ("Приватность", "privacy-coins", ["privacy"]),
    ("Оракулы", "oracle", ["oracle"]),
    ("Хранилища", "storage", ["storage"]),
    ("Биржевые токены", "exchange-based-tokens", ["exchange-based"]),
    ("Лаунчпады", "launchpad", ["launchpad"]),
    ("Инфраструктура", "infrastructure", ["infrastructure"]),
    ("Pump.fun", "pump-fun", ["pump.fun"]),
    ("Made in China", "made-in-china", ["made in china"]),
    ("Made in USA", "made-in-usa", ["made in usa"]),
]


def cg(url, params=None):
    for i in range(8):
        r = Hst.get(url, params=params)
        if r is not None and r.status_code == 200:
            return r.json()
        time.sleep(15 * (i + 1))
    print("CoinGecko не ответил:", url, params, file=sys.stderr)
    return None


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = [s for s in Hst.list_symbols() if s not in STOCKS]
    only = [x for x in os.environ.get("SYMBOLS", "").split(",") if x]
    if only:
        syms = [s for s in syms if s in only]
    bases = {Hst.base_asset(s): s for s in syms}
    caps = Hst.coingecko_mcaps()
    lst = cg("https://api.coingecko.com/api/v3/coins/categories/list") or []
    pd.DataFrame(lst).to_csv(f"{OUT}/coingecko_categories.csv", index=False)
    ids = {c["category_id"]: c["name"] for c in lst}
    cats, log = {}, []
    for label, cid, kw in WANT:
        use = cid if cid in ids else None
        if use is None:
            cand = [k for k, n in ids.items() if all(w in n.lower() for w in kw)]
            use = min(cand, key=lambda k: len(ids[k])) if cand else None
        if use is None:
            log.append(f"{label}: категория не найдена")
            continue
        members = set()
        for page in range(1, 6):
            js = cg("https://api.coingecko.com/api/v3/coins/markets",
                    {"vs_currency": "usd", "category": use, "per_page": 250, "page": page})
            time.sleep(7)
            if not js:
                break
            for c in js:
                b = (c.get("symbol") or "").upper()
                m = c.get("market_cap") or 0
                # тикеры совпадают у разных монет: берём, только если это «та самая» (капа не в разы меньше главной)
                if b in bases and (not caps.get(b) or m >= 0.3 * caps[b]):
                    members.add(bases[b])
            if len(js) < 250:
                break
        cats[label] = sorted(members)
        log.append(f"{label}: {ids.get(use, use)} ({use}) — монет с фьючерсом Binance: {len(members)}")
        print(log[-1])
    json.dump({"symbols": syms, "categories": cats}, open(f"{OUT}/universe.json", "w"), ensure_ascii=False)
    open(f"{OUT}/categories_log.txt", "w").write("\n".join(log) + "\n")


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    uni = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(sorted(uni["symbols"])) if i % NSHARDS == SHARD]

    def work(s):
        try:
            k = Hst.load_klines(s, "4h")
            if k is None or not len(k):
                return None
            k = k[["open_time", "high", "low", "close", "quote_volume"]].astype(float)
            k["sym"] = s
            return k
        except Exception as e:
            print(f"{s}: {e!r}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(work, mine) if p is not None]
    (pd.concat(parts) if parts else pd.DataFrame()).to_csv(f"{OUT}/parts/k{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: {len(parts)} из {len(mine)}")


# ---------------- анализ ----------------
def shift(a, k):
    """a[t+k] на месте t (NaN за концом)."""
    out = np.full_like(a, np.nan)
    out[:-k] = a[k:]
    return out


def trade(C, Hh, Ll, side, tp, sl):
    """Результат сделки из каждой ячейки (t, монета): вход по C[t], тейк/стоп в долях, выход через 7 дней."""
    res = np.full(C.shape, np.nan)
    open_ = np.isfinite(C)
    for k in range(1, D7 + 1):
        hk, lk = shift(Hh, k) / C - 1, shift(Ll, k) / C - 1
        if side > 0:
            stop, take = lk <= -sl, hk >= tp
        else:
            stop, take = hk >= sl, lk <= -tp
        s_ = open_ & stop
        t_ = open_ & take & ~stop
        res[s_], res[t_] = -sl, tp
        open_ &= ~(s_ | t_)
    end = side * (shift(C, D7) / C - 1)
    res[open_] = end[open_]
    return res - FEE


def load_all():
    files = [f"{OUT}/parts/{f}" for f in sorted(os.listdir(f"{OUT}/parts")) if f.endswith(".csv.gz")]
    k = pd.concat([pd.read_csv(f) for f in files if os.path.getsize(f) > 50])
    k = k[k.open_time >= (FROM - pd.Timedelta(days=10)).timestamp() * 1000]
    piv = {c: k.pivot_table(index="open_time", columns="sym", values=c) for c in ("high", "low", "close", "quote_volume")}
    idx = np.arange(piv["close"].index.min(), piv["close"].index.max() + H4, H4)
    piv = {c: v.reindex(idx) for c, v in piv.items()}
    cols = piv["close"].columns
    return idx, list(cols), {c: v[cols].to_numpy(float) for c, v in piv.items()}


def kyiv(ms):
    return pd.Timestamp(ms + H4, unit="ms", tz="UTC").tz_convert("Europe/Kyiv")


def pct(x):
    return f"{x * 100:+.1f}%" if np.isfinite(x) else "—"


def combine():
    uni = json.load(open(f"{OUT}/universe.json"))
    t_ms, syms, M = load_all()
    C, Hh, Ll, QV = M["close"], M["high"], M["low"], M["quote_volume"]
    alive = np.isfinite(C)
    Cf = pd.DataFrame(C).ffill(limit=D7).to_numpy()
    r24 = C / np.vstack([np.full((D24, C.shape[1]), np.nan), Cf[:-D24]]) - 1
    vol24 = pd.DataFrame(QV).rolling(D24, min_periods=D24).sum().to_numpy()
    # максимум/минимум за следующие 72 часа от цены входа
    fmax = pd.DataFrame(Hh[::-1]).rolling(D72, min_periods=1).max().to_numpy()[::-1]
    fmin = pd.DataFrame(Ll[::-1]).rolling(D72, min_periods=1).min().to_numpy()[::-1]
    up72 = np.vstack([fmax[1:], np.full((1, C.shape[1]), np.nan)]) / C - 1
    dn72 = np.vstack([fmin[1:], np.full((1, C.shape[1]), np.nan)]) / C - 1
    ret72 = shift(Cf, D72) / C - 1
    print("сделки…", flush=True)
    TR = {"лонг 10/10": trade(C, Hh, Ll, 1, .10, .10), "лонг 20/10": trade(C, Hh, Ll, 1, .20, .10),
          "шорт 10/10": trade(C, Hh, Ll, -1, .10, .10), "шорт 20/10": trade(C, Hh, Ll, -1, .20, .10)}
    t0 = int(np.searchsorted(t_ms, FROM.timestamp() * 1000))
    col = {s: i for i, s in enumerate(syms)}
    ok = alive & (vol24 >= MINVOL)

    def fresh(sig):
        prev = np.zeros_like(sig)
        for k in range(1, D72 + 1):
            prev[k:] |= sig[:-k]
        return sig & ~prev

    pump_up = fresh(np.nan_to_num(r24, nan=0) >= PUMP) & ok
    pump_dn = fresh(np.nan_to_num(r24, nan=0) <= -PUMP) & ok

    rows, frows, events = [], [], []
    for side, ev_mat in ((1, pump_up), (-1, pump_dn)):
        name = "лонг" if side > 0 else "шорт"
        for label, mem in uni["categories"].items():
            mi = np.array([col[s] for s in mem if s in col], dtype=int)
            if len(mi) < 5:
                continue
            notm = np.setdiff1d(np.arange(len(syms)), mi)
            sub = ev_mat[:, mi]
            for t in range(t0, len(t_ms) - D7):
                hits = np.flatnonzero(sub[t])
                if not len(hits):
                    continue
                lead = mi[hits]
                r_now = r24[t]
                # сколько разных монет сектора уже «выстрелили» за последние 72ч, включая этот
                n72 = int(ev_mat[max(0, t - D72 + 1):t + 1, mi].any(axis=0).sum())
                first = n72 == len(hits)
                lag_ok = ok[t] & (np.nan_to_num(side * r_now, nan=9) < LAG)
                fol = np.setdiff1d(mi[lag_ok[mi]], lead)
                ctl = notm[lag_ok[notm]]
                if len(fol) < 2 or len(ctl) < 10:
                    continue
                others = np.setdiff1d(mi, lead)
                sect_med = np.nanmedian(side * r_now[others]) if len(others) else np.nan
                pumped = (up72[t] >= PUMP) if side > 0 else (dn72[t] <= -PUMP)
                e = {"сторона": name, "сектор": label, "t": t_ms[t], "время (Киев)": kyiv(t_ms[t]).strftime("%Y-%m-%d %H:%M"),
                     "год": kyiv(t_ms[t]).year, "лидеры": " ".join(syms[j] for j in lead),
                     "лидер 24ч": float(np.nanmax(side * r_now[lead])), "сектор 24ч (медиана)": sect_med,
                     "выстрелили за 72ч": n72, "первый": first, "догоняющих": len(fol),
                     "пампнулись 72ч: догон.": np.nanmean(pumped[fol]), "пампнулись 72ч: контроль": np.nanmean(pumped[ctl]),
                     "цена 72ч: догон.": np.nanmean(side * ret72[t, fol]), "цена 72ч: контроль": np.nanmean(side * ret72[t, ctl])}
                for tn, m in TR.items():
                    if tn.startswith(name):
                        e[f"{tn}: догон."] = np.nanmean(m[t, fol])
                        e[f"{tn}: контроль"] = np.nanmean(m[t, ctl])
                        e[f"{tn}: винрейт догон."] = np.nanmean(m[t, fol] > 0)
                        e[f"{tn}: винрейт контроль"] = np.nanmean(m[t, ctl] > 0)
                events.append(e)
                n = len(fol)
                fr = {"сторона": [name] * n, "сектор": [label] * n, "t": np.full(n, t_ms[t]), "год": np.full(n, e["год"]),
                      "монета": fol, "первый": np.full(n, first), "выстрелили за 72ч": np.full(n, n72),
                      "сектор 24ч (медиана)": np.full(n, sect_med), "своя 24ч": side * r_now[fol],
                      "пампнулась 72ч": pumped[fol].astype(float), "цена 72ч": side * ret72[t, fol]}
                for tn, m in TR.items():
                    if tn.startswith(name):
                        fr[tn] = m[t, fol]
                frows.append(fr)
            print(name, label, "событий:", sum(1 for e in events if e["сектор"] == label and e["сторона"] == name), flush=True)

    ev = pd.DataFrame(events)
    fo = pd.concat([pd.DataFrame({k: np.asarray(v) for k, v in fr.items()}) for fr in frows], ignore_index=True)
    fo["монета"] = np.array(syms)[fo["монета"].to_numpy(int)]
    for tn in TR:
        if tn not in fo:
            fo[tn] = np.nan
    ev.to_csv(f"{OUT}/events.csv.gz", index=False)
    weeks = (t_ms[-1] - t_ms[t0]) / (7 * 24 * 3600e3)
    md = ["# Нарративы: идут ли монеты сектора одна за другой\n",
          f"Период: с {FROM.date()} по {kyiv(t_ms[-1]).date()}, 4h-свечи, USDT-перпетуалы Binance (оборот ≥ $1M за 24ч).",
          "Лидер — монета сектора +20% за 24ч (шорт: −20%), впервые за 72ч. Догоняющие — остальные монеты сектора, ещё не "
          "выросшие (< +10% за 24ч; шорт: не упавшие > −10%). Контроль — монеты НЕ из сектора в тот же момент.",
          "«Пампнулась 72ч» — за следующие 72ч хай ≥ +20% от входа (шорт: лой ≤ −20%). Сделки: вход по закрытию свечи, "
          "комиссия 0.1%, тейк/стоп в одной свече = стоп. «Перевес» = догоняющие минус контроль.\n"]

    def block(e, f, title):
        out = []
        for side in ("лонг", "шорт"):
            es = e[e["сторона"] == side]
            fs = f[f["сторона"] == side]
            if not len(es):
                continue
            r = {"срез": title, "сторона": side, "событий": len(es), "в неделю": f"{len(es) / weeks:.1f}",
                 "догон. пампнулись 72ч": pct(fs["пампнулась 72ч"].mean()),
                 "контроль пампнулись": pct(es["пампнулись 72ч: контроль"].mean()),
                 "перевес цены 72ч": pct((es["цена 72ч: догон."] - es["цена 72ч: контроль"]).mean())}
            for tn in ("10/10", "20/10"):
                k = f"{side} {tn}"
                r[f"{tn}: догон. ср."] = pct(fs[k].mean())
                r[f"{tn}: винрейт"] = f"{(fs[k] > 0).mean() * 100:.0f}%"
                r[f"{tn}: перевес"] = pct((es[f"{k}: догон."] - es[f"{k}: контроль"]).mean())
            out.append(r)
        return out

    T = []
    for lab in ev["сектор"].unique():
        T += block(ev[ev["сектор"] == lab], fo[fo["сектор"] == lab], lab)
    secs = pd.DataFrame(T)
    for side in ("лонг", "шорт"):
        s = secs[secs["сторона"] == side].copy()
        s["_k"] = s["перевес цены 72ч"].str.rstrip("%").replace("—", "nan").astype(float)
        md += [f"\n## По секторам — {side}\n", s.sort_values("_k", ascending=False).drop(columns="_k").to_markdown(index=False)]

    V = []
    for nm, msk_e, msk_f in [
            ("все", ev.index == ev.index, fo.index == fo.index),
            ("первый в секторе за 72ч", ev["первый"], fo["первый"]),
            ("уже 2-й выстрелил", ev["выстрелили за 72ч"] == 2, fo["выстрелили за 72ч"] == 2),
            ("уже 3+ выстрелили", ev["выстрелили за 72ч"] >= 3, fo["выстрелили за 72ч"] >= 3),
            ("сектор ещё стоит (медиана < 3%)", ev["сектор 24ч (медиана)"] < .03, fo["сектор 24ч (медиана)"] < .03),
            ("сектор уже пошёл (медиана ≥ 3%)", ev["сектор 24ч (медиана)"] >= .03, fo["сектор 24ч (медиана)"] >= .03)]:
        V += block(ev[msk_e], fo[msk_f], nm)
    for y in sorted(ev["год"].unique()):
        V += block(ev[ev["год"] == y], fo[fo["год"] == y], f"год {y}")
    md += ["\n## Все сектора вместе: варианты и годы\n",
           "(одна и та же монета может быть в нескольких секторах — тут посчитана по каждому)\n",
           pd.DataFrame(V).to_markdown(index=False)]

    B = []
    for side in ("лонг", "шорт"):
        fs = fo[fo["сторона"] == side]
        for lo, hi in ((-9, -.05), (-.05, 0), (0, .03), (.03, .06), (.06, .10)):
            g = fs[(fs["своя 24ч"] >= lo) & (fs["своя 24ч"] < hi)]
            if len(g):
                B.append({"сторона": side, "своя 24ч (в сторону сделки)": f"{max(lo, -.99) * 100:.0f}…{hi * 100:.0f}%",
                          "сделок": len(g), "пампнулась 72ч": pct(g["пампнулась 72ч"].mean()),
                          "цена 72ч": pct(g["цена 72ч"].mean()),
                          "10/10 ср.": pct(g[f"{side} 10/10"].mean()), "10/10 винрейт": f"{(g[f'{side} 10/10'] > 0).mean() * 100:.0f}%",
                          "20/10 ср.": pct(g[f"{side} 20/10"].mean())})
    md += ["\n## Догоняющие: зависит ли от того, начала ли монета уже двигаться\n", pd.DataFrame(B).to_markdown(index=False)]

    # ---- месяц под лупой ----
    m0 = pd.Timestamp(MONTH + "-01", tz="Europe/Kyiv")
    m1 = m0 + pd.offsets.MonthBegin(1)
    md += [f"\n# Месяц под лупой: {MONTH}\n"]
    ins = {s: [l for l, mem in uni["categories"].items() if s in mem] for s in syms}
    md.append("Монеты из неудачных сделок L3 и их сектора:\n")
    for b in LOSERS:
        s = next((x for x in syms if Hst.base_asset(x) == b), b + "USDT")
        md.append(f"- **{b}**: {', '.join(ins.get(s, [])) or 'нет ни в одной категории'}")
    em = ev[(ev["t"] >= m0.timestamp() * 1000) & (ev["t"] < m1.timestamp() * 1000)]
    for side in ("лонг", "шорт"):
        es = em[em["сторона"] == side].sort_values("t")
        md.append(f"\n## {side}: кто выстрелил и что было с сектором (по времени)\n")
        if len(es):
            md.append(es[["время (Киев)", "сектор", "лидеры", "лидер 24ч", "выстрелили за 72ч", "догоняющих",
                          "пампнулись 72ч: догон.", "пампнулись 72ч: контроль", "цена 72ч: догон.", "цена 72ч: контроль"]]
                      .assign(**{c: lambda d, c=c: d[c].map(pct) for c in ["лидер 24ч", "пампнулись 72ч: догон.",
                                 "пампнулись 72ч: контроль", "цена 72ч: догон.", "цена 72ч: контроль"]})
                      .to_markdown(index=False))
    # дневная медианная доходность секторов в этом месяце
    day = pd.to_datetime(t_ms + H4, unit="ms", utc=True).tz_convert("Europe/Kyiv").normalize()
    mm = (day >= m0 - pd.Timedelta(days=1)) & (day < m1)
    if mm.any():
        dc = pd.DataFrame(Cf[mm], index=day[mm]).groupby(level=0).last()
        dr = dc.pct_change()
        tab = {}
        for label, mem in uni["categories"].items():
            mi = [col[s] for s in mem if s in col]
            if len(mi) >= 5:
                tab[label] = dr[mi].median(axis=1)
        tab["весь рынок"] = dr.median(axis=1)
        dt_ = pd.DataFrame(tab).iloc[1:]
        dt_.index = dt_.index.strftime("%d.%m")
        md += ["\n## Медианное изменение цены за день по секторам, %\n",
               (dt_ * 100).round(1).T.to_markdown()]
    open(f"{OUT}/narrative_summary.md", "w").write("\n".join(md) + "\n")
    print("готово")


if __name__ == "__main__":
    m = os.environ.get("MODE", "")
    {"prep": prep, "combine": combine}.get(m, run_shard)()
