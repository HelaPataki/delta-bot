"""
Две «структурные» идеи для ШОРТА. Только история. Живых ботов не трогает.
Правила выбираем на 2021–2024, проверяем на 2025–2026 (не наоборот).

Данные: архив Binance — 1ч-свечи фьючерсов (все USDT-перпетуалы, включая делистнутые) и фандинг.
Результат шорта — % цены без плеча, с комиссией 0,1% на круг И С ФАНДИНГОМ (шорт платит при отрицательном фандинге).

А. ШОРТ НОВЫХ ЛИСТИНГОВ
  Листинг = первая часовая свеча фьючерса в архиве (берём листинги с 2021-02-01, чтобы это был именно запуск).
  Вход: через 2 часа, 1, 3, 7, 14, 30 дней после листинга (по открытию часа).
  Выходы: стоп +10/+15/+20/+30% × тейк −10/−20/−30% или «без тейка» (держим до срока); срок 7/14/30 дней.
  Стоп и тейк в одной свече — стоп.

Б. ШОРТ САМЫХ СЛАБЫХ АЛЬТОВ
  Каждый день в 00:00 UTC (02:00 Киев зимой) берём монеты с историей ≥ 30 дней и оборотом за сутки ≥ $5M,
  считаем «силу» = изменение монеты за N дней минус изменение BTC за N дней (N = 3 / 7 / 30),
  шортим 5 самых слабых. Держим 1 / 3 / 7 дней; без стопа или со стопом +10% / +15%.
  Вариант «рынок рос»: только дни, когда BTC за N дней вырос (монета не отскакивает вместе с рынком).
  Сравнение: шорт ВСЕХ подходящих монет в тот же день (средний альт) — чтобы отделить эффект «слабости»
  от того, что альты в целом падали.
"""

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
import pump_life as PL

OUT = os.environ.get("OUT_DIR", "results/short_struct")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
H, DAYMS = 3600 * 1000, 24 * 3600 * 1000
D0 = dt.date(2020, 11, 1)
FEE = 0.1
LIST_FROM = pd.Timestamp("2021-02-01", tz="UTC").timestamp() * 1000
DELAYS = {"2ч": 2, "1д": 24, "3д": 72, "7д": 168, "14д": 336, "30д": 720}
A_STOPS, A_TAKES, A_HOLDS = [10, 15, 20, 30], [10, 20, 30, None], [7, 14, 30]
B_LOOK, B_HOLD, B_STOPS = [3, 7, 30], [1, 3, 7], [None, 10, 15]
MIN_TURN = 5e6


def funding_raw(sym):
    frames = []
    for m in Hst.months(D0, dt.datetime.now(dt.timezone.utc).date()):
        f = PL.zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if f is not None:
            f = f[pd.to_numeric(f.iloc[:, 0], errors="coerce").notna()].iloc[:, [0, 2]].astype(float)
            f.columns = ["t", "rate"]
            frames.append(f)
    if not frames:
        return np.array([]), np.array([])
    f = pd.concat(frames).drop_duplicates("t").sort_values("t")
    return f.t.to_numpy(), f.rate.to_numpy() * 100            # % за одно начисление


class Fund:
    def __init__(self, ft, fr):
        self.t, self.c = ft, np.r_[0, np.cumsum(fr)]

    def short_gain(self, t0, t1):
        """Сколько % получает шорт за (t0, t1]: при положительном фандинге шорт получает, при отрицательном — платит."""
        if not len(self.t):
            return 0.0
        a, b = np.searchsorted(self.t, t0, side="right"), np.searchsorted(self.t, t1, side="right")
        return float(self.c[b] - self.c[a])


def short_trade(o, h, l, c, t, j, hours, stop, take, fund):
    """Шорт по открытию свечи j, держим до hours часов. Возвращает % (с комиссией и фандингом) и часы в сделке."""
    n = min(hours, len(c) - j)
    if n < hours * 0.9 or j >= len(c):
        return np.nan, np.nan
    e = o[j]
    hh, ll = h[j:j + n], l[j:j + n]
    js = int(np.argmax(hh >= e * (1 + stop / 100))) if stop and (hh >= e * (1 + stop / 100)).any() else n
    jt = int(np.argmax(ll <= e * (1 - take / 100))) if take and (ll <= e * (1 - take / 100)).any() else n
    if js == n and jt == n:
        r, k = (1 - c[j + n - 1] / e) * 100, n
    elif js <= jt:
        r, k = -stop, js + 1
    else:
        r, k = take, jt + 1
    r += fund.short_gain(t[j], t[j] + k * H) - FEE
    return r, k


def coin(sym, btc):
    k = PL.klines_any("um", sym, D0, dt.datetime.now(dt.timezone.utc).date())
    if k is None or len(k) < 24 * 10:
        return [], []
    t = k.open_time.to_numpy().astype("int64")
    o, h, l, c, qv = (k[x].to_numpy() for x in ("open", "high", "low", "close", "quote_volume"))
    fund = Fund(*funding_raw(sym))
    A, B = [], []

    # ---- А: листинги ----
    if t[0] >= LIST_FROM and sym != "BTCUSDT":
        for dn, dh in DELAYS.items():
            j = int(np.searchsorted(t, t[0] + dh * H))
            if j >= len(t):
                continue
            row = {"symbol": sym, "листинг": int(t[0]), "вход": dn, "t": int(t[j]),
                   "оборот 1-х суток $": qv[:24].sum(), "рост от старта, %": (o[j] / o[0] - 1) * 100}
            for hold in A_HOLDS:
                for st in A_STOPS:
                    for tk in A_TAKES:
                        r, _ = short_trade(o, h, l, c, t, j, hold * 24, st, tk, fund)
                        row[f"{hold}д стоп{st} тейк{tk or '—'}"] = r
                row[f"{hold}д без стопа"] = short_trade(o, h, l, c, t, j, hold * 24, None, None, fund)[0]
            A.append(row)

    # ---- Б: ежедневные строки ----
    day0 = np.where(t % DAYMS == 0)[0]
    for j in day0:
        if j < 24 * 30 or t[j] < pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000:
            continue
        turn = qv[j - 24:j].sum()
        if turn < MIN_TURN:
            continue
        row = {"symbol": sym, "t": int(t[j]), "оборот $": turn}
        for L in B_LOOK:
            jl = j - L * 24
            if jl < 0:
                continue
            bi = np.searchsorted(btc[0], t[j], side="right") - 1
            bl = np.searchsorted(btc[0], t[jl], side="right") - 1
            btc_r = (btc[1][bi] / btc[1][bl] - 1) * 100 if bi >= 0 and bl >= 0 else np.nan
            row[f"сила {L}д"] = (o[j] / o[jl] - 1) * 100 - btc_r
            row[f"BTC {L}д"] = btc_r
        for hd in B_HOLD:
            for st in B_STOPS:
                row[f"{hd}д стоп{st or '—'}"] = short_trade(o, h, l, c, t, j, hd * 24, st, None, fund)[0]
        B.append(row)
    return A, B


def prep():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(s for s in Hst.list_symbols() if s.endswith("USDT"))
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    syms = json.load(open(f"{OUT}/universe.json"))
    mine = [s for i, s in enumerate(syms) if i % NSHARDS == SHARD]
    b = PL.klines_any("um", "BTCUSDT", D0, dt.datetime.now(dt.timezone.utc).date())
    btc = (b.open_time.to_numpy(), b.open.to_numpy())

    def work(s):
        try:
            return coin(s, btc)
        except Exception as ex:
            print(f"{s}: {ex!r}", file=sys.stderr)
            return [], []

    A, B = [], []
    with ThreadPoolExecutor(3) as ex:
        for a, b_ in ex.map(work, mine):
            A += a
            B += b_
    pd.DataFrame(A).to_csv(f"{OUT}/parts/a{SHARD}.csv.gz", index=False)
    pd.DataFrame(B).to_csv(f"{OUT}/parts/b{SHARD}.csv.gz", index=False)
    print(f"часть {SHARD}: листингов {len(A)}, дней {len(B)}, монет {len(mine)}")


def read(prefix):
    parts = []
    for fn in sorted(os.listdir(f"{OUT}/parts")):
        if fn.startswith(prefix):
            try:
                parts.append(pd.read_csv(f"{OUT}/parts/{fn}"))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def st(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"сделок": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {"сделок": len(r), "винрейт, %": (r > 0).mean() * 100, "средняя, %": r.mean(),
            "медиана, %": float(np.median(r)), "PF": pos / neg if neg > 0 else np.inf, "серия минусов": best}


def by_year(g, col):
    return {str(y): round(x[col].mean(), 2) for y, x in g.groupby("год")}


def combine():
    w = open(f"{OUT}/short_struct_summary.md", "w").write
    w("# Структурные шорты: новые листинги и самые слабые альты\n\n" + __doc__.split("\n", 2)[2] + "\n")

    # ---------- А ----------
    a = read("a")
    a["год"] = pd.to_datetime(a.листинг, unit="ms").dt.year
    a.to_csv(f"{OUT}/listings.csv.gz", index=False)
    cols = [x for x in a.columns if x.endswith("без стопа") or "стоп" in x and "тейк" in x]
    w(f"## А. Шорт новых листингов — {a.symbol.nunique()} листингов\n\n")
    w("Сколько листингов по годам: " + ", ".join(f"{y}: {n}" for y, n in
                                                a[a.вход == '2ч'].groupby('год').size().items()) + "\n\n")
    rows = []
    for dn in DELAYS:
        g = a[a.вход == dn].sort_values("t")
        tr, te = g[g.год <= 2024], g[g.год >= 2025]
        for col in cols:
            s1 = st(tr[col])
            if s1.get("сделок", 0) < int(os.environ.get("MIN_A", "30")):
                continue
            rows.append({"вход": dn, "выход": col, **{f"21–24 {k}": v for k, v in s1.items()},
                         "25–26 средняя, %": te[col].mean(), "25–26 винрейт, %": (te[col].dropna() > 0).mean() * 100,
                         "25–26 сделок": te[col].notna().sum()})
    rd = pd.DataFrame(rows, columns=["вход", "выход", "21–24 средняя, %"]) if not rows else pd.DataFrame(rows)
    w("### Лучшие 20 вариантов по 2021–2024 (по средней сделке) и как они прошли 2025–2026\n\n")
    top = rd.sort_values("21–24 средняя, %", ascending=False).head(20)
    w(top.round(2).to_markdown(index=False) + "\n\n")
    w("### Держать без стопа — для понимания, как ведут себя листинги (средняя / медиана, %)\n\n")
    rows = []
    for dn in DELAYS:
        g = a[a.вход == dn]
        for hd in A_HOLDS:
            col = f"{hd}д без стопа"
            rows.append({"вход": dn, "держать": f"{hd}д", **by_year(g, col),
                         "медиана 21–24": g[g.год <= 2024][col].median(), "медиана 25–26": g[g.год >= 2025][col].median(),
                         "худшая сделка": g[col].min()})
    w(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")
    if len(top):
        b0 = top.iloc[0]
        g = a[a.вход == b0.вход]
        w(f"Лучший по 2021–2024 ({b0.вход}, {b0.выход}) по годам: " +
          ", ".join(f"{k}: {v:+.2f}%" for k, v in by_year(g, b0.выход).items()) + "\n\n")

    # ---------- Б ----------
    b = read("b")
    b["год"] = pd.to_datetime(b.t, unit="ms").dt.year
    w(f"## Б. Шорт самых слабых альтов — {b.symbol.nunique()} монет, {b.t.nunique()} дней\n\n")
    w("«слабые» — средняя сделка по 5 самым слабым монетам дня; «все» — средняя по всем подходящим монетам того же дня; "
      "«разница» — насколько слабые лучше среднего альта для шорта.\n\n")
    b = b[b.symbol != "BTCUSDT"]
    rows, keep = [], {}
    for L in B_LOOK:
        for only_up in (False, True):
            g = b.dropna(subset=[f"сила {L}д"])
            if only_up:
                g = g[g[f"BTC {L}д"] > 0]
            weak = g.sort_values(f"сила {L}д").groupby("t").head(5)
            for hd in B_HOLD:
                for stp in B_STOPS:
                    col = f"{hd}д стоп{stp or '—'}"
                    for per, sel in (("21–24", lambda x: x[x.год <= 2024]), ("25–26", lambda x: x[x.год >= 2025])):
                        wk, al = sel(weak), sel(g)
                        s = st(wk.sort_values("t")[col])
                        rows.append({"сила за": f"{L}д", "только рынок рос": only_up, "держать": f"{hd}д",
                                     "стоп": stp or "—", "период": per, **s, "все: средняя, %": al[col].mean(),
                                     "разница, %": s.get("средняя, %", np.nan) - al[col].mean()})
                    keep[(L, only_up, col)] = weak
    rb = pd.DataFrame(rows)
    tr = rb[rb.период == "21–24"].sort_values("средняя, %", ascending=False)
    te = rb[rb.период == "25–26"].set_index(["сила за", "только рынок рос", "держать", "стоп"])
    tr = tr.join(te[["средняя, %", "винрейт, %", "разница, %"]].rename(columns=lambda x: "25–26 " + x),
                 on=["сила за", "только рынок рос", "держать", "стоп"])
    w("### Все варианты, отсортированы по 2021–2024, рядом — 2025–2026\n\n")
    w(tr.drop(columns=["период"]).round(2).to_markdown(index=False) + "\n\n")
    if len(tr):
        x = tr.iloc[0]
        L = int(x["сила за"][:-1])
        col = f"{x['держать'][:-1]}д стоп{x['стоп']}"
        g = keep[(L, bool(x["только рынок рос"]), col)]
        w(f"Лучший по 2021–2024 (сила за {x['сила за']}, рынок рос: {x['только рынок рос']}, {col}) по годам: " +
          ", ".join(f"{k}: {v:+.2f}%" for k, v in by_year(g, col).items()) + "\n\n")
        g.assign(**{"время (Киев)": g.t.map(PL.kyiv)}).to_csv(f"{OUT}/weak_best_trades.csv.gz", index=False)
    print(open(f"{OUT}/short_struct_summary.md").read()[:9000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        run_shard()
