"""
ВРЕМЕННАЯ проверка (удалить после). Бот не меняется — используются его функции и данные Coinalyze.
1) Лонги: все монеты бота, все 4ч-свечи 20.09–03.11.2026 — объём ≥ x5 и дельта ≤ −17% (этап 1), затем POC
   (минутки Coinalyze — где ещё хранятся). Сравнение с архивом Binance (results/poc_signals.csv).
2) Ранний отбор лонгов (за 42 мин): поймал бы архивные сигналы — в основной список (≥ x2), во второй проход (x1,2–2)
   или нет. Объём неполной свечи — по часовым барам за первые 3 ч (чуть меньше, чем видит бот в T−42 мин).
3) Ранний отбор шортов (за 65 мин) для архивных шортов: прогноз < 0 или фандинг 24ч ≤ −0,1% → OI ≥ +15%.
Coinalyze делится с живым ботом: в окно его работы проверка ждёт.
"""

import datetime as dt
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live as L

KYIV = ZoneInfo("Europe/Kyiv")
H, H4 = 3600, 4 * 3600
START = int(dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc).timestamp())
END = int(dt.datetime(2026, 11, 3, 12, tzinfo=dt.timezone.utc).timestamp())
_orig_cz = L.cz


def polite_cz(*a, **k):
    while True:
        ph = time.time() % H4
        if 15 * 60 <= ph < 100 * 60:
            break
        wait = (15 * 60 - ph) % H4 + 5
        L.log(f"окно живого бота — жду {wait / 60:.0f} мин")
        time.sleep(wait)
    return _orig_cz(*a, **k)


L.cz = polite_cz


def kv(T):
    return dt.datetime.fromtimestamp(T, KYIV).strftime("%d.%m %H:%M")


def send(lines):
    print("\n".join(lines), flush=True)
    chunk = ""
    for ln in lines:
        if len(chunk) + len(ln) > 3800:
            L.tg(chunk)
            chunk = ""
        chunk += ln + "\n"
    if chunk:
        L.tg(chunk)


def partial_ratio(sym_cz, T):
    """Объём неполной свечи (первые 3 ч) к медиане 12 прошлых 4ч-свечей — как ранний отбор бота."""
    b4 = L.history("ohlcv-history", [sym_cz], "4hour", T - 14 * H4, T - H4 + 60).get(sym_cz, [])
    b1 = L.history("ohlcv-history", [sym_cz], "1hour", T - H4, T - H + 60).get(sym_cz, [])
    prev = [(b["o"] + b["h"] + b["l"] + b["c"]) / 4 * (b.get("v") or 0) for b in b4 if T - 13 * H4 <= b["t"] < T - H4]
    cur = [b for b in b1 if T - H4 <= b["t"] < T - H]
    if len(prev) < 12 or not cur:
        return np.nan
    qv = sum((b["o"] + b["h"] + b["l"] + b["c"]) / 4 * (b.get("v") or 0) for b in cur)
    return qv / float(np.median(prev[-12:]))


def main():
    uni = L.universe()
    rev = {v: k for k, v in uni.items()}
    Ts = list(range(START + H4, END + 1, H4))
    lines = ["🧪 Лонги: все монеты, свечи 20.09–03.11 — бот на данных Coinalyze против архива Binance"]

    # ---------- архив ----------
    p = pd.read_csv("results/poc_signals.csv")
    tt = pd.to_datetime(p.candle_close_kyiv).dt.tz_localize(KYIV, ambiguous="NaT", nonexistent="NaT")
    p = p[tt.notna()].copy()
    p["T"] = (tt[tt.notna()] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    p = p[(p["T"] > START) & (p["T"] <= END) & p.mcap_est.between(L.MCAP_MIN, L.MCAP_MAX)]
    st1_a = p[(p.vol_ratio >= 5) & (p.delta_pct <= -17) & (p.turnover_usd >= L.MIN_TURNOVER)]
    sig_a = st1_a[(st1_a.poc_pos <= 30) & (st1_a.close_below_poc == "нет")]
    A1 = {(r.symbol, r.T) for r in st1_a.itertuples()}
    AS = {(r.symbol, r.T) for r in sig_a.itertuples()}

    # ---------- этап 1 по Coinalyze ----------
    ohlc = L.history("ohlcv-history", list(uni), "4hour", START - 14 * H4, END + 60)
    st1 = {}
    for s, bars in ohlc.items():
        for T in Ts:
            c = L.candle_feats(bars, T)
            if c and c["turnover"] >= L.MIN_TURNOVER and c["vol_ratio"] >= 5 and c["delta_pct"] <= -17:
                st1[(uni[s], T)] = (s, c)
    B1 = set(st1)
    in_uni = lambda k: k[0] in rev  # noqa: E731
    lines.append(f"Этап 1 (объём ≥ x5, дельта ≤ −17%): совпали {len(B1 & A1)}, только в боте {len(B1 - A1)}, "
                 f"только в архиве {len(A1 - B1)} (из них монет вне списка бота {sum(not in_uni(k) for k in A1 - B1)})")
    for b, T in sorted(A1 - B1, key=lambda x: x[1]):
        if not in_uni((b, T)):
            continue
        s = rev[b]
        c = L.candle_feats(ohlc.get(s, []), T)
        r = st1_a[(st1_a.symbol == b) & (st1_a["T"] == T)].iloc[0]
        got = "нет данных" if c is None else f"объём x{c['vol_ratio']:.1f}, дельта {c['delta_pct']:.0f}%"
        lines.append(f"  ➖ {b} {kv(T)}: бот {got} · архив x{r.vol_ratio:.1f}, {r.delta_pct:.0f}%")
    for b, T in sorted(B1 - A1, key=lambda x: x[1])[:15]:
        c = st1[(b, T)][1]
        lines.append(f"  ➕ {b} {kv(T)}: бот объём x{c['vol_ratio']:.1f}, дельта {c['delta_pct']:.0f}%")

    # ---------- POC по минуткам (где есть) ----------
    byT = {}
    for (b, T), (s, c) in st1.items():
        byT.setdefault(T, []).append((b, s, c))
    BS, nopoc = set(), 0
    for T, items in byT.items():
        mb = L.history("ohlcv-history", [s for _, s, _ in items], "1min", T - H4, T - 60)
        for b, s, c in items:
            pi = L.poc_info(mb.get(s, []), T)
            if pi is None:
                nopoc += 1
            elif L.is_long(c, pi):
                BS.add((b, T))
    checked = {k for k in B1 if True}
    lines.append(f"Полный сигнал (с POC): минуток нет у {nopoc} из {len(B1)} свечей этапа 1 — там POC не проверить")
    AS_check = {k for k in AS if k in B1}
    lines.append(f"  среди свечей с минутками: совпали {len(BS & AS)}, только в боте {len(BS - AS)}, "
                 f"только в архиве {len(AS_check - BS)} (часть — нет минуток)")
    for b, T in sorted(BS - AS, key=lambda x: x[1]):
        lines.append(f"  ➕ {b} {kv(T)}")

    # ---------- ранний отбор лонгов ----------
    lines.append(f"Ранний отбор лонгов для архивных сигналов ({len(AS)}):")
    for b, T in sorted(AS, key=lambda x: x[1]):
        if b not in rev:
            lines.append(f"  {b} {kv(T)}: вне списка монет бота")
            continue
        r = partial_ratio(rev[b], T)
        where = "нет данных" if np.isnan(r) else ("✅ основной список" if r >= 2 else "🟡 второй проход" if r >= 1.2 else "❌ выпал бы")
        lines.append(f"  {b} {kv(T)}: объём за 3 ч x{r:.1f} → {where}")

    # ---------- ранний отбор шортов ----------
    sh = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz").drop_duplicates(["symbol", "open_time"])
    sh["T"] = (sh.open_time // 1000 + H4).astype(int)
    sh = sh[(sh["T"] > START) & (sh["T"] <= END)]
    lines.append(f"Ранний отбор шортов для архивных сигналов ({len(sh)}):")
    for r in sh.sort_values("T").itertuples():
        b, T = r.symbol, int(r.T)
        if b not in rev:
            lines.append(f"  {b} {kv(T)}: вне списка монет бота")
            continue
        s, X = rev[b], T - 65 * 60
        pr = L.history("predicted-funding-rate-history", [s], "1hour", X - 30 * H, X).get(s, [])
        ob = L.history("open-interest-history", [s], "1hour", X - 26 * H, X, convert_to_usd="false").get(s, [])
        pr = [x for x in pr if x["t"] + H <= X + 300]
        hours = L.funding_interval(b)
        neg = any((x.get("c") or 0) < 0 for x in pr[-25:])
        pred = pr[-1]["c"] * 8 / hours if pr else np.nan
        f24 = L.fund24(pr, X, hours)
        o = {x["t"] + H: x["c"] for x in ob}
        oend = max((t for t in o if t <= X), default=None)
        o24 = o.get(oend - 24 * H) if oend else None
        oi = (o[oend] / o24 - 1) * 100 if oend and o24 else np.nan
        cand = neg and (pred < 0 or f24 <= -0.1)
        where = "❌ выпал бы" if not cand else ("✅ основной список" if (oi >= 15 or np.isnan(oi)) else "🟡 второй проход")
        lines.append(f"  {b} {kv(T)}: прогноз {pred:.2f}%, фандинг 24ч {f24:.2f}%, OI {oi:+.0f}% → {where}")
    send(lines)


if __name__ == "__main__":
    main()
