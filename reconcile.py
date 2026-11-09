"""
Сверка живого бота с архивом Binance (то, на чём считался тест). Живой бот не меняется.

Каждый день (MODE=daily) — за вчерашние сутки и все пропущенные дни, для которых архив уже выложен:
  🟢 лонг — полностью: свечи 4ч и минутки из архива → те же правила → сравнение с журналом бота;
  🔴 шорт — для каждого шорта бота сверяются OI за 24 ч и цена за 24 ч с архивом.
     Фандинг за текущий месяц Binance выкладывает только после конца месяца, поэтому полная проверка шортов — раз в месяц.
Раз в месяц (там же, когда появился фандинг за прошлый месяц) — шорты полностью: лишние и пропущенные.
Итог — короткое сообщение в Telegram и строки в reconcile_log.csv.
"""

import csv
import datetime as dt
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import history as Hst
from tradfi import TRADFI

H, H4, MIN = 3600 * 1000, 4 * 3600 * 1000, 60_000
BOT_START = pd.Timestamp("2026-11-07 12:00", tz="UTC")       # первая свеча, которую бот вёл с журналом (14:00 Киев)
STATE, LOG = "reconcile_state.json", "reconcile_log.csv"
SIGNALS = "live_signals.csv"
KYIV = "Europe/Kyiv"
TOKEN, CHAT = os.environ.get("TG_TOKEN", ""), os.environ.get("TG_CHAT", "")
Hst.MCAP_MIN, Hst.MCAP_MAX, Hst.MIN_TURNOVER = 10e6, 400e6, 50_000


def tg(text):
    print("---- TELEGRAM ----\n" + text)
    if TOKEN:
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT, "text": text[:4000]}, timeout=30)
        except requests.RequestException as e:
            print("telegram:", e)


def kyiv(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(KYIV).strftime("%Y-%m-%d %H:%M")


def zip_df(url, ncols, names):
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    if isinstance(df.iloc[0, 0], str):
        df = df.iloc[1:]
    df = df.iloc[:, :ncols]
    df.columns = names
    return df


def k4(sym, day):
    d = zip_df(f"{Hst.ARCHIVE}/data/futures/um/daily/klines/{sym}/4h/{sym}-4h-{day:%Y-%m-%d}.zip", 12, Hst.COLS)
    return None if d is None else d.astype(float)


def m1(sym, day):
    d = zip_df(f"{Hst.ARCHIVE}/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day:%Y-%m-%d}.zip", 6,
               ["open_time", "open", "high", "low", "close", "volume"])
    return None if d is None else d.astype(float)


def metrics(sym, day):
    url = f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip"
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
    t = pd.to_datetime(df.create_time, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ms]").astype("int64")
    return pd.DataFrame({"t": t.astype(float), "oi": df.sum_open_interest.astype(float)}).sort_values("t")


def archive_ready(day):
    return k4("BTCUSDT", day) is not None


def universe():
    syms = Hst.list_symbols()
    caps = Hst.coingecko_mcaps()
    return [s for s in syms if Hst.MCAP_MIN <= caps.get(Hst.base_asset(s), 0) <= Hst.MCAP_MAX and s not in TRADFI]


def bot_rows():
    if not os.path.exists(SIGNALS):
        b = pd.DataFrame(columns=["свеча закрылась (Киев)", "сторона", "монета"])
    else:
        b = pd.read_csv(SIGNALS)
    b["T"] = [pd.Timestamp(x, tz=KYIV).tz_convert("UTC").timestamp() * 1000 for x in b["свеча закрылась (Киев)"]]
    return b


def in_day(b, day):
    """Свечи, открытые в UTC-сутки day: закрытия в (00:00, 24:00] UTC."""
    d0 = pd.Timestamp(day, tz="UTC").timestamp() * 1000
    return b[(b["T"] > d0) & (b["T"] <= d0 + 24 * H)]


# ---------------- лонг за день ----------------
def long_day(sym, day):
    frames = [k4(sym, day - dt.timedelta(days=k)) for k in (2, 1, 0)]
    if frames[2] is None:
        return None                      # архив за этот день по монете ещё не выложен
    k = pd.concat([f for f in frames if f is not None]).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    qv, v, tbv = k.quote_volume.to_numpy(), k.volume.to_numpy(), k.taker_buy_volume.to_numpy()
    out = []
    start = pd.Timestamp(day, tz="UTC").timestamp() * 1000
    for i in range(12, len(k)):
        t = k.open_time.iloc[i]
        if t < start or t + H4 <= BOT_START.timestamp() * 1000:
            continue
        ratio = qv[i] / np.median(qv[i - 12:i])
        delta = (2 * tbv[i] - v[i]) / v[i] * 100 if v[i] else np.nan
        if not (ratio >= 5 and delta <= -17 and qv[i] >= Hst.MIN_TURNOVER):
            continue
        mm = m1(sym, day)
        if mm is None:
            out.append({"монета": sym, "T": t + H4, "примечание": "нет минуток в архиве"})
            continue
        w = mm[(mm.open_time >= t) & (mm.open_time < t + H4)]
        if len(w) < 60:
            continue
        top = w.loc[w.volume.idxmax()]
        poc = (top.high + top.low) / 2
        hi, lo = k.high.iloc[i], k.low.iloc[i]
        pos = (poc - lo) / (hi - lo) * 100 if hi > lo else 50
        if pos <= 30 and k.close.iloc[i] >= poc:
            out.append({"монета": sym, "T": t + H4, "объём x": round(ratio, 1), "дельта %": round(delta, 1), "POC позиция %": round(pos, 1)})
    return out


# ---------------- шорт: проверка OI и цены у сигналов бота ----------------
def short_values(sym, T):
    day = pd.Timestamp(T, unit="ms", tz="UTC").date()
    oi = [metrics(sym, day - dt.timedelta(days=k)) for k in (2, 1, 0)]
    oi = [o for o in oi if o is not None]
    ks = [k4(sym, day - dt.timedelta(days=k)) for k in (2, 1, 0)]
    ks = [x for x in ks if x is not None]
    res = {}
    if oi:
        o = pd.concat(oi).sort_values("t")

        def at(x):
            s = o[o.t <= x]
            return s.oi.iloc[-1] if len(s) else np.nan
        res["OI 24ч архив %"] = round((at(T) / at(T - 24 * H) - 1) * 100, 1)
    if ks:
        k = pd.concat(ks).drop_duplicates("open_time").set_index("open_time")
        c_now, c_24 = k.close.get(T - H4), k.close.get(T - H4 - 24 * H)
        if c_now and c_24:
            res["цена 24ч архив %"] = round((c_now / c_24 - 1) * 100, 2)
    return res


def daily(day, uni, bot):
    """Сверка за день. None — архив Binance за этот день ещё выложен не по всем нужным монетам (проверим завтра)."""
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda s: long_day(s, day), uni))
    missing = {s for s, r in zip(uni, res) if r is None}
    arch = [x for r in res if r for x in r]
    bl = in_day(bot[bot["сторона"] == "лонг"], day)
    bot_missing = (set(bl["монета"]) | set(in_day(bot[bot["сторона"] == "шорт"], day)["монета"])) & missing
    if bot_missing or len(missing) > 0.05 * len(uni):
        print("архив за", day, "неполный: нет", len(missing), "монет, из них у бота", sorted(bot_missing), flush=True)
        return None, len(missing), sorted(bot_missing)
    a_keys = {(a["монета"], kyiv(a["T"])) for a in arch if "примечание" not in a}
    b_keys = {(r["монета"], r["свеча закрылась (Киев)"]) for _, r in bl.iterrows()}
    lines = [f"🔎 Сверка за {day:%d.%m} (архив Binance)"]
    ok_l = a_keys == b_keys
    lines.append(f"🟢 Лонг: бот {len(b_keys)} · архив {len(a_keys)} {'✅' if ok_l else '⚠️'}")
    for m, t in sorted(a_keys - b_keys):
        lines.append(f"   пропустил: {m} {t[11:]}")
    for m, t in sorted(b_keys - a_keys):
        lines.append(f"   лишний: {m} {t[11:]}")
    rows = [{"дата": str(day), "сторона": "лонг", "монета": m, "свеча": t, "статус": "совпал" if (m, t) in a_keys & b_keys else
             ("пропущен ботом" if (m, t) in a_keys else "лишний у бота")} for m, t in sorted(a_keys | b_keys)]

    bs = in_day(bot[bot["сторона"] == "шорт"], day)
    lines.append(f"🔴 Шорт: бот {len(bs)}" + (" (сверка OI и цены)" if len(bs) else ""))
    for _, r in bs.iterrows():
        v = short_values(r["монета"], int(r["T"]))
        oi_b, oi_a = r.get("OI 24ч %"), v.get("OI 24ч архив %")
        px_b, px_a = r.get("цена 24ч %"), v.get("цена 24ч архив %")
        good = (oi_a is not None and oi_a >= 60 and (px_a is None or px_a >= -2))
        lines.append(f"   {r['монета']}: OI {oi_b}% / архив {oi_a}% · цена {px_b}% / архив {px_a}% {'✅' if good else '⚠️'}")
        rows.append({"дата": str(day), "сторона": "шорт", "монета": r["монета"], "свеча": r["свеча закрылась (Киев)"],
                     "статус": "OI и цена совпали" if good else "расхождение OI/цены", **v})
    lines.append("Полная проверка шортов (с фандингом) — в начале следующего месяца")
    return "\n".join(lines), rows


# ---------------- шорт за месяц, полностью ----------------
def monthly(month, uni, bot):
    os.environ["SIG_FROM"] = f"{month:%Y-%m-01}"
    import importlib
    import oi_funding as OF
    OF = importlib.reload(OF)
    Hst.MCAP_MIN, Hst.MCAP_MAX, Hst.MIN_TURNOVER = 10e6, 400e6, 50_000
    caps = Hst.coingecko_mcaps()
    end = (pd.Timestamp(month) + pd.offsets.MonthBegin(1)).tz_localize("UTC").timestamp() * 1000
    m0 = pd.Timestamp(month).tz_localize("UTC").timestamp() * 1000

    def work(sym):
        try:
            return OF.coin(sym, caps.get(Hst.base_asset(sym), 0))
        except Exception as e:  # noqa: BLE001
            print(sym, e)
            return None
    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(work, uni) if p is not None and len(p)]
    a = pd.concat(parts).sort_values(["symbol", "open_time"]) if parts else pd.DataFrame()
    arch = set()
    if len(a):
        g1 = (a.fund24 <= -0.03) & (a.oi_chg24 >= 20) & (a.px24 >= -2)
        sig = OF.first_only(a, g1) & (a.fund24 <= -0.2) & (a.oi_chg24 >= 60)
        s = a[sig & (a.open_time + H4 <= end) & (a.open_time + H4 > m0) & (a.open_time + H4 >= BOT_START.timestamp() * 1000)]
        arch = {(r.symbol, kyiv(r.open_time + H4)) for r in s.itertuples()}
    mstr = f"{month:%Y-%m}"
    bs = bot[(bot["сторона"] == "шорт") & (bot["T"] > m0) & (bot["T"] <= end)]
    b_keys = {(r["монета"], r["свеча закрылась (Киев)"]) for _, r in bs.iterrows()}
    both = arch & b_keys
    lines = [f"🔎 Сверка шортов за {month:%m.%Y} (архив Binance, с фандингом)",
             f"🔴 бот {len(b_keys)} · архив {len(arch)} · совпали {len(both)} {'✅' if arch == b_keys else '⚠️'}"]
    miss, extra = sorted(arch - b_keys), sorted(b_keys - arch)
    if miss:
        lines.append("   пропустил: " + ", ".join(f"{m} {t[5:]}" for m, t in miss[:15]))
    if extra:
        lines.append("   лишние: " + ", ".join(f"{m} {t[5:]}" for m, t in extra[:15]))
    rows = [{"дата": mstr, "сторона": "шорт (месяц)", "монета": m, "свеча": t,
             "статус": "совпал" if (m, t) in both else ("пропущен ботом" if (m, t) in arch else "лишний у бота")}
            for m, t in sorted(arch | b_keys)]
    return "\n".join(lines), rows


def write_log(rows):
    if not rows:
        return
    cols = ["дата", "сторона", "монета", "свеча", "статус", "OI 24ч архив %", "цена 24ч архив %"]
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"last_day": "2026-11-06", "last_month": "2026-10"}
    bot = bot_rows()
    uni = universe()
    print("монет:", len(uni), flush=True)
    today = dt.datetime.now(dt.timezone.utc).date()
    day = dt.date.fromisoformat(st["last_day"]) + dt.timedelta(days=1)
    while day < today:
        if not archive_ready(day):
            print("архив за", day, "ещё не выложен")
            break
        res = daily(day, uni, bot)
        if res[0] is None:                       # архив неполный — этот день проверим при следующем запуске
            if (today - day).days >= 3:
                tg(f"⏳ Сверка за {day:%d.%m}: Binance до сих пор не выложил архив по {res[1]} монетам — жду дальше")
            break
        text, rows = res
        tg(text)
        write_log(rows)
        st["last_day"] = str(day)
        day += dt.timedelta(days=1)
    prev_month = (pd.Timestamp(today).replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
    if f"{prev_month:%Y-%m}" > st["last_month"]:
        test = Hst.get(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{prev_month:%Y-%m}.zip")
        if test is not None and test.status_code == 200:
            text, rows = monthly(prev_month.date(), uni, bot)
            tg(text)
            write_log(rows)
            st["last_month"] = f"{prev_month:%Y-%m}"
    json.dump(st, open(STATE, "w"), indent=1)


if __name__ == "__main__":
    main()
