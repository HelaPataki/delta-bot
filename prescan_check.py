"""
Проверка живого бота на истории: сколько настоящих сигналов попало бы в предварительный отбор,
который бот делает ДО закрытия свечи. Старые файлы только читаются.

Шорт (379 сигналов финального фильтра): за 70 мин до закрытия бот берёт монеты с отрицательным текущим фандингом,
  затем оставляет те, где фандинг 24ч ≤ −0,1% или последняя ставка ≤ −0,1% (на 8 ч) и OI за 24ч ≥ +25%.
  Плюс проверка интервала фандинга: бот берёт интервал из архива за прошлый месяц — совпал ли он с настоящим
  и не потерялся ли из-за этого сигнал.
Лонг (87 сигналов): за 22 мин до закрытия бот берёт монеты, у которых объём незакрытой свечи уже ≥ 2,5x к обычному.
"""

import datetime as dt
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import history as Hst
from poc import minutes

OUT_DIR = "results/prescan_check"
H = 3600 * 1000
H4 = 4 * H
SHORT_BEFORE = 70 * 60 * 1000
LONG_BEFORE = 22 * 60 * 1000


def zip_csv(url):
    r = Hst.get(url)
    if r is None or r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, header=None)


def funding_file(sym, month):
    df = zip_csv(f"{Hst.ARCHIVE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{month:%Y-%m}.zip")
    if df is None:
        return None
    if isinstance(df.iloc[0, 0], str):
        df = df.iloc[1:]
    df = df.iloc[:, :3].astype(float)
    df.columns = ["t", "hours", "rate"]
    return df


def oi_day(sym, day):
    df = zip_csv(f"{Hst.ARCHIVE}/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip")
    if df is None:
        return None
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    ts = pd.to_datetime(df.create_time, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ms]")
    return pd.DataFrame({"t": ts.astype("int64").astype(float), "oi": df.sum_open_interest.astype(float).to_numpy()})


def month_of(ms):
    d = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()
    return dt.date(d.year, d.month, 1)


def prev_month(m):
    return (m - dt.timedelta(days=1)).replace(day=1)


def asof(df, col, x, max_age=None):
    s = df[df.t <= x]
    if len(s) == 0 or (max_age and x - s.t.iloc[-1] > max_age):
        return np.nan
    return s[col].iloc[-1]


def short_one(sym, T):
    X = T - SHORT_BEFORE
    m = month_of(T)
    parts = [funding_file(sym, mm) for mm in (prev_month(prev_month(m)), prev_month(m), m)]
    f = pd.concat([p for p in parts if p is not None]) if any(p is not None for p in parts) else None
    days = sorted({dt.datetime.fromtimestamp((T - k * H) / 1000, dt.timezone.utc).date() for k in (0, 12, 26, 30)})
    oi = [oi_day(sym, d) for d in days]
    oi = pd.concat([o for o in oi if o is not None]).sort_values("t") if any(o is not None for o in oi) else None
    r = {"symbol": sym, "T": T}
    if f is None or oi is None:
        r["нет данных"] = True
        return r
    f = f.sort_values("t")
    f["rate8"] = f.rate * 8 / f.hours * 100        # в % на 8 ч
    last = f[f.t <= X]
    r["ставка сейчас (сырая) %"] = last.rate.iloc[-1] * 100 if len(last) else np.nan
    win = f[(f.t > X - 24 * H) & (f.t <= X)]
    r["фандинг 24ч заранее"] = win.rate8.mean() if len(win) else np.nan
    r["последняя ставка заранее"] = last.rate8.iloc[-1] if len(last) else np.nan
    r["OI 24ч заранее"] = (asof(oi, "oi", X, 30 * 60 * 1000) / asof(oi, "oi", X - 24 * H) - 1) * 100
    pre = r["ставка сейчас (сырая) %"] <= -0.005
    oi_ok = r["OI 24ч заранее"] >= 25 or np.isnan(r["OI 24ч заранее"])
    r["попал в отбор"] = bool(pre and (r["фандинг 24ч заранее"] <= -0.1 or r["последняя ставка заранее"] <= -0.1) and oi_ok)
    # интервал: настоящий на момент сигнала против последнего в прошлом месяце (так считает бот)
    win_T = f[(f.t > T - 24 * H) & (f.t <= T)]
    pm = parts[1]
    bot_h = pm.hours.iloc[-1] if pm is not None and len(pm) else 8.0
    r["интервал настоящий"] = win_T.hours.iloc[-1] if len(win_T) else np.nan
    r["интервал у бота"] = bot_h
    r["интервал совпал"] = bool(set(win_T.hours.unique()) == {bot_h})
    # что посчитал бы бот: берёт ставки в моменты, кратные своему интервалу, и делит на свой интервал
    step = int(bot_h * H)
    marks = [s for s in range(int(T // step * step), int(T - 24 * H), -step)]
    rates = [f[f.t <= s0 + 60_000].rate.iloc[-1] * 8 / bot_h * 100 for s0 in marks if len(f[f.t <= s0 + 60_000])]
    r["фандинг 24ч у бота"] = float(np.mean(rates)) if rates else np.nan
    r["шорт не потерян из-за интервала"] = bool(r["фандинг 24ч у бота"] <= -0.2)
    return r


def long_one(sym, T, kl):
    t_open = T - H4
    m = minutes(sym, dt.datetime.fromtimestamp(t_open / 1000, dt.timezone.utc).date())
    r = {"symbol": sym, "T": T}
    if m is None or kl is None:
        r["нет данных"] = True
        return r
    w = m[(m.open_time >= t_open) & (m.open_time < T - LONG_BEFORE)]
    prev = kl[(kl.open_time < t_open) & (kl.open_time >= t_open - 12 * H4)]
    norm = prev.volume.astype(float).median()
    r["объём заранее x"] = w.volume.sum() / norm if norm > 0 else np.nan
    r["объём итог x"] = float(kl[kl.open_time == t_open].volume.astype(float).iloc[0]) / norm if norm > 0 and (kl.open_time == t_open).any() else np.nan
    r["попал в отбор"] = bool(r["объём заранее x"] >= 2.5)
    return r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sh = pd.read_csv("results/delay8/short/oi_sim_trades.csv.gz")[["symbol", "open_time"]].drop_duplicates()
    lg = pd.read_csv("results/delay8/long/oi_sim_trades.csv.gz")[["symbol", "open_time"]].drop_duplicates()

    with ThreadPoolExecutor(12) as ex:
        S = pd.DataFrame(list(ex.map(lambda r: short_one(r.symbol, int(r.open_time) + H4), sh.itertuples())))
    kls = {}
    with ThreadPoolExecutor(8) as ex:
        for s, k in zip(lg.symbol.unique(), ex.map(Hst.load_klines, lg.symbol.unique())):
            kls[s] = k
    with ThreadPoolExecutor(12) as ex:
        L = pd.DataFrame(list(ex.map(lambda r: long_one(r.symbol, int(r.open_time) + H4, kls.get(r.symbol)), lg.itertuples())))

    for d in (S, L):
        d["сигнал (Киев)"] = pd.to_datetime(d["T"], unit="ms", utc=True).dt.tz_convert("Europe/Kyiv").dt.strftime("%Y-%m-%d %H:%M")
    S.to_csv(f"{OUT_DIR}/short_check.csv", index=False)
    L.to_csv(f"{OUT_DIR}/long_check.csv", index=False)

    s_ok = S[S.get("нет данных", pd.Series(False, index=S.index)).ne(True)]
    l_ok = L[L.get("нет данных", pd.Series(False, index=L.index)).ne(True)]
    out = ["# Проверка предварительного отбора живого бота\n",
           "## Шорт\n",
           f"Сигналов: {len(S)}, с данными: {len(s_ok)}.\n",
           f"- попали бы в отбор за 70 мин до закрытия: **{int(s_ok['попал в отбор'].sum())} из {len(s_ok)}**",
           f"- интервал фандинга совпал с прошлым месяцем: {int(s_ok['интервал совпал'].sum())} из {len(s_ok)}",
           f"- сигнал сохранился бы с интервалом «как у бота»: **{int(s_ok['шорт не потерян из-за интервала'].sum())} из {len(s_ok)}**",
           f"- оба условия вместе: **{int((s_ok['попал в отбор'] & s_ok['шорт не потерян из-за интервала']).sum())} из {len(s_ok)}**\n",
           "## Лонг\n",
           f"Сигналов: {len(L)}, с данными: {len(l_ok)}.\n",
           f"- попали бы в отбор за 22 мин до закрытия (объём уже ≥ 2,5x): **{int(l_ok['попал в отбор'].sum())} из {len(l_ok)}**",
           f"- объём к этому моменту, медиана: {l_ok['объём заранее x'].median():.1f}x (итог свечи {l_ok['объём итог x'].median():.1f}x), "
           f"минимум: {l_ok['объём заранее x'].min():.1f}x\n",
           "Подробно по каждому сигналу — short_check.csv и long_check.csv.\n"]
    open(f"{OUT_DIR}/prescan_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
