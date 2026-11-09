"""ВРЕМЕННО (удалить): архивные цифры для лонгов 09.11 02:00, которые сверка не подтвердила."""
import datetime as dt
import numpy as np
import pandas as pd
import reconcile as R

H4 = 4 * 3600 * 1000
day = dt.date(2026, 11, 8)
T = int(pd.Timestamp("2026-11-09 00:00", tz="UTC").timestamp() * 1000)
for sym in ["PENDLEUSDT", "XVGUSDT", "SNXUSDT", "ZENUSDT", "TAIKOUSDT", "BANKUSDT"]:
    frames = [R.k4(sym, day - dt.timedelta(days=k)) for k in (2, 1, 0)]
    k = pd.concat([f for f in frames if f is not None]).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    i = int(np.where(k.open_time.to_numpy() == T - H4)[0][0])
    qv, v, tbv = k.quote_volume.to_numpy(), k.volume.to_numpy(), k.taker_buy_volume.to_numpy()
    ratio = qv[i] / np.median(qv[i - 12:i]); delta = (2 * tbv[i] - v[i]) / v[i] * 100
    mm = R.m1(sym, day)
    w = mm[(mm.open_time >= T - H4) & (mm.open_time < T)] if mm is not None else None
    if w is None or not len(w):
        print(sym, "минуток нет", round(ratio, 2), round(delta, 1)); continue
    top = w.loc[w.volume.idxmax()]; poc = (top.high + top.low) / 2
    hi, lo = k.high.iloc[i], k.low.iloc[i]
    print(f"{sym}: объём x{ratio:.2f} дельта {delta:.1f}% оборот {qv[i]:.0f} минуток {len(w)} POC {poc} "
          f"поз {(poc-lo)/(hi-lo)*100:.1f}% закрытие {k.close.iloc[i]} hi {hi} lo {lo} время POC {pd.to_datetime(top.open_time, unit='ms')}")
