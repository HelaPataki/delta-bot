"""ВРЕМЕННО (удалить): почему сверка не подтвердила лонги 09.11 02:00."""
import datetime as dt
import pandas as pd
import reconcile as R

H4 = 4 * 3600 * 1000
day = dt.date(2026, 11, 8)
T = int(pd.Timestamp("2026-11-09 00:00", tz="UTC").timestamp() * 1000)
uni = set(R.universe())
print("готов ли архив:", R.archive_ready(day))
for sym in ["PENDLEUSDT", "XVGUSDT", "SNXUSDT", "ZENUSDT", "TAIKOUSDT", "BANKUSDT"]:
    fr = [R.k4(sym, day - dt.timedelta(days=k)) for k in (2, 1, 0)]
    info = [None if f is None else (len(f), str(pd.to_datetime(f.open_time.max(), unit="ms"))) for f in fr]
    m = R.m1(sym, day)
    print(f"{sym}: в списке сверки {sym in uni}; 4ч файлы {info}; минуток {None if m is None else len(m)}; long_day → {R.long_day(sym, day)}")
