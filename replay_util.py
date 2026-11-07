"""ВРЕМЕННО (удалить вместе с replay_check.py): помощники проверки."""

import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd
import requests

H = 3600


def oi_at(bars, X, step=H):
    best = None
    for b in bars:
        if b["t"] + step <= X and X - (b["t"] + step) <= max(1800, step):
            best = b
    return best["c"] if best else np.nan


_iv = {}


def interval_as_live(sym, T):
    """Интервал так, как его увидел бы живой бот в день сигнала: последний в архиве за предыдущий месяц, иначе 4 ч."""
    d = dt.datetime.fromtimestamp(T, dt.timezone.utc).date().replace(day=1)
    pm = (d - dt.timedelta(days=1)).replace(day=1)
    key = (sym, pm)
    if key not in _iv:
        hours = None
        try:
            r = requests.get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{pm:%Y-%m}.zip", timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]))
                df.columns = [c.strip().lower() for c in df.columns]
                hours = float(df.funding_interval_hours.iloc[-1])
        except Exception:  # noqa: BLE001
            pass
        _iv[key] = hours or 4.0
    return _iv[key]
