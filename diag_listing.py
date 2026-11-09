"""ВРЕМЕННО (удалить): похоже ли на акцию — доля объёма в выходные и в часы биржи США (октябрь 2026, 1ч свечи)."""
import io, zipfile
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor

SYMS = "NOW NVDA META SPY EDGE ESP KAT MEGA BTC CAT O BILL RE CAP STAR CHIP SPACE OPG CRCL SPCX EWT".split()


def stat(sym):
    u = f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}USDT/1h/{sym}USDT-1h-2026-10.zip"
    r = requests.get(u, timeout=60)
    if r.status_code != 200:
        return sym, "нет файла"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        d = pd.read_csv(z.open(z.namelist()[0]))
    d.columns = [c.strip().lower() for c in d.columns]
    t = pd.to_datetime(d.open_time, unit="ms", utc=True)
    q = d.quote_volume.astype(float)
    wk = q[t.dt.weekday >= 5].mean() / q[t.dt.weekday < 5].mean()
    us = q[(t.dt.weekday < 5) & (t.dt.hour >= 14) & (t.dt.hour < 20)].sum() / q[t.dt.weekday < 5].sum()
    return sym, f"выходные/будни {wk:.2f}, доля часов США {us:.0%}"


with ThreadPoolExecutor(8) as ex:
    for s, m in ex.map(stat, SYMS):
        print(f"{s}: {m}")
