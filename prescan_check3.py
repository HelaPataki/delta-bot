"""
Проверка более раннего предварительного отбора (под скорость Coinalyze): шорт — за 65 мин до закрытия,
лонг — за 42 мин. Использует функции прошлых проверок, их файлы не меняются.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import history as Hst
import prescan_check as PC
import prescan_check2 as PC2

OUT = "results/prescan_check3"
H4 = 4 * 3600 * 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    PC2.BEFORE, PC2.OUT_DIR = 65 * 60 * 1000, OUT
    PC2.main()
    s = open(f"{OUT}/prescan2_summary.md").read().replace("за 45 мин", "за 65 мин")
    PC.LONG_BEFORE = 42 * 60 * 1000
    lg = pd.read_csv("results/delay8/long/oi_sim_trades.csv.gz")[["symbol", "open_time"]].drop_duplicates()
    kls = {}
    with ThreadPoolExecutor(8) as ex:
        for sym, k in zip(lg.symbol.unique(), ex.map(Hst.load_klines, lg.symbol.unique())):
            kls[sym] = k
    with ThreadPoolExecutor(12) as ex:
        L = pd.DataFrame(list(ex.map(lambda r: PC.long_one(r.symbol, int(r.open_time) + H4, kls.get(r.symbol)), lg.itertuples())))
    L.to_csv(f"{OUT}/long_check3.csv", index=False)
    ok = L[L.get("нет данных", pd.Series(False, index=L.index)).ne(True)]
    s += ("\n## Лонг (за 42 мин до закрытия)\n\n"
          f"- попали бы в отбор (объём уже ≥ 2,5x): **{int(ok['попал в отбор'].sum())} из {len(ok)}**\n"
          f"- объём к этому моменту, медиана: {ok['объём заранее x'].median():.1f}x, минимум: {ok['объём заранее x'].min():.1f}x\n")
    open(f"{OUT}/prescan3_summary.md", "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
