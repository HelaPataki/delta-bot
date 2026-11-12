"""
Проверка перед живым ботом L3: отдаёт ли Coinalyze те же данные, что архив Binance, на котором считалась история.
  • OI по часам (для карты ликвидаций) — насколько далеко в прошлое и совпадает ли с архивом;
  • соотношение лонгов/шортов по аккаунтам — совпадает ли с архивным count_long_short_ratio («все аккаунты»).
Живого бота не трогает. Пауза между запросами — бережём общий ключ.
"""

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

import pump_life as PL

KEY = os.environ.get("COINALYZE_KEY", "")
OUT = "results/cz_check"
SYMS = ["BTCUSDT", "SOLUSDT", "STGUSDT", "HUMAUSDT", "AGLDUSDT", "GIGGLEUSDT"]
H = 3600


def cz(endpoint, sym, frm, to, extra=None):
    params = {"symbols": f"{sym}_PERP.A", "interval": "1hour", "from": int(frm), "to": int(to), **(extra or {})}
    for _ in range(6):
        time.sleep(3)
        r = requests.get(f"https://api.coinalyze.net/v1/{endpoint}", headers={"api_key": KEY}, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 30)) + 2)
            continue
        if r.status_code != 200:
            return None, f"{r.status_code} {r.text[:150]}"
        js = r.json()
        return (pd.DataFrame(js[0]["history"]) if js and js[0].get("history") else pd.DataFrame()), None
    return None, "429"


def main():
    os.makedirs(OUT, exist_ok=True)
    now = int(time.time())
    frm = now - 120 * 86400
    lines = ["# Coinalyze против архива Binance\n", __doc__.split("\n", 2)[2], ""]
    rows = []
    for sym in SYMS:
        oi, e1 = cz("open-interest-history", sym, frm, now, {"convert_to_usd": "false"})
        ls, e2 = cz("long-short-ratio-history", sym, frm, now)
        days = {dt.date.today() - dt.timedelta(days=d) for d in range(3, 63)}
        met = PL.metrics(sym, days)
        r = {"монета": sym}
        if oi is None:
            r["OI"] = f"ошибка: {e1}"
        elif len(oi):
            oi["t"] = oi["t"].astype(int)
            r["OI: точек"] = len(oi)
            r["OI: с какой даты"] = str(pd.Timestamp(oi.t.min(), unit="s").date())
            r["OI: дней истории"] = round((oi.t.max() - oi.t.min()) / 86400, 1)
            if met is not None:
                b = met.assign(t=(met.t // 1000 // H) * H).groupby("t").oi.last()
                c = oi.set_index("t")["c"].astype(float)
                j = pd.concat([c, b], axis=1, keys=["cz", "bn"]).dropna()
                if len(j) > 48:
                    r["OI: Coinalyze / Binance (медиана)"] = round((j.cz / j.bn).median(), 3)
                    r["OI: совпадение часовых изменений (корр.)"] = round(j.cz.diff().corr(j.bn.diff()), 3)
        if ls is None:
            r["L/S"] = f"ошибка: {e2}"
        elif len(ls):
            ls["t"] = ls["t"].astype(int)
            r["L/S: точек"] = len(ls)
            r["L/S: поля"] = ",".join(c for c in ls.columns if c != "t")
            if met is not None and "r" in ls:
                for name, col in (("все аккаунты", "glob"), ("топы (позиции)", "top_pos")):
                    b = met.assign(t=(met.t // 1000 // H) * H).groupby("t")[col].last()
                    j = pd.concat([ls.set_index("t")["r"].astype(float), b], axis=1, keys=["cz", "bn"]).dropna()
                    if len(j) > 48:
                        r[f"L/S vs {name}: корр."] = round(j.cz.corr(j.bn), 3)
                        r[f"L/S vs {name}: средняя разница"] = round((j.cz - j.bn).abs().mean(), 3)
        rows.append(r)
        print(r)
    df = pd.DataFrame(rows)
    lines.append(df.T.to_markdown())
    open(f"{OUT}/cz_check.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
