"""
Сделки после 4h-свечи с объёмом ×5+ (по итогам vol_after). Все USDT-перпетуалы Binance с делистнутыми, без акций, 2021–2026.

ШОРТ: большая зелёная ×5-свеча (выросла > +10% за 4 часа).
ЛОНГ: две идеи — большая красная ×5 (упала > −10%) и средняя красная (−10…−3%): отскок после капитуляции.
Сигнал: объём ≥ 5 × средний за 12 предыдущих 4h-свечей, оборот ≥ $50k, монете ≥ 7 дней; повтор у монеты в течение 20 дней — не новый сигнал
(как в vol_after).

Входы (по открытию 4h-свечи):
  сразу     — следующая свеча после сигнальной;
  через 1д  — через сутки; через 3д — через 3 суток (для шорта это пик в среднем);
  пробой    — шорт: первая 4h-свеча, закрывшаяся ниже ОТКРЫТИЯ сигнальной свечи (всплеск «съеден»), ждём до 5 дней;
              лонг: первая 4h-свеча, закрывшаяся выше ОТКРЫТИЯ сигнальной (падение «выкуплено»), ждём до 5 дней.
Выход: стоп / тейк (в % цены) / по времени. Если в одной свече задеты и стоп, и тейк — считаем стоп. Комиссия 0,1% за круг, фандинг не учтён.
Признаки для фильтров: BTC выше/ниже MA50 (дневной), сила объёма, фитили, оборот, возраст монеты.

Правила выбираем на 2021–2024, проверяем на 2025–2026.
MODE=shard (SHARD/NSHARDS) или MODE=combine.
"""

import glob
import itertools
import os

import numpy as np
import pandas as pd

os.environ.setdefault("START", "2021-01-01")
import history as Hst  # noqa: E402
import tradfi  # noqa: E402

DIR = "results/vol_trade"
MULT, LOOK, MIN_TURN, MIN_AGE, COOL = 5.0, 12, 50_000, 42, 120
FEE = 0.1
ENTRIES = {"сразу": 1, "через 1д": 6, "через 3д": 18, "пробой": None}
SHORT_GRID = dict(stop=[10, 15, 20, 30], take=[10, 15, 20, 30], hold=[3, 7, 14, 20])
LONG_GRID = dict(stop=[7, 10, 15, 20], take=[10, 15, 20, 30], hold=[3, 7, 14, 20])


def btc_regime():
    b = Hst.load_klines("BTCUSDT", "4h")
    ma = b.close.rolling(300).mean()            # 300 свечей 4h = 50 дней
    return pd.Series((b.close > ma).to_numpy(), index=b.open_time.astype("int64").to_numpy())


def trade(side, o, h, l, c, k, stop, take, hold):
    """Вход по открытию свечи k. Возвращает (pnl %, итог, часов)."""
    px = o[k]
    if side == "short":
        sl, tp = px * (1 + stop / 100), px * (1 - take / 100)
    else:
        sl, tp = px * (1 - stop / 100), px * (1 + take / 100)
    end = min(len(c) - 1, k + hold * 6 - 1)
    for j in range(k, end + 1):
        if side == "short":
            if h[j] >= sl:
                return -stop - FEE, "стоп", (j - k + 1) * 4
            if l[j] <= tp:
                return take - FEE, "тейк", (j - k + 1) * 4
        else:
            if l[j] <= sl:
                return -stop - FEE, "стоп", (j - k + 1) * 4
            if h[j] >= tp:
                return take - FEE, "тейк", (j - k + 1) * 4
    r = (c[end] / px - 1) * 100 * (-1 if side == "short" else 1) - FEE
    return r, "время", (end - k + 1) * 4


def analyse(sym, df, btc):
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    qv = df["quote_volume"].to_numpy()
    t = df.open_time.to_numpy().astype("int64")
    rows, last = [], -10**9
    for i in range(max(LOOK, MIN_AGE), len(df) - 1):
        base = v[i - LOOK:i].mean()
        if base <= 0 or v[i] < MULT * base or qv[i] < MIN_TURN or i - last <= COOL:
            continue
        last = i
        move = (c[i] / o[i] - 1) * 100
        if move > 10:
            kinds = [("short", "шорт: зелёная > +10%")]
        elif move < -10:
            kinds = [("long", "лонг: красная < −10%")]
        elif move < -3:
            kinds = [("long", "лонг: красная −10…−3%")]
        else:
            continue
        reg = btc.asof(t[i]) if len(btc) else np.nan
        feat = {"symbol": sym, "t": int(t[i]), "год": pd.Timestamp(int(t[i]), unit="ms").year,
                "свеча %": move, "объём x": v[i] / base, "x к 30д": v[i] / v[max(0, i - 180):i].mean(),
                "оборот $": qv[i], "возраст дн": i / 6, "BTC выше MA50": bool(reg) if reg == reg else None,
                "фитиль верх %": (h[i] / max(o[i], c[i]) - 1) * 100, "фитиль низ %": (1 - l[i] / min(o[i], c[i])) * 100}
        for side, name in kinds:
            grid = SHORT_GRID if side == "short" else LONG_GRID
            for ename, off in ENTRIES.items():
                if off is None:
                    k = None
                    for j in range(i + 1, min(i + 31, len(df) - 1)):
                        if (side == "short" and c[j] < o[i]) or (side == "long" and c[j] > o[i]):
                            k = j + 1
                            break
                    if k is None:
                        continue
                else:
                    k = i + off
                if k >= len(df):
                    continue
                for stop, take, hold in itertools.product(grid["stop"], grid["take"], grid["hold"]):
                    if k + hold * 6 > len(df):
                        continue                       # сделка ещё не закончилась бы — не считаем
                    pnl, res, hrs = trade(side, o, h, l, c, k, stop, take, hold)
                    rows.append({**feat, "идея": name, "вход": ename, "стоп": stop, "тейк": take, "держать дн": hold,
                                 "pnl %": round(pnl, 3), "итог": res, "часов": hrs, "вход t": int(t[k])})
    return rows


def shard():
    k, n = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
    btc = btc_regime()
    syms = [s for s in Hst.list_symbols() if s not in tradfi.TRADFI and not s.startswith(("BTCDOM", "USDC"))][k::n]
    rows = []
    for m, s in enumerate(syms):
        try:
            df = Hst.load_klines(s, "4h")
        except Exception as e:  # noqa: BLE001
            print(s, "ошибка", repr(e), flush=True)
            continue
        if df is None or len(df) < 200:
            continue
        rows += analyse(s, df, btc)
        if m % 20 == 0:
            print(f"{m}/{len(syms)} {s}: строк {len(rows)}", flush=True)
    os.makedirs(f"{DIR}/parts", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{DIR}/parts/tr{k}.csv.gz", index=False)


# ---------------- сводка ----------------
def stats(g):
    return pd.Series({"сделок": len(g), "в плюсе": (g["pnl %"] > 0).mean() * 100, "в среднем %": g["pnl %"].mean(),
                      "стопов": (g["итог"] == "стоп").mean() * 100})


def fmt(t):
    t = t.copy()
    for col in ("в плюсе", "стопов"):
        if col in t:
            t[col] = t[col].map(lambda x: f"{x:.0f}%")
    for col in [c for c in t.columns if "в среднем" in c or c.startswith("20")]:
        t[col] = t[col].map(lambda x: f"{x:+.2f}%" if x == x else "—")
    if "сделок" in t:
        t["сделок"] = t["сделок"].astype(int)
    return t.to_markdown()


def combine():
    parts = [pd.read_csv(f) for f in glob.glob(f"{DIR}/parts/*.csv.gz")]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d.to_csv(f"{DIR}/trades.csv.gz", index=False)
    d["период"] = np.where(d["год"] <= 2024, "2021–24", "2025–26")
    keys = ["идея", "вход", "стоп", "тейк", "держать дн"]
    out = ["# Сделки после 4h-свечи с объёмом ×5+", "",
           "Шорт после большой зелёной (> +10%), лонг после красной (< −10% и −10…−3%). Входы: сразу, через 1 или 3 дня, "
           "«пробой» — после закрытия 4h-свечи за открытием сигнальной (ждём до 5 дней). Стоп/тейк в % цены, комиссия 0,1%, "
           "фандинг не учтён. Если в свече задеты и стоп, и тейк — считаем стоп.", "",
           "Правила выбираем на **2021–24**, проверяем на **2025–26**.", ""]
    for idea, g in d.groupby("идея"):
        sel = g[g["период"] == "2021–24"].groupby(keys[1:]).apply(stats).reset_index()
        sel = sel[sel["сделок"] >= int(os.environ.get("MIN_N", 30))]
        if not len(sel):
            out += [f"## {idea}", "", "Мало сделок для выбора.", ""]
            continue
        chk = g[g["период"] == "2025–26"].groupby(keys[1:]).apply(stats).reset_index()
        m = sel.merge(chk, on=keys[1:], how="left", suffixes=(" 21–24", " 25–26"))
        years = g.groupby(keys[1:] + ["год"])["pnl %"].mean().unstack()
        top = m.sort_values("в среднем % 21–24", ascending=False).head(12)
        top = top.merge(years.reset_index(), on=keys[1:], how="left")
        out += [f"## {idea}", "",
                f"Сигналов: {g.drop_duplicates(['symbol', 't']).shape[0]}. Лучшие 12 вариантов по 2021–24 и как они прошли 2025–26, "
                "плюс средний результат на сделку по годам:", ""]
        cols = keys[1:] + ["сделок 21–24", "в плюсе 21–24", "в среднем % 21–24", "сделок 25–26", "в плюсе 25–26",
                           "в среднем % 25–26"] + [y for y in years.columns]
        t = top[cols].copy()
        for col in t.columns:
            if col in ("в плюсе 21–24", "в плюсе 25–26"):
                t[col] = t[col].map(lambda x: f"{x:.0f}%")
            elif str(col).startswith("в среднем") or isinstance(col, (int, np.integer)):
                t[col] = t[col].map(lambda x: f"{x:+.2f}%" if x == x else "—")
            elif str(col).startswith("сделок"):
                t[col] = t[col].astype(int)
        out += [t.to_markdown(index=False), ""]
        best = top.iloc[0]
        b = g[(g["вход"] == best["вход"]) & (g["стоп"] == best["стоп"]) & (g["тейк"] == best["тейк"]) &
              (g["держать дн"] == best["держать дн"])].copy()
        out += [f"### Лучший вариант ({best['вход']}, стоп {best['стоп']} / тейк {best['тейк']} / {best['держать дн']} дн.) — по признакам", ""]
        for feat, bins, labels in [
            ("BTC выше MA50", None, None),
            ("объём x", [5, 10, 20, 1e9], ["x5–10", "x10–20", "x20+"]),
            ("x к 30д", [0, 5, 10, 1e9], ["<5 к 30д", "5–10 к 30д", "10+ к 30д"]),
            ("оборот $", [0, 1e6, 5e6, 2e7, 1e12], ["<$1M", "$1–5M", "$5–20M", "$20M+"]),
            ("возраст дн", [0, 60, 365, 1e9], ["< 2 мес", "2–12 мес", "> 1 года"]),
            ("фитиль верх %", [-1, 2, 5, 1e9], ["верх. фитиль <2%", "2–5%", ">5%"]),
            ("фитиль низ %", [-1, 2, 5, 1e9], ["ниж. фитиль <2%", "2–5%", ">5%"]),
        ]:
            col = b[feat] if bins is None else pd.cut(b[feat], bins, labels=labels, right=False)
            t = b.groupby([col, "период"], observed=True)["pnl %"].agg(["size", "mean"]).unstack()
            t.columns = [f"{a} {p}" for a, p in t.columns]
            t = t.rename(columns=lambda x: x.replace("size", "сделок").replace("mean", "в среднем %"))
            for col_ in t.columns:
                t[col_] = t[col_].map((lambda x: f"{x:+.2f}%") if "в среднем" in col_ else (lambda x: f"{x:.0f}"))
            out += [f"**{feat}**", "", t.to_markdown(), ""]
    open(f"{DIR}/vol_trade_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    combine() if os.environ.get("MODE") == "combine" else shard()
