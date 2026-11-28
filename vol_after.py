"""
Что происходит в течение 20 дней после 4h-свечи с объёмом ×5 и больше (2021–2026, все USDT-перпетуалы Binance,
включая делистнутые; акции/фонды/сырьё исключены).

Сигнал: объём 4h-свечи ≥ 5 × средний объём 12 предыдущих 4h-свечей, оборот свечи ≥ $50k, монете на фьючерсах ≥ 7 дней.
Если у монеты новый сигнал пришёл в течение 20 дней после прошлого — не считаем его отдельным (это та же история),
но отмечаем, сколько ещё ×5-свечей было за 20 дней.

Уровни сигнальной свечи: с фитилём — максимум/минимум; без фитиля — верх/низ тела (большее/меньшее из открытия и закрытия).
Что смотрим за 20 дней (120 свечей 4h):
  • пробит ли максимум/минимум — касанием (фитилём следующих свечей) и закрытием 4h-свечи за уровнем, что раньше и через сколько дней;
  • то же для верха/низа тела (закрытием за уровнем);
  • цена через 1/3/5/10/20 дней к закрытию сигнальной свечи, максимальный рост и падение за 20 дней и в какой день;
  • куда ушёл график за 20 дней: вверх (> +10%), вниз (< −10%), боком.
Разрезы: цвет свечи, сила объёма, размер свечи, год. Для сравнения — обычные свечи тех же монет (раз в 10 дней).

MODE=shard (SHARD/NSHARDS) — расчёт части монет; MODE=combine — сводка.
"""

import glob
import os

import numpy as np
import pandas as pd

os.environ.setdefault("START", "2021-01-01")
import history as Hst  # noqa: E402
import tradfi  # noqa: E402

DIR = "results/vol_after"
MULT, LOOK, MIN_TURN, MIN_AGE = 5.0, 12, 50_000, 42
N = 120                                  # 20 дней по 6 свечей
DAYS = [1, 3, 5, 10, 20]


def first(mask):
    i = np.flatnonzero(mask)
    return int(i[0]) + 1 if len(i) else 0          # номер свечи после сигнала (1…120), 0 — не было


def analyse(sym, df):
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    qv = df["quote_volume"].to_numpy() if "quote_volume" in df else v * c
    t = df.open_time.to_numpy()
    out, last = [], -10**9
    for i in range(max(LOOK, MIN_AGE), len(df)):
        base = v[i - LOOK:i].mean()
        if base <= 0:
            continue
        # для сравнения: обычные свечи той же монеты (каждая 60-я, т.е. раз в 10 дней), без всплеска объёма
        control = i % 60 == 0 and v[i] < 2 * base and qv[i] >= MIN_TURN
        if not control:
            if v[i] < MULT * base or qv[i] < MIN_TURN:
                continue
            if i - last <= N:
                continue
            last = i
        H, L, C, O = h[i], l[i], c[i], o[i]
        BT, BB = max(O, C), min(O, C)
        j = slice(i + 1, i + 1 + N)
        hh, ll, cc = h[j], l[j], c[j]
        full = len(cc) == N
        vv = v[j]
        more = int(sum(1 for k in range(i + 1, min(i + 1 + N, len(df)))
                       if v[k - LOOK:k].mean() > 0 and v[k] >= MULT * v[k - LOOK:k].mean()))
        r = {"тип": "контроль" if control else "сигнал", "symbol": sym, "t": int(t[i]), "год": pd.Timestamp(int(t[i]), unit="ms").year,
             "цвет": "зелёная" if C >= O else "красная", "объём x": v[i] / base,
             "свеча %": (C / O - 1) * 100, "размах %": (H / L - 1) * 100,
             "фитиль верх %": (H / BT - 1) * 100, "фитиль низ %": (1 - L / BB) * 100,
             "оборот $": qv[i], "полные 20д": full, "ещё x5 за 20д": more,
             # тот же всплеск относительно нормы за 7 и 30 дней (для разреза «всплеск и на длинной норме»)
             "x к 7д": v[i] / v[max(0, i - 42):i].mean(), "x к 30д": v[i] / v[max(0, i - 180):i].mean(),
             "касание max": first(hh > H), "касание min": first(ll < L),
             "закрытие выше max": first(cc > H), "закрытие ниже min": first(cc < L),
             "закрытие выше тела": first(cc > BT), "закрытие ниже тела": first(cc < BB)}
        for d in DAYS:
            k = d * 6 - 1
            r[f"{d}д %"] = (cc[k] / C - 1) * 100 if len(cc) > k else np.nan
        if len(hh):
            r["макс рост %"] = (hh.max() / C - 1) * 100
            r["макс падение %"] = (ll.min() / C - 1) * 100
            r["день макс"] = (int(hh.argmax()) + 1) / 6
            r["день мин"] = (int(ll.argmin()) + 1) / 6
        out.append(r)
    return out


def shard():
    k, n = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
    syms = [s for s in Hst.list_symbols() if s not in tradfi.TRADFI and not s.startswith(("BTCDOM", "USDC"))]
    syms = syms[k::n]
    rows = []
    for m, s in enumerate(syms):
        try:
            df = Hst.load_klines(s, "4h")
        except Exception as e:  # noqa: BLE001
            print(s, "ошибка", repr(e), flush=True)
            continue
        if df is None or len(df) < 200:
            continue
        rows += analyse(s, df)
        if m % 20 == 0:
            print(f"{m}/{len(syms)} {s}: сигналов всего {len(rows)}", flush=True)
    os.makedirs(f"{DIR}/parts", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{DIR}/parts/sig{k}.csv.gz", index=False)


# ---------------- сводка ----------------
def pct(s):
    return f"{s.mean() * 100:.0f}%"


def med(s, d=1):
    s = s.dropna()
    return f"{s.median():+.{d}f}%" if len(s) else "—"


def days(col, d):
    x = d[col][d[col] > 0] / 6
    return f"{x.median():.1f}" if len(x) else "—"


def block(d):
    """Основная таблица для группы сигналов (только с полными 20 днями)."""
    n = len(d)
    up_first = ((d["закрытие выше max"] > 0) & ((d["закрытие ниже min"] == 0) | (d["закрытие выше max"] < d["закрытие ниже min"])))
    dn_first = ((d["закрытие ниже min"] > 0) & ((d["закрытие выше max"] == 0) | (d["закрытие ниже min"] < d["закрытие выше max"])))
    t_up = ((d["касание max"] > 0) & ((d["касание min"] == 0) | (d["касание max"] < d["касание min"])))
    t_dn = ((d["касание min"] > 0) & ((d["касание max"] == 0) | (d["касание min"] < d["касание max"])))
    b_up = ((d["закрытие выше тела"] > 0) & ((d["закрытие ниже тела"] == 0) | (d["закрытие выше тела"] < d["закрытие ниже тела"])))
    b_dn = ((d["закрытие ниже тела"] > 0) & ((d["закрытие выше тела"] == 0) | (d["закрытие ниже тела"] < d["закрытие выше тела"])))
    trend = np.select([d["20д %"] > 10, d["20д %"] < -10], ["вверх", "вниз"], "боком")
    return {
        "сигналов": n,
        "фитиль: касание max / min за 20д": f"{pct(d['касание max'] > 0)} / {pct(d['касание min'] > 0)}",
        "фитиль: первым коснулся max / min": f"{pct(t_up)} / {pct(t_dn)}",
        "закрытие за max / за min за 20д": f"{pct(d['закрытие выше max'] > 0)} / {pct(d['закрытие ниже min'] > 0)}",
        "закрытием первым пробит max / min / ничего": f"{pct(up_first)} / {pct(dn_first)} / {pct(~up_first & ~dn_first)}",
        "дней до закрытия за max / за min (медиана)": f"{days('закрытие выше max', d)} / {days('закрытие ниже min', d)}",
        "тело: первым пробит верх / низ (закрытием)": f"{pct(b_up)} / {pct(b_dn)}",
        "цена через 1 / 3 / 5 / 10 / 20 дн. (медиана)": " / ".join(med(d[f'{x}д %']) for x in DAYS),
        "выше закрытия свечи через 20 дн.": pct(d["20д %"] > 0),
        "за 20 дн.: вверх >10% / вниз >10% / боком": f"{pct(trend == 'вверх')} / {pct(trend == 'вниз')} / {pct(trend == 'боком')}",
        "макс. рост / макс. падение за 20д (медиана)": f"{med(d['макс рост %'])} / {med(d['макс падение %'])}",
        "день макс. / день мин. (медиана)": f"{d['день макс'].median():.1f} / {d['день мин'].median():.1f}",
    }


def table(groups):
    cols = list(groups)
    rows = list(next(iter(groups.values())).keys())
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        lines.append(f"| {r} | " + " | ".join(str(groups[c][r]) for c in cols) + " |")
    return "\n".join(lines)


def combine():
    parts = [pd.read_csv(f) for f in glob.glob(f"{DIR}/parts/*.csv.gz")]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d.to_csv(f"{DIR}/signals.csv.gz", index=False)
    ctrl = d[(d["тип"] == "контроль") & d["полные 20д"]]
    d = d[d["тип"] == "сигнал"]
    f = d[d["полные 20д"]].copy()
    f["сила"] = pd.cut(f["объём x"], [5, 10, 20, 1e9], right=False, labels=["x5–10", "x10–20", "x20+"])
    f["размер"] = pd.cut(f["свеча %"], [-1e9, -10, -3, 3, 10, 1e9], labels=["< −10%", "−10…−3%", "−3…+3%", "+3…+10%", "> +10%"])
    out = ["# Что происходит за 20 дней после 4h-свечи с объёмом ×5+", "",
           f"Все USDT-перпетуалы Binance (с делистнутыми, без акций), 2021-01 … {pd.Timestamp(d.t.max(), unit='ms'):%Y-%m-%d}. "
           f"Сигналов: {len(d)}, из них с полными 20 днями после: {len(f)} (в таблицах — только они). "
           "Повторный ×5 у той же монеты в течение 20 дней — не новый сигнал.", "",
           "«Закрытие за max» — 4h-свеча закрылась выше максимума сигнальной свечи (с фитилём); «касание» — хватило фитиля. "
           "«Тело» — верх/низ тела сигнальной свечи (для зелёной верх тела = её закрытие). Цены — к закрытию сигнальной свечи.", "",
           "## Все сигналы и по цвету свечи", "",
           "«Обычные свечи» — для сравнения: свечи тех же монет без всплеска объёма (раз в 10 дней). Альты в целом дешевеют, "
           "поэтому смотреть надо на разницу с ними, а не на сами цифры.", "",
           table({"все": block(f), "зелёная": block(f[f["цвет"] == "зелёная"]), "красная": block(f[f["цвет"] == "красная"]),
                  "обычные свечи": block(ctrl)}), "",
           "## Обычные свечи по годам (для сравнения)", "",
           table({str(k): block(g) for k, g in ctrl.groupby("год")}), "",
           "## По размеру сигнальной свечи (закрытие к открытию)", "",
           table({str(k): block(g) for k, g in f.groupby("размер", observed=True)}), "",
           "## Всплеск и на длинной норме (объём ×5 и к среднему за 7 / 30 дней)", "",
           table({"×5 только к 3 дням": block(f[f["x к 7д"] < 5]), "×5 и к 7 дням": block(f[f["x к 7д"] >= 5]),
                  "×5 и к 30 дням": block(f[f["x к 30д"] >= 5])}), "",
           "## По силе объёма", "",
           table({str(k): block(g) for k, g in f.groupby("сила", observed=True)}), "",
           "## По годам", "",
           table({str(k): block(g) for k, g in f.groupby("год")}), ""]
    for color in ("зелёная", "красная"):
        out += [f"## По годам — {color}", "", table({str(k): block(g) for k, g in f[f["цвет"] == color].groupby("год")}), ""]
    out += ["## Ещё ×5-свечей у той же монеты в следующие 20 дней", "",
            f["ещё x5 за 20д"].clip(upper=5).value_counts().sort_index().rename(index={5: "5+"}).to_frame("сигналов").to_markdown(), ""]
    open(f"{DIR}/vol_after_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    combine() if os.environ.get("MODE") == "combine" else shard()
