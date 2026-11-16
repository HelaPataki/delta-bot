"""
Капитуляция, часть 3: как получить больше сигналов. Живых ботов не трогает. TradFi (и SNDK) исключены.

Принцип везде один: первый вынос — ловушка, вынос посреди серии — точка входа.
«Серия» = сколько разных суток из последних W дней содержали хотя бы один час-вынос (текущий час не считается).

1. РЫНОК, ЛОНГ, пороги мягче: час-слив = ≥20 / 25 / 30% ликвидных монет упали за час на 3%+;
   серия за 5 / 7 / 10 дней, нужно ≥1 / 2 / 3 дней. Корзины: «5 сильнее всех упавших», «весь рынок».
2. МОНЕТА, ЛОНГ: у отдельной монеты час-вынос = падение за час ≥3% и ≥3 своих обычных колебаний (σ часа за 30 дней);
   серия этой монеты за 7 дней ≥0…4 дней. Только 100 самых торгуемых монет. Одна монета — не чаще раза в 24 ч.
   Отдельно — было ли в этот час общее падение рынка (≥20% монет −3%+) или монета падала одна.
3. РЫНОК, ШОРТ (зеркало, «идея блогера + серия»): час-сквиз = ≥20 / 25 / 30% монет выросли за час на 3%+;
   шорт корзины «5 сильнее всех выросших» / «весь рынок».
4. МОНЕТА, ШОРТ: монета выросла за час ≥3% и ≥3σ, серия своих сквизов.
Вход — открытие следующего часа. Выходы: 72 ч; 24 ч; тейк/стоп 10/10; 72 ч с аварийным стопом 20%.
Комиссия 0,1% на круг, без плеча. Подбор — 2021–2023, проверка — 2024–2026.
ДЕПОЗИТ: $100, делим на 1 / 2 / 3 / 5 частей; новая сделка берёт свободную часть (1/N текущего капитала),
  если свободной нет — сигнал пропускается. Выход 72 ч.
"""

import os

os.environ.setdefault("START", "2020-12-01")

import numpy as np
import pandas as pd

import capitulation as C
import capitulation2 as C2
import history as Hst
from tradfi import TRADFI

OUT = os.environ.get("OUT_DIR", "results/capitulation3")
C.OUT = OUT
C2.OUT = OUT
H = C.H
EXITS = ["72ч", "24ч", "10/10", "72ч стоп20"]
FIT_END = 2023
EXTRA_SKIP = {"SNDKUSDT"}


def prep():
    os.makedirs(OUT, exist_ok=True)
    import json
    syms = [s for s in Hst.list_symbols()
            if not any(x in s for x in C.SKIP) and s not in TRADFI and s not in EXTRA_SKIP]
    json.dump(syms, open(f"{OUT}/universe.json", "w"))
    print("монет:", len(syms))


def trade(M, j, i0, side):
    o, h, l, c = M["o"][:, j], M["h"][:, j], M["l"][:, j], M["c"][:, j]
    if i0 + 72 > len(o):
        return None
    e = o[i0]
    if not np.isfinite(e) or e <= 0:
        return None
    hh, ll, cc = h[i0:i0 + 72], l[i0:i0 + 72], c[i0:i0 + 72]
    if not (np.isfinite(hh).all() and np.isfinite(ll).all() and np.isfinite(cc[[23, 71]]).all()):
        return None
    good = hh if side > 0 else ll
    bad = ll if side > 0 else hh
    move = lambda p: side * (p / e - 1) * 100
    adv = (bad <= e * 0.9) if side > 0 else (bad >= e * 1.1)
    fav = (good >= e * 1.1) if side > 0 else (good <= e * 0.9)
    js = int(np.argmax(adv)) if adv.any() else 72
    jt = int(np.argmax(fav)) if fav.any() else 72
    res = {"72ч": move(cc[71]) - C.FEE, "24ч": move(cc[23]) - C.FEE,
           "10/10": (move(cc[71]) if js == 72 and jt == 72 else (-10 if js <= jt else 10)) - C.FEE}
    st = (bad <= e * 0.8) if side > 0 else (bad >= e * 1.2)
    res["72ч стоп20"] = (-20 if st.any() else move(cc[71])) - C.FEE
    return res


def series_days(flag_col, i, W):
    a = max(0, i - 24 * W)
    fh = np.where(flag_col[a:i])[0]
    if not len(fh):
        return 0
    return len(set(((i - (fh + a) - 1) // 24).tolist()))


def slots_sim(g, n_slots, x="72ч"):
    """g: события (t, x, год) по времени. Депозит делится на n_slots частей."""
    cash, open_, eq_hist, taken, yrs = 100.0, [], [], 0, {}
    peak, dd = 100.0, 0.0
    for tt, v, y in zip(g.t, g[x], g.год):
        if not np.isfinite(v):
            continue
        open_.sort()
        while open_ and open_[0][0] <= tt:
            _, size, r = open_.pop(0)
            cash += size * (1 + max(r, -100) / 100)
            eq = cash + sum(s for _, s, _ in open_)
            peak = max(peak, eq)
            dd = max(dd, (peak - eq) / peak)
        if len(open_) >= n_slots:
            continue
        equity = cash + sum(s for _, s, _ in open_)
        size = min(cash, equity / n_slots)
        if size <= 0:
            continue
        cash -= size
        open_.append((tt + 72 * H, size, v))
        taken += 1
        yrs.setdefault(y, []).append(v)
    for _, size, r in open_:
        cash += size * (1 + max(r, -100) / 100)
    peak = max(peak, cash)
    dd = max(dd, (peak - cash) / peak)
    return {"взято сделок": taken, "итог $": cash, "макс. просадка, %": dd * 100}


def block(w, rows, title):
    w(f"### {title}\n\n" + pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n\n")


def describe(g, months, x="72ч"):
    fa, fb = g[g.год <= FIT_END], g[g.год > FIT_END]
    yrs = g.groupby("год")[x].mean()
    m_b = months - 36
    return {"сигналов": len(g), "в месяц (всего)": len(g) / months,
            "в месяц 2024–26": len(fb) / m_b if m_b > 0 else np.nan,
            "2021–23: средняя": fa[x].mean(), "2021–23: в плюс, %": (fa[x] > 0).mean() * 100,
            "2024–26: средняя": fb[x].mean(), "2024–26: в плюс, %": (fb[x] > 0).mean() * 100,
            "годов в плюсе": f"{(yrs > 0).sum()}/{len(yrs)}"}


def combine():
    syms, t, M = C.load_matrix()
    n, m = M["c"].shape
    print("матрица:", M["c"].shape)
    sets = C.liquid_sets(t, M)
    o, c = M["o"], M["c"]
    r1 = (c / o - 1).astype("float32")
    sig = pd.DataFrame(r1).rolling(720, min_periods=500).std().shift(1).to_numpy(dtype="float32")
    start = int(np.searchsorted(t, C.T0))
    months = (t[-1] - C.T0) / (30.44 * 24 * H)
    year = pd.to_datetime(t, unit="ms").year.to_numpy()

    # ---- доли упавших / выросших по ликвидному рынку ----
    sh_dn = np.full(n, np.nan)
    sh_up = np.full(n, np.nan)
    liquid = np.zeros((n, m), dtype=bool)
    for i in range(n):
        S = sets[i]
        if len(S) < 50:
            continue
        liquid[i, S] = True
        x = r1[i, S]
        x = x[np.isfinite(x)]
        if len(x) >= 50:
            sh_dn[i], sh_up[i] = (x <= -0.03).mean(), (x >= 0.03).mean()

    def market_events(share, thr, side):
        flag = np.nan_to_num(share) >= thr
        evs, last = [], -10 ** 9
        for i in np.where(flag)[0]:
            if i < start or i >= n - 73 or i - last < 24:
                continue
            last = i
            evs.append(i)
        return flag, evs

    def market_rows(evs, flag, side, label):
        rows = []
        for i in evs:
            S = sets[i]
            top50 = [j for j in S[:50] if np.isfinite(r1[i, j])]
            ext = sorted(top50, key=lambda j: side * r1[i, j])[:5]      # лонг: сильнее всех упавшие; шорт: выросшие
            feat = {"t": int(t[i]), "год": int(year[i]), "время (Киев)": C.kyiv(t[i]),
                    **{f"серия {W}д": series_days(flag, i, W) for W in (5, 7, 10)}}
            for bname, js in (("5 крайних", ext), ("весь рынок", list(S))):
                res = [r for r in (trade(M, j, i + 1, side) for j in js) if r]
                if res:
                    rows.append({**feat, "событие": label, "корзина": bname,
                                 "монеты": ",".join(syms[j].replace("USDT", "") for j in js[:5]),
                                 **{x: float(np.mean([r[x] for r in res])) for x in EXITS}})
        return rows

    allrows = []
    for side, share, nm in ((+1, sh_dn, "рынок лонг"), (-1, sh_up, "рынок шорт")):
        for thr in (0.20, 0.25, 0.30):
            flag, evs = market_events(share, thr, side)
            allrows += market_rows(evs, flag, side, f"{nm} {int(thr * 100)}%")
            print(nm, thr, len(evs))

    # ---- монеты по отдельности ----
    coin_rows = []
    for side, nm in ((+1, "монета лонг"), (-1, "монета шорт")):
        mv = -r1 if side > 0 else r1
        F = (mv >= 0.03) & (mv >= 3 * sig) & liquid
        mkt = np.nan_to_num(sh_dn if side > 0 else sh_up) >= 0.20
        for j in range(m):
            idx = np.where(F[:, j])[0]
            last = -10 ** 9
            col = F[:, j]
            for i in idx:
                if i < start or i >= n - 73 or i - last < 24:
                    continue
                last = i
                r = trade(M, j, i + 1, side)
                if not r:
                    continue
                coin_rows.append({"t": int(t[i]), "год": int(year[i]), "время (Киев)": C.kyiv(t[i]),
                                  "событие": nm, "монета": syms[j], "ход часа, %": float(r1[i, j] * 100),
                                  "серия 7д": series_days(col, i, 7), "рынок тоже": bool(mkt[i]), **r})
        print(nm, sum(1 for x in coin_rows if x["событие"] == nm))
    ev = pd.DataFrame(allrows)
    cv = pd.DataFrame(coin_rows)
    ev.to_csv(f"{OUT}/market_events.csv.gz", index=False)
    cv.to_csv(f"{OUT}/coin_events.csv.gz", index=False)
    summary(ev, cv, months)


def summary(ev, cv, months):
    w = open(f"{OUT}/capitulation3_summary.md", "w").write
    w("# Капитуляция, часть 3: больше сигналов\n\n" + __doc__.split("\n", 2)[2] + "\n")
    w("«в месяц» — сколько сигналов в среднем (сделок будет меньше, если депозит занят). Результаты — % цены, без плеча.\n\n")
    for side_nm in ("рынок лонг", "рынок шорт"):
        w(f"## {side_nm.upper()}\n\n")
        for b in ("5 крайних", "весь рынок"):
            rows = []
            for thr in (20, 25, 30):
                e = ev[(ev.событие == f"{side_nm} {thr}%") & (ev.корзина == b)]
                for W in (5, 7, 10):
                    for k in (0, 1, 2, 3):
                        g = e[e[f"серия {W}д"] >= k]
                        if len(g) < 10:
                            continue
                        rows.append({"порог": f"{thr}%", "окно": f"{W}д", "серия ≥": k, **describe(g, months)})
            block(w, rows, f"Корзина «{b}», выход 72ч")
        rows = []
        for thr in (20, 25, 30):
            for W, k in ((7, 0), (7, 2), (7, 3), (5, 2), (10, 3)):
                e = ev[(ev.событие == f"{side_nm} {thr}%") & (ev.корзина == "5 крайних") & (ev[f"серия {W}д"] >= k)]
                e = e.sort_values("t")
                for x in ("72ч", "10/10", "72ч стоп20"):
                    if not len(e):
                        continue
                    d = describe(e, months, x)
                    rows.append({"порог": f"{thr}%", "серия": f"≥{k} за {W}д", "выход": x,
                                 "2021–23": d["2021–23: средняя"], "2024–26": d["2024–26: средняя"],
                                 "годов в плюсе": d["годов в плюсе"]})
        block(w, rows, "Корзина «5 крайних» — разные выходы")
        rows = []
        for thr in (20, 25, 30):
            for W, k in ((7, 2), (7, 3), (5, 2), (10, 3)):
                e = ev[(ev.событие == f"{side_nm} {thr}%") & (ev.корзина == "5 крайних") & (ev[f"серия {W}д"] >= k)]
                e = e.sort_values("t")
                for ns in (1, 2, 3, 5):
                    rows.append({"порог": f"{thr}%", "серия": f"≥{k} за {W}д", "частей": ns, **slots_sim(e, ns)})
        block(w, rows, "Депозит $100, корзина «5 крайних», 72ч")

    for nm in ("монета лонг", "монета шорт"):
        w(f"## {nm.upper()}\n\n")
        e = cv[cv.событие == nm]
        rows = []
        for alone in (None, True, False):
            g0 = e if alone is None else e[e["рынок тоже"] == (not alone)]
            lab = "все" if alone is None else ("монета одна" if alone else "вместе с рынком")
            for k in (0, 1, 2, 3, 4):
                g = g0[g0["серия 7д"] >= k]
                if len(g) < 10:
                    continue
                rows.append({"когда": lab, "серия 7д ≥": k, **describe(g, months)})
        block(w, rows, "Выход 72ч")
        rows = []
        for k in (2, 3):
            for alone in (None, True):
                g = e[e["серия 7д"] >= k]
                if alone:
                    g = g[~g["рынок тоже"]]
                g = g.sort_values("t")
                for x in ("72ч", "10/10", "72ч стоп20"):
                    d = describe(g, months, x)
                    rows.append({"серия ≥": k, "когда": "монета одна" if alone else "все", "выход": x,
                                 "2021–23": d["2021–23: средняя"], "2024–26": d["2024–26: средняя"],
                                 "годов в плюсе": d["годов в плюсе"]})
                for ns in (3, 5):
                    rows.append({"серия ≥": k, "когда": "монета одна" if alone else "все", "выход": f"депозит, {ns} частей",
                                 **slots_sim(g, ns)})
        block(w, rows, "Выходы и депозит")
        yr = e[e["серия 7д"] >= 3].groupby("год")["72ч"].agg(["size", "mean"]).round(2)
        w("По годам (серия ≥3, выход 72ч): " + ", ".join(f"{y}: {r['mean']:+.2f}% ({int(r['size'])})" for y, r in yr.iterrows()) + "\n\n")
    print(open(f"{OUT}/capitulation3_summary.md").read()[:5000])


if __name__ == "__main__":
    mode = os.environ.get("MODE", "")
    if mode == "prep":
        prep()
    elif mode == "combine":
        combine()
    else:
        C.run_shard()
