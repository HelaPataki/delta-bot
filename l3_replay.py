"""
Проверка живого L3: что бы нашла ТА ЖЕ формула на полном архиве Binance за последние дни, без короткого списка,
без предфильтра «L/S вчера < 1,0» и без Coinalyze. Сравниваем со списком сигналов, которые прислал бот.
Монеты и капа — те же, что у живого бота (l3_universe.json). Живого бота не трогает.
MODE=(пусто) — часть SHARD; MODE=combine — сводка.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import live_l3 as B

OUT = "results/l3_replay"
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
DAYS = int(os.environ.get("REPLAY_DAYS", "15"))
H = 3600 * 1000


def run_shard():
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    coins = json.load(open(B.UNI_FILE))["coins"]
    mine = [s for i, s in enumerate(sorted(coins)) if i % NSHARDS == SHARD]

    def work(s):
        try:
            k, met = B.archive(s)
            ser = B.condition_series(k, met, coins[s]["cap"], hours=24 * (DAYS + 1))
            rows = []
            for j, r in enumerate(ser):
                on = B.is_on(r)
                fresh = on and j >= 24 and not any(B.is_on(ser[x]) for x in range(j - 24, j))
                # почти-сигналы: не хватило чуть-чуть
                near_ls = (np.isfinite(r["up"]) and r["up"] >= B.UP_MIN and r["up"] > r["dn"]
                           and np.isfinite(r["ls"]) and B.LS_MAX <= r["ls"] < 0.9)
                near_up = (np.isfinite(r["ls"]) and r["ls"] < B.LS_MAX and 2.0 <= r["up"] < B.UP_MIN and r["up"] > r["dn"])
                if fresh or near_ls or near_up:
                    rows.append({"монета": s, "t": int(r["t"]), "сигнал": fresh, "почти: L/S 0,8–0,9": near_ls,
                                 "почти: сверху 2–3%": near_up, "сверху %капы": round(r["up"], 2),
                                 "снизу %капы": round(r["dn"], 2), "L/S все": round(r["ls"], 3),
                                 "капа $M": round(r["mcap"] / 1e6, 1), "цена": r["px"]})
            last = ser[-1]["t"] if ser else None
            return rows, (s, last, len(k))
        except Exception as e:  # noqa: BLE001
            print(s, repr(e), file=sys.stderr)
            return [], (s, None, 0)

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(work, mine))
    rows = [r for x in res for r in x[0]]
    pd.DataFrame(rows).to_csv(f"{OUT}/parts/r{SHARD}.csv", index=False)
    pd.DataFrame([x[1] for x in res], columns=["монета", "последний час", "свечей"]).to_csv(
        f"{OUT}/parts/c{SHARD}.csv", index=False)
    print(f"часть {SHARD}: монет {len(mine)}, строк {len(rows)}")


def combine():
    parts = [f"{OUT}/parts/{f}" for f in os.listdir(f"{OUT}/parts")]
    R = pd.concat([pd.read_csv(f) for f in parts if "/r" in f and os.path.getsize(f) > 5])
    C = pd.concat([pd.read_csv(f) for f in parts if "/c" in f])
    R["время (Киев)"] = R.t.map(lambda x: B.kyiv(x, "%d.%m %H:%M"))
    last = C["последний час"].dropna().max()
    first = last - DAYS * 24 * H
    R = R[R.t > first].sort_values("t")
    live = pd.read_csv(B.SIGNALS_FILE) if os.path.exists(B.SIGNALS_FILE) else pd.DataFrame(columns=["монета"])
    sig = R[R["сигнал"]].copy()
    sig["бот прислал"] = sig["монета"].isin(set(live.get("монета", [])))
    bot_start = pd.Timestamp("2026-11-12 22:00", tz="Europe/Kyiv").timestamp() * 1000
    weeks = DAYS / 7
    md = ["# L3: живой бот против полного архива", "",
          f"Архив по {B.kyiv(last, '%d.%m %H:%M')} (Киев) включительно, окно {DAYS} дней. "
          f"Монет: {len(C)}, с данными: {(C['свечей'] > 0).sum()}.", "",
          f"Сигналов по архиву за {DAYS} дней: **{len(sig)}** (≈{len(sig) / weeks:.1f} в неделю). "
          f"После запуска бота (12.11 22:00): {(sig.t >= bot_start).sum()}.", "",
          "## Сигналы по архиву", "",
          sig[["время (Киев)", "монета", "сверху %капы", "снизу %капы", "L/S все", "капа $M", "бот прислал"]]
          .to_markdown(index=False) if len(sig) else "нет", "",
          "## Почти-сигналы после запуска бота (не хватило L/S или ликвидаций)", ""]
    nr = R[(~R["сигнал"]) & (R.t >= bot_start)]
    if len(nr):
        g = nr.groupby("монета").agg(часов=("t", "size"), с=("время (Киев)", "first"), по=("время (Киев)", "last"),
                                    **{"сверху макс": ("сверху %капы", "max"), "L/S мин": ("L/S все", "min"),
                                       "из-за L/S": ("почти: L/S 0,8–0,9", "sum"),
                                       "из-за ликвидаций": ("почти: сверху 2–3%", "sum")})
        md.append(g.sort_values("часов", ascending=False).to_markdown())
    else:
        md.append("нет")
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/l3_replay_summary.md", "w").write("\n".join(md) + "\n")
    R.to_csv(f"{OUT}/l3_replay_rows.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    (combine if os.environ.get("MODE") == "combine" else run_shard)()
