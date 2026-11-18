"""
Сбор графиков разблокировок из открытых данных DefiLlama (defillama-datasets.llama.fi — то, чем питается их сайт).
Только сбор. Живых ботов не трогает.

Выход (results/unlocks):
  • unlocks_meta.csv — по протоколу все «простые» поля верхнего уровня и метаданных (токен, coingecko id и т. п.),
    список категорий и сколько в каждой всего;
  • unlocks_daily.csv.gz — по протоколу и дню: разблокировано всего (накопительно) и отдельно «инсайдеры»
    (команда, инвесторы, советники, private/seed, основатели, контрибьюторы);
  • structure.txt — как устроен один файл (для проверки).
"""

import csv
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://defillama-datasets.llama.fi"
OUT = "results/unlocks"
INSIDER = re.compile(r"team|investor|advis|private|seed|founder|contributor|backer|strategic|core|insider|employee|vc|series",
                     re.I)


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                        timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if k == tries - 1:
                print(url, repr(e), file=sys.stderr)
            time.sleep(2 + 3 * k)
    return None


def scalars(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[prefix + k] = v
            elif isinstance(v, dict) and len(prefix) < 30:
                out.update(scalars(v, prefix + k + "."))
            elif isinstance(v, list) and v and all(isinstance(x, (str, int, float)) for x in v) and len(v) < 20:
                out[prefix + k] = "|".join(map(str, v))
    return out


def structure(d, depth=0, lines=None):
    lines = [] if lines is None else lines
    if depth > 4:
        return lines
    if isinstance(d, dict):
        for k, v in list(d.items())[:25]:
            t = type(v).__name__ + (f"[{len(v)}]" if isinstance(v, (list, dict)) else f" = {str(v)[:80]}")
            lines.append("  " * depth + f"{k}: {t}")
            structure(v, depth + 1, lines)
    elif isinstance(d, list) and d:
        lines.append("  " * depth + "[0]:")
        structure(d[0], depth + 1, lines)
    return lines


def series(doc):
    """Список (label, [(ts, unlocked)]) из documentedData.data, иначе из data."""
    for path in (("documentedData", "data"), ("data",)):
        x = doc
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if isinstance(x, list) and x and isinstance(x[0], dict) and "data" in x[0]:
            res = []
            for cat in x:
                pts = [(int(p["timestamp"]), float(p.get("unlocked") or 0)) for p in cat.get("data", [])
                       if isinstance(p, dict) and "timestamp" in p]
                res.append((str(cat.get("label", "?")), pts))
            return "/".join(path), res
    return None, []


def main():
    os.makedirs(OUT, exist_ok=True)
    names = get(f"{BASE}/emissionsProtocolsList") or []
    print("протоколов:", len(names))
    docs = {}

    def one(n):
        return n, get(f"{BASE}/emissions/{n}")

    with ThreadPoolExecutor(6) as ex:
        for n, d in ex.map(one, names):
            if d:
                docs[n] = d
    print("скачано:", len(docs))
    if "arbitrum" in docs:
        open(f"{OUT}/structure.txt", "w").write("\n".join(structure(docs["arbitrum"])))
    meta_rows, keys = [], set()
    with gzip.open(f"{OUT}/unlocks_daily.csv.gz", "wt", newline="") as gz:
        wr = csv.writer(gz)
        wr.writerow(["protocol", "date", "unlocked_total", "unlocked_insiders"])
        for n, d in sorted(docs.items()):
            src, cats = series(d)
            m = {"protocol": n, "source": src, "categories": "|".join(c for c, _ in cats),
                 "insider_categories": "|".join(c for c, _ in cats if INSIDER.search(c)),
                 "category_totals": "|".join(f"{c}={max((u for _, u in p), default=0):.0f}" for c, p in cats)}
            for k, v in scalars(d).items():
                m[k] = v
            meta_rows.append(m)
            keys |= set(m)
            tot, ins = {}, {}
            for c, pts in cats:
                for ts, u in pts:
                    day = time.strftime("%Y-%m-%d", time.gmtime(ts))
                    tot[day] = tot.get(day, 0) + u
                    if INSIDER.search(c):
                        ins[day] = ins.get(day, 0) + u
            for day in sorted(tot):
                wr.writerow([n, day, round(tot[day], 2), round(ins.get(day, 0), 2)])
    first = ["protocol", "source", "categories", "insider_categories", "category_totals"]
    with open(f"{OUT}/unlocks_meta.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=first + sorted(keys - set(first)))
        wr.writeheader()
        wr.writerows(meta_rows)
    print("готово:", len(meta_rows), "протоколов; поля:", sorted(keys)[:60])


if __name__ == "__main__":
    main()
