"""Search for the project's 153-patient set by reproducing its recorded split counts.

The recorded counts are train 45/87, validation 12/23, test 20/9. The split procedure is
deterministic (sorted patients, random.Random(42).shuffle, 70/15/15), so a candidate patient
set can be scored without any image data.
"""
import csv
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
headers = {h["series_uid"]: h for h in json.loads(Path(sys.argv[2]).read_text())}
TARGET = {"train": (45, 87), "validation": (12, 23), "test": (20, 9)}

chosen = {}
for u, h in headers.items():
    p = h["patient_id"]
    if p not in chosen or (h["n_images"] or 0) > (headers[chosen[p]]["n_images"] or 0):
        chosen[p] = u
per = defaultdict(lambda: [0, 0])
for r in rows:
    if r["malignancy_mean"] in ("", "None") or r["series_uid"] != chosen[r["patient_id"]]:
        continue
    s = float(r["malignancy_mean"])
    if s >= 4:
        per[r["patient_id"]][0] += 1
    elif s <= 2:
        per[r["patient_id"]][1] += 1
pats = sorted(chosen)


def score(hset):
    plist = sorted(hset)
    random.Random(42).shuffle(plist)
    n = len(plist)
    nt, nv = int(n * 0.7), int(n * 0.15)
    c = {"train": [0, 0], "validation": [0, 0], "test": [0, 0]}
    place = {}
    for i, p in enumerate(plist):
        s = "train" if i < nt else "validation" if i < nt + nv else "test"
        place[p] = s
        c[s][0] += per[p][0]
        c[s][1] += per[p][1]
    dist = sum(abs(c[k][j] - TARGET[k][j]) for k in TARGET for j in (0, 1))
    return dist, c, place


base = pats[:153]
print("first 153:", base[0], "to", base[-1], score(base)[:2])
best = []
for pool_extra in range(0, 9):
    pool = pats[: 153 + pool_extra]
    for drop in itertools.combinations(pool[20:], pool_extra):
        h = [p for p in pool if p not in drop]
        if len(h) != 153:
            continue
        d, c, place = score(h)
        if place.get("LIDC-IDRI-0026") != "train":
            continue
        best.append((d, drop, c))
    best.sort(key=lambda x: x[0])
    best = best[:6]
    print("extra", pool_extra, "best distance", best[0][0] if best else None, flush=True)
    if pool_extra >= 3 or (best and best[0][0] == 0):
        break
for d, drop, c in best[:6]:
    print(d, drop, c)
Path(sys.argv[3]).write_text(json.dumps([{"distance": d, "dropped": list(drop), "counts": c} for d, drop, c in best], indent=1))
