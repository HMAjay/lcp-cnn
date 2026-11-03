"""Diameter-only AUC within size bands for the validation and test partitions of the rebuilt cohort.

    python size_bands.py nodules.csv headers.json size_bands.json

Same cohort, label rule and partition as cohort_153.py. Bands match tolerance_study/added_value.py.
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BANDS = [(0.0, 10.0), (10.0, 20.0), (20.0, 1e9)]


def auc(y, p):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
headers = {h["series_uid"]: h for h in json.loads(Path(sys.argv[2]).read_text())}
chosen = {}
for u, h in headers.items():
    p = h["patient_id"]
    if p not in chosen or (h["n_images"] or 0) > (headers[chosen[p]]["n_images"] or 0):
        chosen[p] = u
plist = sorted(chosen)[:153]
random.Random(42).shuffle(plist)
nt, nv = int(len(plist) * 0.7), int(len(plist) * 0.15)
assign = {p: "train" if i < nt else "validation" if i < nt + nv else "test" for i, p in enumerate(plist)}

labeled = []
for r in rows:
    if r["patient_id"] not in assign or r["series_uid"] != chosen[r["patient_id"]] or r["malignancy_mean"] in ("", "None") or r["diameter_mm"] in ("", "None", "nan"):
        continue
    s = float(r["malignancy_mean"])
    if 2 < s < 4:
        continue
    labeled.append({"split": assign[r["patient_id"]], "patient": r["patient_id"], "label": int(s >= 4), "diameter": float(r["diameter_mm"])})

out = {}
for name, keep in (("validation", {"validation"}), ("test", {"test"}), ("all", {"validation", "test"})):
    sel = [r for r in labeled if r["split"] in keep]
    entry = {"nodules": len(sel), "malignant": sum(r["label"] for r in sel), "patients": len({r["patient"] for r in sel}),
             "diameter_auc": auc([r["label"] for r in sel], [r["diameter"] for r in sel]), "size_bands": []}
    for lo, hi in BANDS:
        b = [r for r in sel if lo <= r["diameter"] < hi]
        entry["size_bands"].append({"from_mm": lo, "to_mm": None if hi > 1e8 else hi, "nodules": len(b), "malignant": sum(r["label"] for r in b),
                                    "diameter_auc": auc([r["label"] for r in b], [r["diameter"] for r in b])})
    out[name] = entry
Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(k, v["nodules"], v["malignant"], v["patients"], v["diameter_auc"], [(b["nodules"], b["malignant"], b["diameter_auc"]) for b in v["size_bands"]])
