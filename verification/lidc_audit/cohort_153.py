"""Reconstruct the project cohort (LIDC-IDRI-0001 to 0153) and score a size-only baseline on it."""
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
headers = {h["series_uid"]: h for h in json.loads(Path(sys.argv[2]).read_text())}
for r in rows:
    for k in ("malignancy_mean", "diameter_mm", "volume_mm3", "x", "y", "z", "dx_mm", "dy_mm", "displacement_mm", "origin_x", "origin_y", "pixel_spacing", "fov_x_mm", "fov_y_mm"):
        r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
    r["n_distinct_readers"] = int(r["n_distinct_readers"])
    s = r["malignancy_mean"]
    r["label"] = None if math.isnan(s) else 1 if s >= 4 else 0 if s <= 2 else None

chosen = {}
for u, h in headers.items():
    p = h["patient_id"]
    if p not in chosen or (h["n_images"] or 0) > (headers[chosen[p]]["n_images"] or 0):
        chosen[p] = u
all_patients = sorted(chosen)
cohort = all_patients[:153]


def split(patients, seed):
    plist = sorted(patients)
    random.Random(seed).shuffle(plist)
    n = len(plist)
    nt, nv = int(n * 0.7), int(n * 0.15)
    return {p: ("train" if i < nt else "validation" if i < nt + nv else "test") for i, p in enumerate(plist)}


def auc(y, p):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(sel, n=4000, seed=42):
    groups = defaultdict(list)
    for r in sel:
        groups[r["patient_id"]].append(r)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = [x for i in rng.integers(0, len(ids), len(ids)) for x in groups[ids[i]]]
        v = auc([x["label"] for x in s], [x["diameter_mm"] for x in s])
        if not math.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fit_logistic(x, y):
    x, y = np.asarray(x), np.asarray(y)

    def nll(w):
        z = w[0] + w[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)

    return minimize(nll, np.zeros(2), method="BFGS").x


assign = split(cohort, 42)
keep = [r for r in rows if r["series_uid"] == chosen[r["patient_id"]] and r["patient_id"] in assign]
rated = [r for r in keep if not math.isnan(r["malignancy_mean"])]
labeled = [r for r in rated if r["label"] is not None and not math.isnan(r["diameter_mm"])]
for r in labeled:
    r["split"] = assign[r["patient_id"]]
out = {"patient_range": [cohort[0], cohort[-1]], "rated": len(rated), "labeled": len(labeled), "indeterminate": len(rated) - len(labeled),
       "single_reader_labeled": sum(r["n_distinct_readers"] == 1 for r in labeled), "partitions": {}}
for s in ("train", "validation", "test"):
    sel = [r for r in labeled if r["split"] == s]
    out["partitions"][s] = {"n": len(sel), "malignant": sum(r["label"] for r in sel), "benign": sum(1 - r["label"] for r in sel),
                            "patients_assigned": sum(v == s for v in assign.values()), "patients_contributing": len({r["patient_id"] for r in sel}),
                            "diameter_auc": auc([r["label"] for r in sel], [r["diameter_mm"] for r in sel]), "diameter_auc_ci": boot(sel),
                            "median_diameter": float(np.median([r["diameter_mm"] for r in sel]))}

tr = [r for r in labeled if r["split"] == "train"]
te = [r for r in labeled if r["split"] == "test"]
xtr, ytr = np.log([r["diameter_mm"] for r in tr]), [r["label"] for r in tr]
xte, yte = np.log([r["diameter_mm"] for r in te]), np.array([r["label"] for r in te])
w = fit_logistic(xtr, ytr)
pte = 1 / (1 + np.exp(-(w[0] + w[1] * xte)))
pred = pte >= 0.5
tp, tn, fp, fn = int((pred & (yte == 1)).sum()), int((~pred & (yte == 0)).sum()), int((pred & (yte == 0)).sum()), int((~pred & (yte == 1)).sum())
out["size_model_test"] = {"weights": w.tolist(), "threshold_diameter_mm": float(np.exp(-w[0] / w[1])), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                          "brier": float(np.mean((pte - yte) ** 2)), "auc": auc(yte, pte)}

all_lab = [r for r in rows if r["label"] is not None and not math.isnan(r["diameter_mm"])]
rho = spearmanr([r["diameter_mm"] for r in rows if not (math.isnan(r["malignancy_mean"]) or math.isnan(r["diameter_mm"]))],
                [r["malignancy_mean"] for r in rows if not (math.isnan(r["malignancy_mean"]) or math.isnan(r["diameter_mm"]))])
out["rating_vs_diameter_spearman_all"] = float(rho[0])

# How much does a 24-patient test partition vary by chance within this cohort?
by_p = defaultdict(list)
for r in labeled:
    by_p[r["patient_id"]].append(r)
rng = random.Random(7)
aucs, shares, ns = [], [], []
for _ in range(5000):
    a = split(cohort, rng.randrange(10**9))
    sel = [x for p, v in a.items() if v == "test" for x in by_p.get(p, [])]
    ys = [x["label"] for x in sel]
    v = auc(ys, [x["diameter_mm"] for x in sel])
    if not math.isnan(v):
        aucs.append(v)
        shares.append(sum(ys) / len(ys))
        ns.append(len(ys))
aucs, shares = np.array(aucs), np.array(shares)
np.save(Path(sys.argv[3]).with_name("random_partition_aucs.npy"), aucs)
out["random_partitions"] = {"n": len(aucs), "auc_p2_5": float(np.percentile(aucs, 2.5)), "auc_median": float(np.median(aucs)), "auc_p97_5": float(np.percentile(aucs, 97.5)),
                            "share_median": float(np.median(shares)), "share_ge_observed_pct": float(100 * np.mean(shares >= 20 / 29)),
                            "nodules_median": float(np.median(ns))}

d = np.array([r["displacement_mm"] for r in rated])
box = np.maximum(np.abs([r["dx_mm"] for r in rated]), np.abs([r["dy_mm"] for r in rated]))
rad = np.nan_to_num(np.array([r["diameter_mm"] / 2 for r in rated]))
out["defect_in_cohort"] = {"n": len(rated), "median_mm": float(np.median(d)), "fully_outside_n": int((box > 24 + rad).sum()), "fully_outside_pct": float(100 * (box > 24 + rad).mean()),
                           "labeled_fully_outside_pct": None}
lab_sel = [i for i, r in enumerate(rated) if r["label"] is not None]
out["defect_in_cohort"]["labeled_fully_outside_pct"] = float(100 * (box[lab_sel] > 24 + rad[lab_sel]).mean())

uid26 = "1.3.6.1.4.1.14519.5.2.1.6279.6001.101228986346984399347858840086"
for r in rows:
    if r["series_uid"] == uid26 and r["malignancy_mean"] == 1.5:
        out["screenshot_check"] = {"patient": r["patient_id"], "split": assign.get(r["patient_id"]), "x_vox": r["x"] - r["origin_x"], "y_vox": r["y"] - r["origin_y"], "mean_rating": r["malignancy_mean"], "screenshot_zyx": [111.9, 240.6, 75.3]}
out["test_nodules"] = sorted(((r["patient_id"], r["physical_id"], r["label"], round(r["diameter_mm"], 1)) for r in te), key=lambda t: t[3])
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "test_nodules"}, indent=1))
print("test nodules (patient, id, label, diameter):")
for t in out["test_nodules"]:
    print(" ", t)
