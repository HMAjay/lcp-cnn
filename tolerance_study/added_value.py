"""Test whether the classifier adds information beyond nodule diameter.

Uses the 0 mm rows from shift_eval.py and the diameters regenerated from the public files:

    python tolerance_study/added_value.py tolerance_preds.csv \
        --nodules tolerance_study/nodules.csv --out added_value.json

Intervals resample patients. The likelihood-ratio test asks whether the classifier score
improves a logistic model that already contains log diameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, rankdata

BANDS = [(0.0, 10.0), (10.0, 20.0), (20.0, 1e9)]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def nll(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    z = X @ w
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def fit(X: np.ndarray, y: np.ndarray) -> float:
    return minimize(nll, np.zeros(X.shape[1]), args=(X, y), method="BFGS").fun


def added_value_test(prob: np.ndarray, diameter: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    size = np.log(diameter)
    score = np.log(p / (1 - p))
    size = (size - size.mean()) / size.std()
    score = (score - score.mean()) / (score.std() or 1.0)
    one = np.ones_like(size)
    base = fit(np.c_[one, size], y)
    full = fit(np.c_[one, size, score], y)
    stat = max(0.0, 2 * (base - full))
    return {"lr_statistic": stat, "p_value": float(chi2.sf(stat, 1))}


def summarize(sel: list[dict], n_boot: int, seed: int) -> dict:
    y = np.array([r["label"] for r in sel])
    prob = np.array([r["prob"] for r in sel])
    diam = np.array([r["diameter"] for r in sel])
    patients = np.array([r["patient_id"] for r in sel])
    ids = np.unique(patients)
    members = {i: np.where(patients == i)[0] for i in ids}
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.choice(ids, size=len(ids), replace=True)])
        u, v = auc(y[idx], prob[idx]), auc(y[idx], diam[idx])
        if not (math.isnan(u) or math.isnan(v)):
            a.append(u)
            b.append(v)
    a, b = np.array(a), np.array(b)
    out = {
        "nodules": len(y), "malignant": int(y.sum()), "patients": len(ids),
        "classifier_auc": auc(y, prob), "classifier_auc_ci": np.percentile(a, [2.5, 97.5]).tolist(),
        "diameter_auc": auc(y, diam), "diameter_auc_ci": np.percentile(b, [2.5, 97.5]).tolist(),
        "difference": auc(y, prob) - auc(y, diam), "difference_ci": np.percentile(a - b, [2.5, 97.5]).tolist(),
        "added_value": added_value_test(prob, diam, y),
        "size_bands": [],
    }
    for lo, hi in BANDS:
        m = (diam >= lo) & (diam < hi)
        out["size_bands"].append({"from_mm": lo, "to_mm": None if hi > 1e8 else hi, "nodules": int(m.sum()), "malignant": int(y[m].sum()),
                                  "classifier_auc": auc(y[m], prob[m]), "diameter_auc": auc(y[m], diam[m])})
    return out


def merge(predictions: str, nodules: str) -> list[dict]:
    sizes = {(r["series_uid"], r["physical_id"]): r for r in csv.DictReader(open(nodules, encoding="utf-8"))}
    rows, missing = [], 0
    for r in csv.DictReader(open(predictions, encoding="utf-8")):
        if float(r["shift_mm"]) != 0:
            continue
        m = sizes.get((r["series_uid"], r["physical_id"]))
        if m is None or m["diameter_mm"] in ("", "nan"):
            missing += 1
            continue
        rows.append({"split": r["split"], "patient_id": r["patient_id"], "label": int(r["label"]), "prob": float(r["prob"]), "diameter": float(m["diameter_mm"])})
    if missing:
        print(f"WARNING: {missing} nodules had no diameter record and were left out")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions")
    ap.add_argument("--nodules", default="tolerance_study/nodules.csv")
    ap.add_argument("--out", default="added_value.json")
    ap.add_argument("--bootstraps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = merge(args.predictions, args.nodules)
    groups = defaultdict(list)
    for r in rows:
        groups[r["split"]].append(r)
    if len(groups) > 1:
        groups["all"] = rows
    results = {k: summarize(v, args.bootstraps, args.seed) for k, v in groups.items()}
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for k, r in results.items():
        print(f"\n{k}: {r['nodules']} nodules, {r['patients']} patients")
        print(f"  classifier AUC {r['classifier_auc']:.3f} ({r['classifier_auc_ci'][0]:.3f}-{r['classifier_auc_ci'][1]:.3f})")
        print(f"  diameter AUC   {r['diameter_auc']:.3f} ({r['diameter_auc_ci'][0]:.3f}-{r['diameter_auc_ci'][1]:.3f})")
        print(f"  difference     {r['difference']:+.3f} ({r['difference_ci'][0]:+.3f} to {r['difference_ci'][1]:+.3f})")
        print(f"  adds beyond size: LR {r['added_value']['lr_statistic']:.2f}, P = {r['added_value']['p_value']:.4f}")
        for b in r["size_bands"]:
            print(f"  {b['from_mm']:.0f} to {b['to_mm'] or 'up'} mm: {b['nodules']} nodules, {b['malignant']} malignant, classifier AUC {b['classifier_auc']:.3f}")


if __name__ == "__main__":
    main()
