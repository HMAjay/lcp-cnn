"""Turn tolerance_preds.csv into the tolerance curve, with patient-level confidence intervals.

    python tolerance_study/analyze_tolerance.py tolerance_preds.csv --out tolerance_results

Intervals resample patients, not nodules, so nodules from one patient stay together.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

# Values in test_metrics.json and the final run's best epoch. A tied pair can resolve
# differently without mixed precision, hence the allowance of about one pair.
EXPECTED = {"test": 0.9139, "validation": 0.9493}
TOLERANCE = 0.006
DROP_OF_INTEREST = 0.05


def auc(y: np.ndarray, p: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        r["label"] = int(r["label"])
        r["shift_mm"] = int(float(r["shift_mm"]))
        r["repeat"] = int(r["repeat"])
        r["prob"] = float(r["prob"])
    return rows


def group_arrays(rows: list[dict]) -> dict:
    """Index predictions by (shift, repeat) with one fixed nodule order."""
    nodules = sorted({(r["series_uid"], r["physical_id"]) for r in rows})
    index = {k: i for i, k in enumerate(nodules)}
    y = np.zeros(len(nodules), dtype=int)
    patient = [""] * len(nodules)
    cells: dict = defaultdict(lambda: np.full(len(nodules), np.nan))
    for r in rows:
        i = index[(r["series_uid"], r["physical_id"])]
        y[i] = r["label"]
        patient[i] = r["patient_id"]
        cells[(r["shift_mm"], r["repeat"])][i] = r["prob"]
    return {"y": y, "patient": np.array(patient), "cells": dict(cells)}


def curve(y: np.ndarray, cells: dict, idx: np.ndarray, shifts: list[int]) -> np.ndarray:
    out = []
    for s in shifts:
        vals = [auc(y[idx], p[idx]) for (sh, _), p in cells.items() if sh == s]
        out.append(np.nanmean(vals))
    return np.array(out)


def analyze(rows: list[dict], n_boot: int, seed: int) -> dict:
    data = group_arrays(rows)
    y, patient, cells = data["y"], data["patient"], data["cells"]
    shifts = sorted({s for s, _ in cells})
    if 0 not in shifts:
        raise SystemExit("The file has no 0 mm rows, so there is no reference to compare against")
    point = curve(y, cells, np.arange(len(y)), shifts)
    patients = np.unique(patient)
    members = {p: np.where(patient == p)[0] for p in patients}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([members[p] for p in chosen])
        if 0 < y[idx].sum() < len(idx):
            boots.append(curve(y, cells, idx, shifts))
    boots = np.array(boots)
    base = shifts.index(0)
    drops = boots - boots[:, [base]]
    p0 = cells[(0, 0)]
    pred = (p0 >= 0.5).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    return {
        "nodules": len(y),
        "malignant": int(y.sum()),
        "benign": int(len(y) - y.sum()),
        "patients_contributing": len(patients),
        "bootstrap_samples_used": len(boots),
        "shift_mm": shifts,
        "auc": point.tolist(),
        "auc_ci_low": np.nanpercentile(boots, 2.5, axis=0).tolist(),
        "auc_ci_high": np.nanpercentile(boots, 97.5, axis=0).tolist(),
        "auc_change_from_0mm": (point - point[base]).tolist(),
        "change_ci_low": np.nanpercentile(drops, 2.5, axis=0).tolist(),
        "change_ci_high": np.nanpercentile(drops, 97.5, axis=0).tolist(),
        "at_0mm": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "brier": float(np.mean((p0 - y) ** 2))},
    }


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    styles = {"test": ("#0072B2", "o", "-"), "validation": ("#D55E00", "s", "--"), "all": ("#000000", "^", ":")}
    for name, r in results.items():
        color, marker, ls = styles.get(name, ("#000000", "o", "-"))
        x, y = np.array(r["shift_mm"]), np.array(r["auc"])
        err = [y - np.array(r["auc_ci_low"]), np.array(r["auc_ci_high"]) - y]
        ax.errorbar(x, y, yerr=err, color=color, marker=marker, linestyle=ls, capsize=2, linewidth=1.2,
                    label=f"{name} ({r['nodules']} nodules, {r['patients_contributing']} patients)")
    ax.set_xlabel("Distance of crop centre from nodule centre (mm)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.3, 1.02)
    ax.axhline(0.5, color="#808080", linewidth=0.7, linestyle=":")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions")
    parser.add_argument("--out", default="tolerance_results")
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load(Path(args.predictions))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {s: [r for r in rows if r["split"] == s] for s in sorted({r["split"] for r in rows})}
    if len(groups) > 1:
        groups["all"] = rows
    results = {name: analyze(g, args.bootstraps, args.seed) for name, g in groups.items()}

    ok = True
    for name, r in results.items():
        got = r["auc"][r["shift_mm"].index(0)]
        print(f"\n{name}: {r['nodules']} nodules ({r['malignant']} malignant, {r['benign']} benign), "
              f"{r['patients_contributing']} patients")
        if name in EXPECTED:
            good = abs(got - EXPECTED[name]) <= TOLERANCE
            ok = ok and good
            print(f"  check: AUC at 0 mm {got:.4f}, final run recorded {EXPECTED[name]:.4f} -> {'OK' if good else 'MISMATCH'}")
        print("  shift_mm    AUC   (95% CI)         change  (95% CI)")
        for i, s in enumerate(r["shift_mm"]):
            print(f"  {s:8d}  {r['auc'][i]:.3f}  ({r['auc_ci_low'][i]:.3f}-{r['auc_ci_high'][i]:.3f})  "
                  f"{r['auc_change_from_0mm'][i]:+.3f}  ({r['change_ci_low'][i]:+.3f} to {r['change_ci_high'][i]:+.3f})")
        hit = [s for s, d in zip(r["shift_mm"], r["auc_change_from_0mm"]) if d <= -DROP_OF_INTEREST]
        r["first_shift_with_drop_of_0.05"] = hit[0] if hit else None
        print(f"  first shift with an AUC drop of {DROP_OF_INTEREST} or more: {hit[0] if hit else 'none'}")
        print(f"  at 0 mm and threshold 0.50: {r['at_0mm']}")

    (out / "tolerance_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results, out / "tolerance_curve.png")
    print(f"\nWrote {out / 'tolerance_summary.json'} and {out / 'tolerance_curve.png'}")
    if not ok:
        print("MISMATCH: the data, labels or checkpoint differ from the final run. Do not use these results; find the cause first.")


if __name__ == "__main__":
    main()
