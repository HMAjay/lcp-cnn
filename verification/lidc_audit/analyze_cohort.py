"""Collection-wide statistics, the size-only baseline, and reconstruction of the project split."""
import argparse
import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

TARGET = {"train": (45, 87), "validation": (12, 23), "test": (20, 9)}


def auc(y, p):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def label(score):
    if score is None or math.isnan(score):
        return None
    return 1 if score >= 4.0 else 0 if score <= 2.0 else None


def load(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    for r in rows:
        for k in ("n_readers", "n_distinct_readers"):
            r[k] = int(r[k])
        for k in ("malignancy_mean", "diameter_mm", "volume_mm3", "pixel_spacing", "x", "y", "z", "dx_mm", "dy_mm", "dz_mm",
                  "displacement_mm", "fov_x_mm", "fov_y_mm", "origin_x", "origin_y"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
        r["label"] = label(r["malignancy_mean"])
    return rows


def patient_boot(y, p, patients, n_boot, seed):
    y, p, patients = np.asarray(y), np.asarray(p), np.asarray(patients)
    ids = np.unique(patients)
    members = {i: np.where(patients == i)[0] for i in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.choice(ids, size=len(ids), replace=True)])
        v = auc(y[idx], p[idx])
        if not math.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def split_patients(patients, seed=42):
    """Same procedure as scripts/build_manifests_from_processed.py."""
    plist = sorted(patients)
    random.Random(seed).shuffle(plist)
    n = len(plist)
    n_train, n_val = max(1, int(n * 0.7)), max(1, int(n * 0.15))
    return {p: ("train" if i < n_train else "validation" if i < n_train + n_val else "test") for i, p in enumerate(plist)}


def choose_series(uids, header_by_uid):
    best = {}
    for u in uids:
        h = header_by_uid.get(u)
        if h is None:
            continue
        p = h["patient_id"]
        if p not in best or (h["n_images"] or 0) > (header_by_uid[best[p]]["n_images"] or 0):
            best[p] = u
    return best


def split_counts(chosen, rows_by_series):
    assign = split_patients(chosen.keys())
    counts = defaultdict(lambda: [0, 0])
    for p, u in chosen.items():
        for r in rows_by_series.get(u, []):
            if r["label"] is not None:
                counts[assign[p]][0 if r["label"] == 1 else 1] += 1
    return {k: tuple(v) for k, v in counts.items()}, assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodules", required=True)
    ap.add_argument("--headers", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--series-all", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.nodules)
    headers = {h["series_uid"]: h for h in json.loads(Path(args.headers).read_text())}
    by_series = defaultdict(list)
    for r in rows:
        by_series[r["series_uid"]].append(r)

    out = {}
    spacing = [h["pixel_spacing"][1] for h in headers.values()]
    out["series"] = {
        "ct_series": len(headers),
        "patients": len({h["patient_id"] for h in headers.values()}),
        "spacing_min": min(spacing), "spacing_max": max(spacing),
        "spacing_mean": float(np.mean(spacing)), "spacing_median": float(np.median(spacing)),
        "non_square_pixels": sum(abs(h["pixel_spacing"][0] - h["pixel_spacing"][1]) > 1e-6 for h in headers.values()),
        "not_identity_orientation": sum(any(abs(a - b) > 1e-3 for a, b in zip(h["image_orientation"], [1, 0, 0, 0, 1, 0])) for h in headers.values()),
        "matrix_not_512": sum(h["rows"] != 512 or h["columns"] != 512 for h in headers.values()),
        "image_bytes_total": sum(int(h["file_size"] or 0) for h in headers.values()),
    }

    rated = [r for r in rows if not math.isnan(r["malignancy_mean"])]
    labeled = [r for r in rated if r["label"] is not None]
    m = [r for r in labeled if r["label"] == 1]
    out["nodules"] = {
        "physical_rated": len(rated), "labeled": len(labeled), "malignant": len(m), "benign": len(labeled) - len(m),
        "indeterminate_excluded": len(rated) - len(labeled),
        "labeled_patients": len({r["patient_id"] for r in labeled}),
        "labeled_single_reader": sum(r["n_distinct_readers"] == 1 for r in labeled),
    }

    d = np.array([r["displacement_mm"] for r in rated])
    dx = np.array([abs(r["dx_mm"]) for r in rated])
    dy = np.array([abs(r["dy_mm"]) for r in rated])
    box = np.maximum(dx, dy)
    radius = np.array([r["diameter_mm"] / 2 for r in rated])
    radius = np.where(np.isnan(radius), 0, radius)
    cx = np.array([(r["x"] + r["dx_mm"] - r["origin_x"]) for r in rated])
    cy = np.array([(r["y"] + r["dy_mm"] - r["origin_y"]) for r in rated])
    fovx = np.array([r["fov_x_mm"] for r in rated])
    fovy = np.array([r["fov_y_mm"] for r in rated])
    outside = (cx > fovx) | (cy > fovy)
    half = (cx - 24 > fovx) | (cy - 24 > fovy)
    out["displacement_all_rated"] = {
        "n": len(rated),
        "median": float(np.median(d)), "q1": float(np.percentile(d, 25)), "q3": float(np.percentile(d, 75)),
        "min": float(d.min()), "max": float(d.max()),
        "centre_outside_voi_n": int((box > 24).sum()), "centre_outside_voi_pct": float(100 * (box > 24).mean()),
        "nodule_fully_outside_voi_n": int((box > 24 + radius).sum()), "nodule_fully_outside_voi_pct": float(100 * (box > 24 + radius).mean()),
        "nodule_at_least_partly_in_voi_n": int((box <= 24 + radius).sum()),
        "mapped_centre_outside_image_n": int(outside.sum()), "mapped_centre_outside_image_pct": float(100 * outside.mean()),
        "voi_fully_outside_image_n": int(half.sum()),
        "spacing_of_cases_still_in_voi_median": float(np.median([r["pixel_spacing"] for r, b, rd in zip(rated, box, radius) if b <= 24 + rd] or [float("nan")])),
    }

    y = [r["label"] for r in labeled]
    diam = [r["diameter_mm"] for r in labeled]
    vol = [r["volume_mm3"] for r in labeled]
    pts = [r["patient_id"] for r in labeled]
    keep = [i for i, v in enumerate(diam) if not math.isnan(v)]
    yk, dk, vk, pk = [y[i] for i in keep], [diam[i] for i in keep], [vol[i] for i in keep], [pts[i] for i in keep]
    out["size_baseline_full"] = {
        "n": len(yk), "malignant": sum(yk), "patients": len(set(pk)),
        "diameter_auc": auc(yk, dk), "diameter_ci": patient_boot(yk, dk, pk, 2000, 42),
        "volume_auc": auc(yk, vk), "volume_ci": patient_boot(yk, vk, pk, 2000, 43),
        "median_diameter_malignant": float(np.median([d for d, l in zip(dk, yk) if l == 1])),
        "median_diameter_benign": float(np.median([d for d, l in zip(dk, yk) if l == 0])),
    }

    lines = [l.strip() for l in Path(args.manifest).read_text().splitlines()]
    start = lines.index("ListOfSeriesToDownload=") + 1
    order = [l for l in lines[start:] if l]
    allseries = {s["SeriesInstanceUID"]: s for s in json.loads(Path(args.series_all).read_text())}
    annotated = set(by_series)
    print("manifest series", len(order), "known", sum(u in allseries for u in order))
    found = []
    prefix_stats = []
    for n in range(100, len(order) + 1):
        uids = [u for u in order[:n] if u in annotated]
        chosen = choose_series(uids, headers)
        if len(chosen) == 153:
            counts, assign = split_counts(chosen, by_series)
            total = sum(a + b for a, b in counts.values())
            prefix_stats.append((n, counts, total))
            if all(counts.get(k) == v for k, v in TARGET.items()):
                found.append((n, chosen, assign))
    print("manifest prefixes with 153 patients:", [(n, t) for n, _, t in prefix_stats][:30])
    for n, c, t in prefix_stats[:6]:
        print("  n", n, c)
    sorted_ids = sorted({h["patient_id"] for u, h in headers.items() if u in annotated})
    chosen = choose_series([u for u in annotated if headers[u]["patient_id"] in set(sorted_ids[:153])], headers)
    counts, _ = split_counts(chosen, by_series)
    print("first 153 by patient ID:", counts, "| contains LIDC-IDRI-0026:", "LIDC-IDRI-0026" in chosen)
    out["reconstruction"] = {"exact_match": bool(found), "prefix_candidates": len(prefix_stats), "first_153_by_id_counts": counts}
    if found:
        n, chosen, assign = found[0]
        out["reconstruction"].update(prefix=n, patients=sorted(chosen), lidc0026=assign.get("LIDC-IDRI-0026"))
        for split in ("test", "validation", "train"):
            sel = [r for p, u in chosen.items() if assign[p] == split for r in by_series.get(u, []) if r["label"] is not None and not math.isnan(r["diameter_mm"])]
            ys, ds, ps = [r["label"] for r in sel], [r["diameter_mm"] for r in sel], [r["patient_id"] for r in sel]
            out["reconstruction"][split] = {"n": len(sel), "patients": len(set(ps)), "diameter_auc": auc(ys, ds), "diameter_ci": patient_boot(ys, ds, ps, 2000, 44)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "reconstruction"}, indent=1))
    print("exact split match:", bool(found))


main()
