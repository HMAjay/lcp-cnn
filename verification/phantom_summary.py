"""Summarize phantom_results.csv for the manuscript tables and text."""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
NUM = ("pixel_spacing_mm", "slice_spacing_mm", "radius_mm", "true_i_px", "true_j_px", "n_readers", "error_mm", "error_xy_mm",
       "error_z_mm", "nodule_fraction_in_voi", "voi_padding_fraction", "voi_norm_min", "voi_norm_max", "voi_centre_hu")
for r in rows:
    for k in NUM:
        r[k] = float(r[k])
    r["captured"] = r["nodule_fraction_in_voi"] >= 0.5
    r["full"] = r["nodule_fraction_in_voi"] >= 0.999
    r["inside_nodule"] = r["error_mm"] <= r["radius_mm"]
    r["predicted_d"] = (1 - r["pixel_spacing_mm"]) * math.hypot(r["true_i_px"], r["true_j_px"])


def q(v, p):
    v = sorted(v)
    return st.quantiles(v, n=4, method="inclusive")[p]


def summ(sub):
    e = [r["error_mm"] for r in sub]
    return {
        "n": len(sub),
        "error_median": st.median(e),
        "error_q1": q(e, 0),
        "error_q3": q(e, 2),
        "error_min": min(e),
        "error_max": max(e),
        "captured_n": sum(r["captured"] for r in sub),
        "captured_pct": 100 * sum(r["captured"] for r in sub) / len(sub),
        "full_n": sum(r["full"] for r in sub),
        "centre_inside_nodule_n": sum(r["inside_nodule"] for r in sub),
        "padding_gt50_n": sum(r["voi_padding_fraction"] > 0.5 for r in sub),
        "shape_ok_n": sum(r["voi_shape_ok"] == "True" for r in sub),
        "range_ok_n": sum(0.0 <= r["voi_norm_min"] and r["voi_norm_max"] <= 1.0 for r in sub),
        "centre_hu_median": st.median(r["voi_centre_hu"] for r in sub),
        "readers_all_4": sum(r["n_readers"] == 4 for r in sub),
    }


out = {"overall": {}, "by_spacing": []}
maps = ("corrected", "defective")
for m in maps:
    out["overall"][m] = summ([r for r in rows if r["mapping"] == m])
excl = [r for r in rows if r["pixel_spacing_mm"] < 0.999]
for m in maps:
    out["overall"][m + "_below_1mm"] = summ([r for r in excl if r["mapping"] == m])
lidc_like = [r for r in rows if 0.5 <= r["pixel_spacing_mm"] <= 0.95 and r["mapping"] == "defective"]
for s in sorted({r["pixel_spacing_mm"] for r in rows}):
    row = {"spacing": s, "centre_displacement": (1 - s) * math.hypot(256, 256)}
    for m in maps:
        row[m] = summ([r for r in rows if r["mapping"] == m and r["pixel_spacing_mm"] == s])
    out["by_spacing"].append(row)

defect = [r for r in rows if r["mapping"] == "defective"]
diffs = [abs(r["error_xy_mm"] - r["predicted_d"]) for r in defect if r["predicted_d"] > 5]
out["analytic"] = {
    "n": len(diffs),
    "abs_diff_median": st.median(diffs),
    "abs_diff_max": max(diffs),
    "r": float(st.correlation([r["predicted_d"] for r in defect], [r["error_xy_mm"] for r in defect])),
    "defective_z_error_max": max(r["error_z_mm"] for r in defect),
    "corrected_z_error_max": max(r["error_z_mm"] for r in rows if r["mapping"] == "corrected"),
}
key = lambda r: (r["phantom"], r["nodule"])
at1 = {m: {key(r): r["error_mm"] for r in rows if r["mapping"] == m and r["pixel_spacing_mm"] > 0.999} for m in maps}
out["control_1mm"] = {
    "n": len(at1["corrected"]),
    "max_abs_difference_mm": max(abs(at1["corrected"][k] - at1["defective"][k]) for k in at1["corrected"]),
}
print(json.dumps(out, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
