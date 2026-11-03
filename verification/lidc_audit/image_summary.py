"""Summarize the real-CT check. Corrected and defective positions are compared with header-derived references."""
import csv
import json
import statistics as st
import sys
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
order = list(dict.fromkeys(r["series_uid"] for r in rows))
if limit:
    order = order[:limit]
rows = [r for r in rows if r["series_uid"] in set(order)]
num = lambda k: [float(r[k]) for r in rows]
q = lambda v, i: st.quantiles(sorted(v), n=4, method="inclusive")[i]
ec, ed = num("error_corrected_mm"), num("error_defective_mm")
out = {
    "series": len(order),
    "nodules": len(rows),
    "corrected_error_median": st.median(ec),
    "corrected_error_q3": q(ec, 2),
    "corrected_error_max": max(ec),
    "corrected_xy_error_max": max(num("error_corrected_xy_mm")),
    "corrected_within_1mm": sum(e <= 1.0 for e in ec),
    "defective_error_median": st.median(ed),
    "defective_error_q1": q(ed, 0),
    "defective_error_q3": q(ed, 2),
    "defective_error_max": max(ed),
    "corrected_centre_hu_median": st.median(num("corrected_centre_hu")),
    "defective_centre_hu_median": st.median(num("defective_centre_hu")),
    "corrected_shape_ok": sum(r["corrected_shape_ok"] == "True" for r in rows),
    "defective_shape_ok": sum(r["defective_shape_ok"] == "True" for r in rows),
    "defective_range_ok": sum(r["defective_range_ok"] == "True" for r in rows),
    "defective_padding_gt50": sum(float(r["defective_padding"]) > 0.5 for r in rows),
    "corrected_padding_gt50": sum(float(r["corrected_padding"]) > 0.5 for r in rows),
}
print(json.dumps(out, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
