"""Turn run_output.txt into added_value.json, in the layout added_value.py writes.

Printed values carry 3 decimals (P values 4). Band diameter AUCs were not printed, so they
stay null here and come from ../lidc_audit/size_bands.json. If the full added_value.json
from the run is available, copy it over this file instead.
"""
import json
import re
from pathlib import Path

here = Path(__file__).parent
num = lambda s: None if s == "nan" else float(s)
out, cur = {}, None
for line in (here / "run_output.txt").read_text().splitlines():
    if m := re.match(r"(\w+): (\d+) nodules, (\d+) patients", line):
        cur = out[m[1]] = {"nodules": int(m[2]), "patients": int(m[3]), "size_bands": []}
    elif m := re.match(r"\s+(classifier|diameter) AUC\s+([\d.]+) \(([\d.]+)-([\d.]+)\)", line):
        cur[f"{m[1]}_auc"], cur[f"{m[1]}_auc_ci"] = float(m[2]), [float(m[3]), float(m[4])]
    elif m := re.match(r"\s+difference\s+([-+\d.]+) \(([-+\d.]+) to ([-+\d.]+)\)", line):
        cur["difference"], cur["difference_ci"] = float(m[1]), [float(m[2]), float(m[3])]
    elif m := re.match(r"\s+adds beyond size: LR ([\d.]+), P = ([\d.]+)", line):
        cur["added_value"] = {"lr_statistic": float(m[1]), "p_value": float(m[2])}
    elif m := re.match(r"\s+(\d+) to ([\d.]+|up) mm: (\d+) nodules, (\d+) malignant, classifier AUC (\S+)", line):
        cur["size_bands"].append({"from_mm": float(m[1]), "to_mm": None if m[2] == "up" else float(m[2]), "nodules": int(m[3]),
                                  "malignant": int(m[4]), "classifier_auc": num(m[5]), "diameter_auc": None})
for v in out.values():
    v["malignant"] = sum(b["malignant"] for b in v["size_bands"])
(here / "added_value.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: (v["nodules"], v["malignant"], v["added_value"]) for k, v in out.items()}))
