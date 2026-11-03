"""Run the repository's corrected and defective mappings over all real LIDC-IDRI annotations.

The corrected parser needs only each series' geometry, so a small object stands in for the
SimpleITK image. Pixel data are not needed. Outputs one row per physical nodule.
"""
import argparse
import csv
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def local(tag):
    return tag.rsplit("}", 1)[-1]


class Geometry:
    """Answers the two SimpleITK calls that parse_lidc_xml makes."""

    def __init__(self, header, z_values):
        self.dy, self.dx = header["pixel_spacing"]
        self.origin = np.array(header["image_position"], dtype=float)
        iop = header["image_orientation"]
        self.row = np.array(iop[:3], dtype=float)
        self.col = np.array(iop[3:6], dtype=float)
        self.z = sorted(z_values)
        self.size = (int(header["columns"]), int(header["rows"]), max(len(self.z), 1))

    def GetSize(self):
        return self.size

    def TransformContinuousIndexToPhysicalPoint(self, idx):
        i, j, k = idx
        p = self.origin + i * self.dx * self.row + j * self.dy * self.col
        z = self.z[int(k)] if self.z else float(self.origin[2])
        return (float(p[0]), float(p[1]), float(z))

    def unit_affine(self):
        """Affine of the 1 mm resampled image, which the defective code applied to pixel indices."""
        normal = np.cross(self.row, self.col)
        a = np.eye(4)
        a[:3, 0], a[:3, 1], a[:3, 2] = self.row, self.col, normal
        a[:3, 3] = [self.origin[0], self.origin[1], self.z[0] if self.z else self.origin[2]]
        return a


def contour_sizes(path, dx, dy):
    """Largest in-plane diameter and contour-based volume for each reader's nodule."""
    root = ET.parse(path).getroot()
    z_values, sizes = set(), {}
    for si, sess in enumerate(c for c in root if local(c.tag) == "readingSession"):
        for nod in (c for c in sess if local(c.tag) == "unblindedReadNodule"):
            nid = next((c.text.strip() for c in nod if local(c.tag) == "noduleID" and c.text), "")
            areas, diameters, zs = [], [], []
            for roi in (c for c in nod if local(c.tag) == "roi"):
                pts, z, include = [], None, True
                for child in roi:
                    name = local(child.tag)
                    if name == "imageZposition":
                        z = float(child.text)
                    elif name == "inclusion":
                        include = (child.text or "").strip().upper() == "TRUE"
                    elif name == "edgeMap":
                        vals = {local(g.tag): float(g.text) for g in child}
                        pts.append((vals["xCoord"] * dx, vals["yCoord"] * dy))
                if z is None:
                    continue
                z_values.add(z)
                if not include or len(pts) < 2:
                    continue
                a = np.array(pts)
                x, y = a[:, 0], a[:, 1]
                areas.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
                diameters.append(float(pdist(a).max()))
                zs.append(z)
            if diameters:
                uz = sorted(set(zs))
                step = float(np.median(np.diff(uz))) if len(uz) > 1 else 0.0
                sizes[(si, nid)] = {"diameter": max(diameters), "area_sum": float(sum(areas)), "step": step, "z": zs}
    return z_values, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--defective", required=True)
    ap.add_argument("--headers", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fixed = load_module("preprocess_fixed", Path(args.repo) / "src/pulmoscan/data/preprocess_lidc.py")
    broken = load_module("preprocess_broken", Path(args.defective))
    headers = {h["series_uid"]: h for h in json.loads(Path(args.headers).read_text())}

    by_series = {}
    pattern = re.compile(r"<SeriesInstanceU[iI][dD]>\s*([^<\s]+)\s*</SeriesInstanceU[iI][dD]>")
    for path in sorted(Path(args.xml).glob("*.xml")):
        m = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        if m and m.group(1) in headers:
            by_series.setdefault(m.group(1), []).append(path)
    multi = {k: v for k, v in by_series.items() if len(v) > 1}

    rows, skipped = [], []
    for uid, paths in by_series.items():
        h = headers[uid]
        path = paths[0]
        dy, dx = h["pixel_spacing"]
        z_values, sizes = contour_sizes(path, dx, dy)
        geo = Geometry(h, z_values)
        try:
            _, marks_f = fixed.parse_lidc_xml(path, geo)
            _, marks_b = broken.parse_lidc_xml(path, geo.unit_affine())
        except Exception as exc:  # noqa: BLE001
            skipped.append((uid, repr(exc)[:100]))
            continue
        moved = {(m.session_index, m.nodule_id): np.array(m.centroid_patient_mm) for m in marks_b}
        for c in fixed.cluster_physical_nodules(marks_f):
            obs = c["observations"]
            keys = [(o["session_index"], o["nodule_id"]) for o in obs]
            true_xyz = {k: np.array(o["centroid_patient_mm"]) for k, o in zip(keys, obs)}
            deltas = [moved[k] - true_xyz[k] for k in keys if k in moved]
            delta = np.mean(deltas, axis=0) if deltas else np.zeros(3)
            sz = [sizes[k] for k in keys if k in sizes]
            diam = float(np.mean([s["diameter"] for s in sz])) if sz else float("nan")
            vol = float(np.mean([s["area_sum"] * (s["step"] or h["slice_thickness"]) for s in sz])) if sz else float("nan")
            c_xyz = c["representative_centroid_patient_mm"]
            rows.append(
                {
                    "patient_id": h["patient_id"],
                    "series_uid": uid,
                    "physical_id": c["physical_id"],
                    "n_readers": c["num_observations"],
                    "n_distinct_readers": len({k[0] for k in keys}),
                    "confidence": c["confidence"],
                    "malignancy_mean": c["malignancy_mean"],
                    "diameter_mm": diam,
                    "volume_mm3": vol,
                    "pixel_spacing": dx,
                    "x": c_xyz[0], "y": c_xyz[1], "z": c_xyz[2],
                    "dx_mm": float(delta[0]), "dy_mm": float(delta[1]), "dz_mm": float(delta[2]),
                    "displacement_mm": float(np.linalg.norm(delta)),
                    "fov_x_mm": h["columns"] * dx,
                    "fov_y_mm": h["rows"] * dy,
                    "origin_x": h["image_position"][0],
                    "origin_y": h["image_position"][1],
                }
            )
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("xml matched to CT series:", len(by_series), "| series with two XML files:", len(multi),
          "| skipped:", len(skipped), skipped[:3], "| physical nodules:", len(rows))


main()
