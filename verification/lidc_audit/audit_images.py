"""Check the mappings on real LIDC-IDRI CT series against each annotated slice's own DICOM header.

For every labeled nodule the reference position comes from the slice the reader marked:
ImagePositionPatient + PixelSpacing applied to the contour pixels of that named SOP instance.
That route does not use the code under test. Series are deleted after scoring.
"""
import argparse
import csv
import importlib.util
import io
import json
import random
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pydicom

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
UA = {"User-Agent": "lidc-geometry-audit/1.0"}
VOI = (48, 48, 48)
HU = (-1000.0, 400.0)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def local(tag):
    return tag.rsplit("}", 1)[-1]


def download(uid, folder):
    url = f"{BASE}/getImage?SeriesInstanceUID={urllib.parse.quote(uid)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                raw = r.read()
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name or (info.external_attr >> 16) & 0o170000 == 0o120000:
                continue
            (folder / name).write_bytes(z.read(info))
    return len(raw)


def slice_headers(folder):
    out = {}
    for p in folder.iterdir():
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        sop = str(getattr(ds, "SOPInstanceUID", ""))
        if sop and hasattr(ds, "ImagePositionPatient"):
            out[sop] = ([float(x) for x in ds.ImagePositionPatient], [float(x) for x in ds.PixelSpacing])
    return out


def reference_positions(xml_path, slices):
    """Physical contour centre for each reader's nodule, from the named slice's own header."""
    root = ET.parse(xml_path).getroot()
    refs = {}
    for si, sess in enumerate(c for c in root if local(c.tag) == "readingSession"):
        for nod in (c for c in sess if local(c.tag) == "unblindedReadNodule"):
            nid = next((c.text.strip() for c in nod if local(c.tag) == "noduleID" and c.text), "")
            pts = []
            for roi in (c for c in nod if local(c.tag) == "roi"):
                sop, xy = None, []
                for child in roi:
                    name = local(child.tag)
                    if name == "imageSOP_UID":
                        sop = (child.text or "").strip()
                    elif name == "edgeMap":
                        vals = {local(g.tag): float(g.text) for g in child}
                        xy.append((vals["xCoord"], vals["yCoord"]))
                if sop in slices and xy:
                    ipp, ps = slices[sop]
                    a = np.mean(np.array(xy), axis=0)
                    pts.append((ipp[0] + a[0] * ps[1], ipp[1] + a[1] * ps[0], ipp[2]))
            if pts:
                refs[(si, nid)] = np.mean(np.array(pts), axis=0)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--defective", required=True)
    ap.add_argument("--nodules", required=True)
    ap.add_argument("--xml-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--series", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo)
    fixed = load_module("preprocess_fixed", repo / "src/pulmoscan/data/preprocess_lidc.py")
    broken = load_module("preprocess_broken", Path(args.defective))
    io_mod = load_module("pulmoscan_io", repo / "src/pulmoscan/data/io.py")
    rows = list(csv.DictReader(Path(args.nodules).open(encoding="utf-8")))
    xml_of = json.loads(Path(args.xml_index).read_text())

    candidates = sorted({r["series_uid"] for r in rows
                         if r["malignancy_mean"] not in ("", "None") and (float(r["malignancy_mean"]) >= 4 or float(r["malignancy_mean"]) <= 2)
                         and r["series_uid"] in xml_of})
    picked = random.Random(args.seed).sample(candidates, args.series)
    out_rows, example, done_bytes = [], None, 0
    for n, uid in enumerate(picked, start=1):
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=args.work) as tmp:
            folder = Path(tmp)
            try:
                done_bytes += download(uid, folder)
                slices = slice_headers(folder)
                volume, affine, meta, original = fixed.load_series_hu_and_affine(folder)
            except Exception as exc:  # noqa: BLE001
                print("skip", uid[-12:], repr(exc)[:100], flush=True)
                continue
            xml = Path(xml_of[uid])
            _, marks_f = fixed.parse_lidc_xml(xml, original)
            _, marks_b = broken.parse_lidc_xml(xml, affine)
            refs = reference_positions(xml, slices)
            moved = {(m.session_index, m.nodule_id): np.array(m.centroid_patient_mm) for m in marks_b}
            for c in fixed.cluster_physical_nodules(marks_f):
                score = c["malignancy_mean"]
                if score is None or 2.0 < score < 4.0:
                    continue
                keys = [(o["session_index"], o["nodule_id"]) for o in c["observations"]]
                good = np.array(c["representative_centroid_patient_mm"])
                ref = [refs[k] for k in keys if k in refs]
                bad = [moved[k] for k in keys if k in moved]
                if not ref or not bad:
                    continue
                ref = np.mean(ref, axis=0)
                bad = np.mean(bad, axis=0)
                row = {"series_uid": uid, "physical_id": c["physical_id"], "malignancy_mean": score,
                       "pixel_spacing": meta["original_spacing_xyz"][0], "slice_spacing": meta["original_spacing_xyz"][2],
                       "error_corrected_mm": float(np.linalg.norm(good - ref)), "error_corrected_xy_mm": float(np.linalg.norm((good - ref)[:2])),
                       "error_defective_mm": float(np.linalg.norm(bad - ref))}
                crops = {}
                for name, xyz in (("corrected", good), ("defective", bad)):
                    centre = io_mod.patient_to_voxel_zyx(xyz, affine)
                    voi = io_mod.crop_voi(volume, centre, VOI, fill=HU[0])
                    seen = io_mod.crop_voi(np.ones_like(volume, dtype=np.uint8), centre, VOI, fill=0.0)
                    crops[name] = (centre, voi)
                    norm = io_mod.window_normalize(voi, HU, True)
                    row[f"{name}_centre_hu"] = float(np.mean(voi[22:26, 22:26, 22:26]))
                    row[f"{name}_padding"] = float(1 - np.mean(seen))
                    row[f"{name}_shape_ok"] = voi.shape == VOI
                    row[f"{name}_range_ok"] = bool(norm.min() >= 0 and norm.max() <= 1)
                out_rows.append(row)
                big = abs(meta["original_spacing_xyz"][0] - 0.7) < 0.08 and row["error_defective_mm"] > 60 and row["corrected_padding"] == 0
                if big and row["corrected_centre_hu"] > -300 and row["defective_padding"] < 0.05 and example is None:
                    example = uid
                    cz = crops["corrected"][0]
                    np.savez_compressed(Path(args.out).with_name("real_example.npz"),
                                        axial=volume[int(round(float(cz[0])))], voi_corrected=crops["corrected"][1], voi_defective=crops["defective"][1],
                                        centre_corrected_zyx=crops["corrected"][0], centre_defective_zyx=crops["defective"][0],
                                        spacing=meta["original_spacing_xyz"][0], slice_spacing=meta["original_spacing_xyz"][2],
                                        patient=np.array(next(r["patient_id"] for r in rows if r["series_uid"] == uid)))
            del volume
        with Path(args.out).open("w", newline="", encoding="utf-8") as f:
            if out_rows:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
                w.writeheader()
                w.writerows(out_rows)
        print(f"series {n}/{args.series} {time.time() - t0:.0f}s nodules={len(out_rows)} GB={done_bytes / 1e9:.2f}", flush=True)
    print("finished", len(out_rows), "nodules", flush=True)


main()
