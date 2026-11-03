"""Fetch one DICOM header per LIDC-IDRI CT series from the public TCIA API.

Only geometry fields are kept. Pixel data are discarded.
"""
import concurrent.futures as cf
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pydicom

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
UA = {"User-Agent": "lidc-geometry-audit/1.0"}


def get(url, timeout=90, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def one(row):
    uid = row["SeriesInstanceUID"]
    sops = json.loads(get(f"{BASE}/getSOPInstanceUIDs?SeriesInstanceUID={urllib.parse.quote(uid)}"))
    sop = sops[len(sops) // 2]["SOPInstanceUID"]
    raw = get(f"{BASE}/getSingleImage?SeriesInstanceUID={urllib.parse.quote(uid)}&SOPInstanceUID={urllib.parse.quote(sop)}")
    ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
    return {
        "series_uid": uid,
        "patient_id": row["PatientID"],
        "n_images": row.get("ImageCount"),
        "file_size": row.get("FileSize"),
        "manufacturer": row.get("Manufacturer"),
        "pixel_spacing": [float(x) for x in ds.PixelSpacing],
        "image_position": [float(x) for x in ds.ImagePositionPatient],
        "image_orientation": [float(x) for x in ds.ImageOrientationPatient],
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0) or 0),
    }


def main():
    series = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    done = {}
    if out_path.exists():
        done = {r["series_uid"]: r for r in json.loads(out_path.read_text())}
    todo = [s for s in series if s["SeriesInstanceUID"] not in done]
    print(len(series), "series,", len(todo), "to fetch", flush=True)
    failed = []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, s): s for s in todo}
        for i, f in enumerate(cf.as_completed(futures), start=1):
            try:
                r = f.result()
                done[r["series_uid"]] = r
            except Exception as exc:  # noqa: BLE001
                failed.append((futures[f]["SeriesInstanceUID"], repr(exc)[:120]))
            if i % 100 == 0 or i == len(todo):
                out_path.write_text(json.dumps(list(done.values())))
                print(i, "done", len(failed), "failed", flush=True)
    out_path.write_text(json.dumps(list(done.values())))
    print("headers", len(done), "failed", len(failed), failed[:5], flush=True)


main()
