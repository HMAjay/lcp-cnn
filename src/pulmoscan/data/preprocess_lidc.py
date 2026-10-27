"""CPU preprocessing: TCIA LIDC-IDRI DICOM + XML → PulmoScan processed/ tree.

No GPU required. Run on your laptop/WSL after a TCIA batch download.

Output layout (matches docs/PREPROCESSING_README.md):
  processed/
    volumes/<SeriesInstanceUID>.nii.gz          # 1 mm isotropic HU
    metadata/lidc/<SeriesInstanceUID>.json      # parsed XML
    metadata/physical_nodules/<SeriesUID>.json  # clustered nodules
    manifests/dataset_manifest.json
    manifests/patient_splits.json
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


@dataclass
class SeriesInfo:
    series_uid: str
    patient_id: str
    dicom_dir: Path
    study_uid: str = ""
    modality: str = "CT"
    n_files: int = 0


@dataclass
class ReaderNodule:
    session_index: int
    nodule_id: str
    malignancy: float | None
    centroid_patient_mm: list[float]
    n_rois: int = 0
    characteristics: dict[str, Any] = field(default_factory=dict)


def _local(tag: str) -> str:
    """Strip XML namespace if present."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _text(el: ET.Element | None, default: str = "") -> str:
    if el is None or el.text is None:
        return default
    return str(el.text).strip()


def _find_child(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent:
        if _local(child.tag) == name:
            return child
    return None


def _findall(parent: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in parent if _local(c.tag) == name]


def discover_ct_series(raw_root: str | Path) -> list[SeriesInfo]:
    """Find CT DICOM series under a TCIA download folder."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pydicom is required for preprocessing. Install with: pip install -e \".[preprocess]\""
        ) from exc

    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    # Group DICOM files by SeriesInstanceUID
    by_series: dict[str, list[Path]] = defaultdict(list)
    meta: dict[str, dict[str, str]] = {}

    dcm_paths = list(raw_root.rglob("*.dcm"))
    # Some TCIA dumps omit .dcm extension
    if not dcm_paths:
        candidates = [p for p in raw_root.rglob("*") if p.is_file() and not p.name.startswith(".")]
        for path in candidates:
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                if getattr(ds, "Modality", None) is None and not hasattr(ds, "SeriesInstanceUID"):
                    continue
                dcm_paths.append(path)
            except Exception:
                continue

    for path in tqdm(dcm_paths, desc="index DICOM", leave=False):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        modality = str(getattr(ds, "Modality", "") or "").upper()
        # Keep CT only (skip DX/CR chest X-rays). Empty modality allowed (some dumps omit it).
        if modality in {"DX", "CR"}:
            continue
        if modality and modality != "CT":
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not series_uid:
            continue
        by_series[series_uid].append(path)
        if series_uid not in meta:
            pid = str(getattr(ds, "PatientID", "") or "")
            # Prefer LIDC-IDRI-XXXX from path
            m = re.search(r"LIDC-IDRI-\d+", str(path))
            if m:
                pid = m.group(0)
            elif not pid:
                pid = series_uid
            meta[series_uid] = {
                "patient_id": pid,
                "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
                "modality": modality or "CT",
                "dir": str(path.parent),
            }

    out: list[SeriesInfo] = []
    for series_uid, files in by_series.items():
        info = meta.get(series_uid, {})
        # Use common parent directory of files
        dirs = {f.parent for f in files}
        dicom_dir = min(dirs, key=lambda p: len(str(p))) if dirs else Path(info.get("dir", "."))
        out.append(
            SeriesInfo(
                series_uid=series_uid,
                patient_id=info.get("patient_id", series_uid),
                dicom_dir=dicom_dir,
                study_uid=info.get("study_uid", ""),
                modality=info.get("modality", "CT"),
                n_files=len(files),
            )
        )
    out.sort(key=lambda s: (s.patient_id, s.series_uid))
    return out


def load_series_hu_and_affine(dicom_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a DICOM CT series as (Z,Y,X) float32 HU + 4x4 patient affine.

    Affine maps voxel (x,y,z) index → patient mm (nibabel convention).
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SimpleITK is required for preprocessing. Install with: pip install -e \".[preprocess]\""
        ) from exc

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        # Fallback: all files in folder
        names = sorted(str(p) for p in dicom_dir.iterdir() if p.is_file())
    if not names:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
    reader.SetFileNames(names)
    image = reader.Execute()

    # Resample to 1×1×1 mm isotropic in physical space
    original_spacing = np.array(image.GetSpacing(), dtype=np.float64)  # (x,y,z)
    original_size = np.array(image.GetSize(), dtype=np.int64)
    new_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    new_size = np.maximum(
        1, np.round(original_size * (original_spacing / new_spacing)).astype(np.int64)
    )

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(tuple(new_spacing.tolist()))
    resample.SetSize([int(x) for x in new_size.tolist()])
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetDefaultPixelValue(-1024.0)
    image_1mm = resample.Execute(image)

    arr = sitk.GetArrayFromImage(image_1mm).astype(np.float32)  # (Z,Y,X)
    # Build nibabel-style affine: index (i,j,k)=(x,y,z) → patient
    spacing = np.array(image_1mm.GetSpacing(), dtype=np.float64)
    origin = np.array(image_1mm.GetOrigin(), dtype=np.float64)
    direction = np.array(image_1mm.GetDirection(), dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin

    meta = {
        "original_spacing_xyz": original_spacing.tolist(),
        "spacing_xyz": spacing.tolist(),
        "origin_xyz": origin.tolist(),
        "size_zyx": list(arr.shape),
        "n_dicom_files": len(names),
    }
    return arr, affine, meta


def save_nifti(path: Path, volume_zyx: np.ndarray, affine: np.ndarray) -> None:
    import nibabel as nib

    # Store as (X,Y,Z) for nibabel; loader transposes back to (Z,Y,X)
    data = np.transpose(volume_zyx.astype(np.float32), (2, 1, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def index_xml_files(xml_root: str | Path | None) -> dict[str, Path]:
    """Map SeriesInstanceUID → XML path."""
    if xml_root is None:
        return {}
    root = Path(xml_root)
    if not root.exists():
        return {}
    index: dict[str, Path] = {}
    for path in root.rglob("*.xml"):
        try:
            # Read a prefix for SeriesInstanceUid
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = re.search(
            r"<SeriesInstanceU[iI][dD]>\s*([^<\s]+)\s*</SeriesInstanceU[iI][dD]>",
            text,
        )
        if not m:
            m = re.search(r'SeriesInstanceUID["\s:=\>]+([0-9.]+)', text)
        if m:
            index[m.group(1).strip()] = path
    return index


def _roi_centroid_patient_mm(roi: ET.Element, affine: np.ndarray) -> list[float] | None:
    """Centroid of one ROI contour in patient mm using NIfTI affine."""
    xs: list[float] = []
    ys: list[float] = []
    z_pos = None
    for child in roi:
        name = _local(child.tag)
        if name == "imageZposition":
            try:
                z_pos = float(_text(child))
            except ValueError:
                z_pos = None
        elif name == "edgeMap":
            x_el = _find_child(child, "xCoord")
            y_el = _find_child(child, "yCoord")
            try:
                xs.append(float(_text(x_el)))
                ys.append(float(_text(y_el)))
            except ValueError:
                continue
    if not xs or not ys:
        return None
    # Pixel indices (i,j) = (x,y); z from imageZposition is already patient Z
    # Prefer mapping x,y through affine with approximate z index from physical Z.
    # affine maps (i,j,k) → (X,Y,Z). For LIDC, imageZposition is patient Z.
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    if z_pos is None:
        return None
    # Invert affine to get k from patient Z while fixing i,j — approximate:
    # patient ≈ origin + A @ [i,j,k]; use direction to solve k from Z.
    A = affine[:3, :3]
    origin = affine[:3, 3]
    # Approximate: assume nearly axial; k ≈ (z_pos - origin_z) / A[2,2]
    if abs(A[2, 2]) > 1e-6:
        k = (z_pos - origin[2]) / A[2, 2]
    else:
        k = 0.0
    ijk = np.array([cx, cy, k, 1.0], dtype=np.float64)
    xyz = affine @ ijk
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def parse_lidc_xml(
    xml_path: Path,
    affine: np.ndarray,
) -> tuple[dict[str, Any], list[ReaderNodule]]:
    """Parse one LIDC XML into JSON-serializable doc + reader nodules (≥3mm with malignancy)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    header: dict[str, Any] = {}
    hdr = None
    for child in root:
        if _local(child.tag) in {"ResponseHeader", "responseHeader"}:
            hdr = child
            break
    if hdr is not None:
        for child in hdr:
            header[_local(child.tag)] = _text(child)

    series_uid = (
        header.get("SeriesInstanceUid")
        or header.get("SeriesInstanceUID")
        or header.get("seriesInstanceUid")
        or ""
    )

    sessions_out: list[dict[str, Any]] = []
    nodules: list[ReaderNodule] = []

    session_els = [c for c in root if _local(c.tag) == "readingSession"]
    for si, sess in enumerate(session_els):
        sess_nodules = []
        for nod in _findall(sess, "unblindedReadNodule"):
            nid = _text(_find_child(nod, "noduleID"))
            chars_el = _find_child(nod, "characteristics")
            chars: dict[str, Any] = {}
            malignancy = None
            if chars_el is not None:
                for ch in chars_el:
                    key = _local(ch.tag)
                    val = _text(ch)
                    if val == "":
                        continue
                    try:
                        chars[key] = float(val) if key != "internalStructure" else val
                    except ValueError:
                        chars[key] = val
                if "malignancy" in chars:
                    try:
                        malignancy = float(chars["malignancy"])
                    except (TypeError, ValueError):
                        malignancy = None

            rois = _findall(nod, "roi")
            cents = []
            for roi in rois:
                c = _roi_centroid_patient_mm(roi, affine)
                if c is not None:
                    cents.append(c)
            if not cents:
                continue
            centroid = [
                float(np.mean([c[0] for c in cents])),
                float(np.mean([c[1] for c in cents])),
                float(np.mean([c[2] for c in cents])),
            ]
            # Skip <3mm marks without characteristics (no malignancy score)
            if malignancy is None and not chars:
                continue
            entry = {
                "nodule_id": nid,
                "characteristics": chars,
                "centroid_patient_mm": centroid,
                "n_rois": len(rois),
            }
            sess_nodules.append(entry)
            nodules.append(
                ReaderNodule(
                    session_index=si,
                    nodule_id=nid,
                    malignancy=malignancy,
                    centroid_patient_mm=centroid,
                    n_rois=len(rois),
                    characteristics=chars,
                )
            )
        sessions_out.append({"session_index": si, "nodules": sess_nodules})

    doc = {
        "series_instance_uid": series_uid,
        "response_header": header,
        "reading_sessions": sessions_out,
        "source_xml": str(xml_path.name),
    }
    return doc, nodules


def cluster_physical_nodules(
    reader_nodules: list[ReaderNodule],
    *,
    distance_mm: float = 6.0,
) -> list[dict[str, Any]]:
    """Greedy spatial clustering of reader marks into physical nodules."""
    remaining = list(enumerate(reader_nodules))
    clusters: list[list[ReaderNodule]] = []

    while remaining:
        _, seed = remaining.pop(0)
        cluster = [seed]
        seed_xyz = np.array(seed.centroid_patient_mm, dtype=np.float64)
        kept: list[tuple[int, ReaderNodule]] = []
        for item in remaining:
            xyz = np.array(item[1].centroid_patient_mm, dtype=np.float64)
            if float(np.linalg.norm(xyz - seed_xyz)) <= distance_mm:
                cluster.append(item[1])
            else:
                kept.append(item)
        remaining = kept
        # Update seed as running mean and try one more absorb pass
        mean = np.mean([n.centroid_patient_mm for n in cluster], axis=0)
        kept2: list[tuple[int, ReaderNodule]] = []
        for item in remaining:
            xyz = np.array(item[1].centroid_patient_mm, dtype=np.float64)
            if float(np.linalg.norm(xyz - mean)) <= distance_mm:
                cluster.append(item[1])
            else:
                kept2.append(item)
        remaining = kept2
        clusters.append(cluster)

    physical: list[dict[str, Any]] = []
    for i, cluster in enumerate(clusters):
        scores = [n.malignancy for n in cluster if n.malignancy is not None]
        mean_xyz = np.mean([n.centroid_patient_mm for n in cluster], axis=0)
        n_readers = len({n.session_index for n in cluster})
        if n_readers >= 3:
            confidence = "high"
        elif n_readers == 2:
            confidence = "medium"
        else:
            confidence = "low"
        physical.append(
            {
                "physical_id": f"phys-{i}",
                "confidence": confidence,
                "num_observations": len(cluster),
                "representative_centroid_patient_mm": [
                    float(mean_xyz[0]),
                    float(mean_xyz[1]),
                    float(mean_xyz[2]),
                ],
                "malignancy_mean": float(sum(scores) / len(scores)) if scores else None,
                "observations": [
                    {
                        "session_index": n.session_index,
                        "nodule_id": n.nodule_id,
                        "characteristics": n.characteristics,
                        "centroid_patient_mm": n.centroid_patient_mm,
                        "malignancy": n.malignancy,
                    }
                    for n in cluster
                ],
            }
        )
    return physical


def process_series(
    series: SeriesInfo,
    *,
    out_root: Path,
    xml_index: dict[str, Path],
    xml_fallback_roots: list[Path],
    skip_existing: bool = True,
) -> dict[str, Any]:
    vol_path = out_root / "volumes" / f"{series.series_uid}.nii.gz"
    phys_path = out_root / "metadata" / "physical_nodules" / f"{series.series_uid}.json"
    lidc_path = out_root / "metadata" / "lidc" / f"{series.series_uid}.json"

    if skip_existing and vol_path.exists() and phys_path.exists():
        return {"series_uid": series.series_uid, "status": "skipped_existing"}

    volume, affine, meta = load_series_hu_and_affine(series.dicom_dir)
    save_nifti(vol_path, volume, affine)

    xml_path = xml_index.get(series.series_uid)
    if xml_path is None:
        # Search near DICOM tree and fallback roots
        search_roots = [series.dicom_dir, *series.dicom_dir.parents[:4], *xml_fallback_roots]
        for root in search_roots:
            if root is None or not Path(root).exists():
                continue
            for cand in Path(root).rglob("*.xml"):
                try:
                    text = cand.read_text(encoding="utf-8", errors="ignore")[:8000]
                except Exception:
                    continue
                if series.series_uid in text:
                    xml_path = cand
                    break
            if xml_path is not None:
                break

    reader_nodules: list[ReaderNodule] = []
    lidc_doc: dict[str, Any] = {
        "series_instance_uid": series.series_uid,
        "reading_sessions": [],
        "note": "no_xml_found",
    }
    if xml_path is not None and xml_path.exists():
        lidc_doc, reader_nodules = parse_lidc_xml(xml_path, affine)
        lidc_doc["series_instance_uid"] = series.series_uid

    lidc_path.parent.mkdir(parents=True, exist_ok=True)
    lidc_path.write_text(json.dumps(lidc_doc, indent=2), encoding="utf-8")

    physical = cluster_physical_nodules(reader_nodules)
    phys_doc = {
        "series_uid": series.series_uid,
        "series_instance_uid": series.series_uid,
        "patient_id": series.patient_id,
        "physical_nodules": physical,
        "ambiguous_observations": [],
        "unmatched_observations": [],
        "volume_meta": meta,
        "lidc_metadata_path": f"metadata/lidc/{series.series_uid}.json",
    }
    phys_path.parent.mkdir(parents=True, exist_ok=True)
    phys_path.write_text(json.dumps(phys_doc, indent=2), encoding="utf-8")

    return {
        "series_uid": series.series_uid,
        "patient_id": series.patient_id,
        "status": "ok",
        "shape_zyx": list(volume.shape),
        "n_reader_nodules": len(reader_nodules),
        "n_physical_nodules": len(physical),
        "xml": str(xml_path) if xml_path else None,
    }


def run_preprocess(
    raw_root: str | Path,
    out_root: str | Path,
    *,
    xml_root: str | Path | None = None,
    max_series: int | None = None,
    seed: int = 42,
    skip_existing: bool = True,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    for sub in (
        "volumes",
        "metadata/lidc",
        "metadata/physical_nodules",
        "manifests",
    ):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    print(f"Indexing DICOM under {raw_root} ...")
    series_list = discover_ct_series(raw_root)
    print(f"Found {len(series_list)} CT series")
    if max_series is not None:
        series_list = series_list[: max(0, int(max_series))]
        print(f"Limiting to first {len(series_list)} series")

    print(f"Indexing XML under {xml_root or raw_root} ...")
    xml_index = index_xml_files(xml_root) if xml_root else {}
    # Also index XML sitting inside the raw download
    for uid, path in index_xml_files(raw_root).items():
        xml_index.setdefault(uid, path)
    print(f"XML files indexed: {len(xml_index)}")

    xml_fallbacks = [Path(xml_root)] if xml_root else []
    results = []
    for series in tqdm(series_list, desc="preprocess"):
        try:
            results.append(
                process_series(
                    series,
                    out_root=out_root,
                    xml_index=xml_index,
                    xml_fallback_roots=xml_fallbacks,
                    skip_existing=skip_existing,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "series_uid": series.series_uid,
                    "patient_id": series.patient_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    # Build manifests from whatever succeeded
    import sys

    repo_scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(repo_scripts) not in sys.path:
        sys.path.insert(0, str(repo_scripts))
    from build_manifests_from_processed import build_manifests

    man_info = build_manifests(out_root, seed=seed)
    summary = {
        "raw_root": str(raw_root.resolve()),
        "out_root": str(out_root.resolve()),
        "n_series_seen": len(series_list),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped_existing"),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "manifests": man_info,
        "errors": [r for r in results if r.get("status") == "error"][:20],
    }
    (out_root / "manifests" / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess TCIA LIDC-IDRI DICOM+XML into PulmoScan processed/ tree (CPU)"
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="TCIA download folder (e.g. C:/Users/Lenovo/Downloads/TCIA_LIDC-IDRI_20200921)",
    )
    parser.add_argument(
        "--xml-root",
        default=None,
        help="Optional folder/unzip of LIDC XML annotations (if not inside raw-root)",
    )
    parser.add_argument(
        "--out",
        default="./data/processed",
        help="Output processed root",
    )
    parser.add_argument("--max-series", type=int, default=None, help="Optional cap for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess series even if outputs exist",
    )
    args = parser.parse_args()
    summary = run_preprocess(
        args.raw_root,
        args.out,
        xml_root=args.xml_root,
        max_series=args.max_series,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
