"""Digital-phantom verification of the LIDC annotation-to-voxel mapping.

Writes DICOM CT phantoms with LIDC-style reader XML, then runs the repository's
corrected mapping (preprocess_lidc.py) and the defective mapping that preceded
commit d87e6d0 on identical inputs. Ground truth is known by construction.

    git show d87e6d0^:src/pulmoscan/data/preprocess_lidc.py > preprocess_lidc_defective.py
    python phantom_geometry_validation.py --repo <repo root> --out <results dir>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

MATRIX = 512
Z_EXTENT_MM = 240.0
NODULES_PER_PHANTOM = 3
READERS = 4
PHANTOMS_PER_SPACING = 20
SPACINGS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
SLICE_SPACINGS = [1.0, 1.25, 2.0, 2.5, 3.0]
VOI = (48, 48, 48)
HU_WINDOW = (-1000.0, 400.0)
HU_AIR, HU_LUNG, HU_TISSUE, HU_NODULE = -1000.0, -850.0, 40.0, 40.0
NOISE_HU = 15.0
CAPTURE_MIN = 0.5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lung_axes(fov: float) -> tuple[list[tuple[float, float]], float, float]:
    """Two lung ellipses as (centre_x, centre_y) offsets from the image centre, plus semi-axes."""
    return [(-0.19 * fov, 0.0), (0.19 * fov, 0.0)], 0.15 * fov, 0.27 * fov


def place_nodules(rng, spacing: float, origin_xyz: np.ndarray, z_max_mm: float) -> list[dict]:
    fov = MATRIX * spacing
    centre = origin_xyz[:2] + fov / 2.0
    lungs, ax, ay = lung_axes(fov)
    nodules: list[dict] = []
    for _ in range(100000):
        if len(nodules) == NODULES_PER_PHANTOM:
            return nodules
        radius = float(rng.uniform(3.0, 12.0))
        lung = lungs[int(rng.integers(0, 2))]
        theta = float(rng.uniform(0, 2 * np.pi))
        rho = float(np.sqrt(rng.uniform(0, 1)))
        margin = 1.0 - (radius + 3.0) / min(ax, ay)
        if margin <= 0:
            continue
        x = centre[0] + lung[0] + rho * margin * ax * np.cos(theta)
        y = centre[1] + lung[1] + rho * margin * ay * np.sin(theta)
        z = float(rng.uniform(origin_xyz[2] + 12 + radius, origin_xyz[2] + z_max_mm - 12 - radius))
        c = np.array([x, y, z], dtype=np.float64)
        if all(np.linalg.norm(c - n["centre_mm"]) > 40.0 for n in nodules):
            nodules.append({"centre_mm": c, "radius_mm": radius})
    raise RuntimeError("could not place nodules")


def write_dicom_series(folder: Path, rng, spacing, dz, n_slices, origin_xyz, nodules, series_uid) -> list[float]:
    fov = MATRIX * spacing
    jj, ii = np.mgrid[0:MATRIX, 0:MATRIX]
    x = origin_xyz[0] + ii * spacing
    y = origin_xyz[1] + jj * spacing
    cx, cy = origin_xyz[0] + fov / 2.0, origin_xyz[1] + fov / 2.0
    base = np.full((MATRIX, MATRIX), HU_AIR, dtype=np.float32)
    base[((x - cx) / (0.44 * fov)) ** 2 + ((y - cy) / (0.33 * fov)) ** 2 <= 1.0] = HU_TISSUE
    lungs, ax, ay = lung_axes(fov)
    for lx, ly in lungs:
        base[((x - cx - lx) / ax) ** 2 + ((y - cy - ly) / ay) ** 2 <= 1.0] = HU_LUNG

    study_uid = generate_uid()
    z_positions: list[float] = []
    for k in range(n_slices):
        z = float(origin_xyz[2] + k * dz)
        z_positions.append(z)
        img = base.copy()
        for nod in nodules:
            c, r = nod["centre_mm"], nod["radius_mm"]
            mask = (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= r * r
            img[mask] = HU_NODULE
        img = img + rng.normal(0.0, NOISE_HU, img.shape)
        stored = np.clip(np.round(img + 1024.0), 0, 4095).astype(np.int16)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{k:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientID = folder.name
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = "digital phantom"
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [float(origin_xyz[0]), float(origin_xyz[1]), z]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [spacing, spacing]
        ds.SliceThickness = dz
        ds.SpacingBetweenSlices = dz
        ds.Rows = ds.Columns = MATRIX
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PixelData = stored.tobytes()
        ds.save_as(str(folder / f"{k:04d}.dcm"))
    return z_positions


def contour_pixels(cx_px, cy_px, r_px) -> list[tuple[int, int]]:
    n = max(16, int(2 * np.pi * max(r_px, 1.0)))
    pts = []
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        p = (int(round(cx_px + r_px * np.cos(t))), int(round(cy_px + r_px * np.sin(t))))
        if p not in pts:
            pts.append(p)
    return pts


def write_reader_xml(path: Path, rng, series_uid, spacing, origin_xyz, z_positions, nodules) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<LidcReadMessage xmlns="http://www.nih.gov">',
        f"<ResponseHeader><SeriesInstanceUid>{series_uid}</SeriesInstanceUid></ResponseHeader>",
    ]
    for reader in range(READERS):
        lines.append("<readingSession>")
        for idx, nod in enumerate(nodules):
            c, r = nod["centre_mm"], nod["radius_mm"]
            jitter = rng.normal(0.0, 0.5, 3)
            rating = int(rng.integers(1, 6))
            lines.append(f"<unblindedReadNodule><noduleID>N{idx}_R{reader}</noduleID>")
            lines.append(f"<characteristics><malignancy>{rating}</malignancy></characteristics>")
            for z in z_positions:
                dz = z - (c[2] + jitter[2])
                if abs(dz) >= r:
                    continue
                r_px = float(np.sqrt(r * r - dz * dz)) / spacing
                cx_px = (c[0] + jitter[0] - origin_xyz[0]) / spacing
                cy_px = (c[1] + jitter[1] - origin_xyz[1]) / spacing
                lines.append(f"<roi><imageZposition>{z:.3f}</imageZposition><inclusion>TRUE</inclusion>")
                for px, py in contour_pixels(cx_px, cy_px, r_px):
                    lines.append(f"<edgeMap><xCoord>{px}</xCoord><yCoord>{py}</yCoord></edgeMap>")
                lines.append("</roi>")
            lines.append("</unblindedReadNodule>")
        lines.append("</readingSession>")
    lines.append("</LidcReadMessage>")
    path.write_text("\n".join(lines), encoding="utf-8")


def nodule_label_volume(shape_zyx, affine, nodules) -> np.ndarray:
    """Ground-truth label volume on the 1 mm grid, computed from the analytic spheres."""
    labels = np.zeros(shape_zyx, dtype=np.uint8)
    inv = np.linalg.inv(affine)
    for i, nod in enumerate(nodules, start=1):
        c, r = nod["centre_mm"], nod["radius_mm"]
        ijk = (inv @ np.append(c, 1.0))[:3]
        pad = int(np.ceil(r)) + 2
        lo = np.maximum(np.floor(ijk).astype(int) - pad, 0)
        hi = np.minimum(np.floor(ijk).astype(int) + pad + 1, np.array(shape_zyx[::-1]))
        xx, yy, zz = np.meshgrid(
            np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]), indexing="ij"
        )
        pts = affine[:3, :3] @ np.stack([xx.ravel(), yy.ravel(), zz.ravel()]) + affine[:3, 3:4]
        inside = np.sum((pts.T - c) ** 2, axis=1) <= r * r
        labels[zz.ravel()[inside], yy.ravel()[inside], xx.ravel()[inside]] = i
    return labels


def score_mapping(name, clusters, nodules, volume, labels, affine, io_mod, record, origin_xyz) -> list[dict]:
    rows = []
    spacing = record["pixel_spacing_mm"]
    for cluster in clusters:
        ids = {int(o["nodule_id"].split("_")[0][1:]) for o in cluster["observations"]}
        assert len(ids) == 1, f"{name}: readers of different nodules were merged"
        idx = ids.pop()
        truth = nodules[idx]
        mapped = np.array(cluster["representative_centroid_patient_mm"], dtype=np.float64)
        centre = io_mod.patient_to_voxel_zyx(mapped, affine)
        crop_hu = io_mod.crop_voi(volume, centre, VOI, fill=HU_WINDOW[0])
        crop_lab = io_mod.crop_voi(labels, centre, VOI, fill=0.0)
        crop_seen = io_mod.crop_voi(np.ones_like(labels, dtype=np.uint8), centre, VOI, fill=0.0)
        n_total = int(np.sum(labels == idx + 1))
        n_in = int(np.sum(crop_lab == idx + 1))
        norm = io_mod.window_normalize(crop_hu, HU_WINDOW, True)
        rows.append(
            {
                **record,
                "mapping": name,
                "nodule": idx,
                "radius_mm": truth["radius_mm"],
                "true_i_px": (truth["centre_mm"][0] - origin_xyz[0]) / spacing,
                "true_j_px": (truth["centre_mm"][1] - origin_xyz[1]) / spacing,
                "n_readers": cluster["num_observations"],
                "error_mm": float(np.linalg.norm(mapped - truth["centre_mm"])),
                "error_xy_mm": float(np.linalg.norm((mapped - truth["centre_mm"])[:2])),
                "error_z_mm": float(abs(mapped[2] - truth["centre_mm"][2])),
                "nodule_fraction_in_voi": n_in / max(n_total, 1),
                "voi_padding_fraction": 1.0 - float(np.mean(crop_seen)),
                "voi_shape_ok": crop_hu.shape == VOI,
                "voi_norm_min": float(norm.min()),
                "voi_norm_max": float(norm.max()),
                "voi_centre_hu": float(np.mean(crop_hu[22:26, 22:26, 22:26])),
            }
        )
    return rows


def run(args) -> None:
    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = load_module("preprocess_fixed", repo / "src/pulmoscan/data/preprocess_lidc.py")
    broken = load_module("preprocess_broken", Path(__file__).parent / "preprocess_lidc_defective.py")
    io_mod = load_module("pulmoscan_io", repo / "src/pulmoscan/data/io.py")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    example_saved = False
    phantom_id = 0
    for spacing in SPACINGS:
        for _ in range(args.per_spacing):
            phantom_id += 1
            dz = float(rng.choice(SLICE_SPACINGS))
            n_slices = int(round(Z_EXTENT_MM / dz))
            fov = MATRIX * spacing
            origin = np.array(
                [
                    -fov / 2.0 + float(rng.uniform(-15, 15)),
                    -fov / 2.0 + float(rng.uniform(-40, 40)),
                    float(rng.uniform(-350, -100)),
                ]
            )
            nodules = place_nodules(rng, spacing, origin, (n_slices - 1) * dz)
            series_uid = generate_uid()
            with tempfile.TemporaryDirectory(dir=args.work) as tmp:
                folder = Path(tmp) / f"PHANTOM-{phantom_id:03d}"
                folder.mkdir()
                z_positions = write_dicom_series(folder, rng, spacing, dz, n_slices, origin, nodules, series_uid)
                xml_path = Path(tmp) / "reader.xml"
                write_reader_xml(xml_path, rng, series_uid, spacing, origin, z_positions, nodules)

                volume, affine, _, original = fixed.load_series_hu_and_affine(folder)
                labels = nodule_label_volume(volume.shape, affine, nodules)
                record = {"phantom": phantom_id, "pixel_spacing_mm": spacing, "slice_spacing_mm": dz}

                _, marks_fixed = fixed.parse_lidc_xml(xml_path, original)
                _, marks_broken = broken.parse_lidc_xml(xml_path, affine)
                clusters_fixed = fixed.cluster_physical_nodules(marks_fixed)
                clusters_broken = broken.cluster_physical_nodules(marks_broken)
                rows += score_mapping("corrected", clusters_fixed, nodules, volume, labels, affine, io_mod, record, origin)
                rows += score_mapping("defective", clusters_broken, nodules, volume, labels, affine, io_mod, record, origin)

                if not example_saved and abs(spacing - 0.70) < 1e-9:
                    good = np.array(clusters_fixed[0]["representative_centroid_patient_mm"])
                    bad = np.array(clusters_broken[0]["representative_centroid_patient_mm"])
                    idx = int(clusters_fixed[0]["observations"][0]["nodule_id"].split("_")[0][1:])
                    cz = io_mod.patient_to_voxel_zyx(good, affine)
                    bz = io_mod.patient_to_voxel_zyx(bad, affine)
                    np.savez_compressed(
                        out / "example_phantom.npz",
                        axial=volume[int(round(cz[0]))],
                        voi_corrected=io_mod.crop_voi(volume, cz, VOI, fill=HU_WINDOW[0]),
                        voi_defective=io_mod.crop_voi(volume, bz, VOI, fill=HU_WINDOW[0]),
                        centre_corrected_zyx=cz,
                        centre_defective_zyx=bz,
                        radius_mm=nodules[idx]["radius_mm"],
                        spacing=spacing,
                        slice_spacing=dz,
                    )
                    example_saved = True
            print(f"phantom {phantom_id} s={spacing:.2f} dz={dz} done", flush=True)

    with (out / "phantom_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "seed": args.seed,
        "phantoms": phantom_id,
        "nodules_per_phantom": NODULES_PER_PHANTOM,
        "readers": READERS,
        "matrix": MATRIX,
        "z_extent_mm": Z_EXTENT_MM,
        "pixel_spacings_mm": SPACINGS,
        "slice_spacings_mm": SLICE_SPACINGS,
        "capture_min_fraction": CAPTURE_MIN,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "SimpleITK": sitk.Version_VersionString(),
        "pydicom": pydicom.__version__,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=None, help="Folder for temporary DICOM files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-spacing", type=int, default=PHANTOMS_PER_SPACING)
    args = parser.parse_args()
    args.work = args.work or tempfile.gettempdir()
    run(args)


if __name__ == "__main__":
    main()
