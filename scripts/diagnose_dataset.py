"""Diagnose why a processed LIDC root yields 0 training samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)

    print("DATA_ROOT", root.resolve())
    for rel in [
        "manifests/dataset_manifest.json",
        "manifests/patient_splits.json",
        "volumes",
        "metadata/physical_nodules",
        "metadata/lidc",
    ]:
        p = root / rel
        print(f"  {'OK' if p.exists() else 'MISSING'}: {rel}")

    manifest = json.loads((root / "manifests" / "dataset_manifest.json").read_text())
    splits = {}
    sp = root / "manifests" / "patient_splits.json"
    if sp.exists():
        splits = json.loads(sp.read_text())
    print("\npatient_splits keys:", list(splits.keys()) if isinstance(splits, dict) else type(splits))
    if isinstance(splits, dict):
        for k in ("train", "validation", "val", "test"):
            if k in splits:
                print(f"  {k}: n={len(splits[k])} sample={splits[k][:3]}")

    series = manifest.get("series", manifest.get("studies", manifest))
    if isinstance(series, dict) and "items" in series:
        series = series["items"]
    if isinstance(series, dict):
        series = [{"series_uid": k, **v} for k, v in series.items()]
    print("\nmanifest series count:", len(series) if isinstance(series, list) else type(series))
    if isinstance(series, list) and series:
        print("first series keys:", sorted(series[0].keys()))
        print("first series:", {k: series[0].get(k) for k in list(series[0])[:12]})

    phys_dir = root / "metadata" / "physical_nodules"
    phys_files = sorted(phys_dir.glob("*.json")) if phys_dir.exists() else []
    print("\nphysical_nodules files:", len(phys_files))
    if phys_files:
        phys = json.loads(phys_files[0].read_text())
        print("phys top keys:", sorted(phys.keys()))
        nods = phys.get("physical_nodules", phys.get("nodules", []))
        print("n physical_nodules:", len(nods))
        if nods:
            n0 = nods[0]
            print("nodule0 keys:", sorted(n0.keys()))
            print("nodule0 confidence:", n0.get("confidence"))
            print("nodule0 centroid fields:", {
                k: n0.get(k)
                for k in (
                    "representative_centroid_patient_mm",
                    "centroid_patient_mm",
                    "centroid",
                    "malignancy_mean",
                    "malignancy",
                )
            })
            obs = n0.get("observations", n0.get("reader_observations", []))
            print("n observations:", len(obs))
            if obs:
                print("obs0 keys:", sorted(obs[0].keys()))
                print("obs0 sample:", json.dumps(obs[0], indent=2)[:1200])

    lidc_dir = root / "metadata" / "lidc"
    lidc_files = sorted(lidc_dir.glob("*.json")) if lidc_dir.exists() else []
    print("\nlidc metadata files:", len(lidc_files))
    if lidc_files:
        lidc = json.loads(lidc_files[0].read_text())
        print("lidc top keys:", sorted(lidc.keys()) if isinstance(lidc, dict) else type(lidc))
        text = json.dumps(lidc)
        print("has 'malignancy' string:", "malignancy" in text.lower())
        # print a small malignancy-containing snippet if present
        lower = text.lower()
        idx = lower.find("malignancy")
        if idx >= 0:
            print("malignancy context:", text[max(0, idx - 80) : idx + 120])

    # Try current loader
    try:
        from pulmoscan.data.labels import build_samples, summarize_samples

        samples = build_samples(root)
        print("\nbuild_samples n=", len(samples))
        print(json.dumps(summarize_samples(samples), indent=2))
    except Exception as e:
        print("\nbuild_samples failed:", repr(e))


if __name__ == "__main__":
    main()
