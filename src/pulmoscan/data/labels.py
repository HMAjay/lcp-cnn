"""Build labeled nodule samples from processed LIDC manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class NoduleSample:
    series_uid: str
    patient_id: str
    split: str
    physical_id: str
    label: int  # 0 benign, 1 malignant
    malignancy_mean: float
    centroid_patient_mm: list[float]
    volume_path: str
    confidence: str


def _mean_malignancy(observations: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    for obs in observations:
        chars = obs.get("characteristics") or obs.get("nodule_characteristics") or {}
        mal = chars.get("malignancy")
        if mal is None and "malignancy" in obs:
            mal = obs["malignancy"]
        if mal is None:
            continue
        try:
            scores.append(float(mal))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _label_from_score(score: float, malignant_min: float, benign_max: float) -> int | None:
    if score >= malignant_min:
        return 1
    if score <= benign_max:
        return 0
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_samples(
    data_root: str | Path,
    *,
    malignant_min: float = 4.0,
    benign_max: float = 2.0,
    min_confidence: Literal["high", "medium", "low"] = "medium",
    use_physical_nodules: bool = True,
) -> list[NoduleSample]:
    """Index labeled physical nodules from processed LIDC layout."""
    root = Path(data_root)
    manifest_path = root / "manifests" / "dataset_manifest.json"
    splits_path = root / "manifests" / "patient_splits.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Point PULMOSCAN_DATA_ROOT at your processed LIDC root."
        )

    manifest = load_json(manifest_path)
    splits = load_json(splits_path) if splits_path.exists() else {}

    # Normalize split maps: patient_id -> split
    patient_split: dict[str, str] = {}
    if isinstance(splits, dict):
        if "train" in splits or "validation" in splits or "test" in splits:
            for split_name in ("train", "validation", "val", "test"):
                key = "validation" if split_name == "val" else split_name
                for pid in splits.get(split_name, splits.get(key, [])) or []:
                    patient_split[str(pid)] = "validation" if split_name in ("val", "validation") else split_name
        elif "splits" in splits:
            for pid, split_name in splits["splits"].items():
                patient_split[str(pid)] = str(split_name)

    series_entries = manifest.get("series", manifest.get("studies", manifest))
    if isinstance(series_entries, dict) and "items" in series_entries:
        series_entries = series_entries["items"]
    if isinstance(series_entries, dict):
        # map series_uid -> info
        iterable = [{"series_uid": k, **v} for k, v in series_entries.items()]
    else:
        iterable = list(series_entries)

    min_rank = CONFIDENCE_RANK[min_confidence]
    samples: list[NoduleSample] = []

    for entry in iterable:
        series_uid = str(entry.get("series_uid") or entry.get("SeriesInstanceUID") or entry.get("id"))
        patient_id = str(entry.get("patient_id") or entry.get("PatientID") or entry.get("patient") or series_uid)
        split = str(
            entry.get("split")
            or patient_split.get(patient_id)
            or patient_split.get(series_uid)
            or "unassigned"
        )
        if split == "val":
            split = "validation"

        volume_rel = entry.get("volume") or entry.get("volume_path") or f"volumes/{series_uid}.nii.gz"
        volume_path = root / volume_rel
        if not volume_path.exists():
            alt = root / "volumes" / f"{series_uid}.nii.gz"
            volume_path = alt if alt.exists() else volume_path

        phys_path = root / "metadata" / "physical_nodules" / f"{series_uid}.json"
        if not phys_path.exists():
            # allow nested structure from manifests
            phys_rel = entry.get("physical_nodules")
            if phys_rel:
                phys_path = root / phys_rel
        if not phys_path.exists():
            continue

        phys = load_json(phys_path)
        nodules = phys.get("physical_nodules", phys.get("nodules", []))
        if not use_physical_nodules:
            nodules = phys.get("observations", nodules)

        for nod in nodules:
            confidence = str(nod.get("confidence", "medium")).lower()
            if CONFIDENCE_RANK.get(confidence, 0) < min_rank:
                continue
            centroid = (
                nod.get("representative_centroid_patient_mm")
                or nod.get("centroid_patient_mm")
                or nod.get("centroid")
            )
            if centroid is None:
                continue
            centroid = [float(x) for x in centroid[:3]]

            observations = nod.get("observations", nod.get("reader_observations", [nod]))
            score = _mean_malignancy(observations)
            if score is None:
                # synthetic / simplified schema may store malignancy directly
                if "malignancy_mean" in nod:
                    score = float(nod["malignancy_mean"])
                elif "malignancy" in nod:
                    score = float(nod["malignancy"])
                else:
                    continue

            label = _label_from_score(score, malignant_min, benign_max)
            if label is None:
                continue

            samples.append(
                NoduleSample(
                    series_uid=series_uid,
                    patient_id=patient_id,
                    split=split,
                    physical_id=str(nod.get("physical_id") or nod.get("id") or len(samples)),
                    label=label,
                    malignancy_mean=score,
                    centroid_patient_mm=centroid,
                    volume_path=str(volume_path),
                    confidence=confidence,
                )
            )

    return samples


def summarize_samples(samples: list[NoduleSample]) -> dict[str, Any]:
    by_split: dict[str, list[NoduleSample]] = {}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)
    summary = {}
    for split, items in by_split.items():
        n_pos = sum(1 for x in items if x.label == 1)
        summary[split] = {
            "n": len(items),
            "malignant": n_pos,
            "benign": len(items) - n_pos,
            "patients": len({x.patient_id for x in items}),
        }
    return summary
