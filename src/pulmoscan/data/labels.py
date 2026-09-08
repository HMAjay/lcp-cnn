"""Build labeled nodule samples from processed LIDC manifests."""

from __future__ import annotations

import json
import re
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk_malignancy_scores(obj: Any, out: list[float]) -> None:
    """Collect malignancy-like numeric scores anywhere in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in {"malignancy", "malignancy_score", "malignancyrating"}:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    pass
            else:
                _walk_malignancy_scores(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_malignancy_scores(item, out)


def _mean_malignancy(obj: Any) -> float | None:
    scores: list[float] = []
    _walk_malignancy_scores(obj, scores)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _label_from_score(score: float, malignant_min: float, benign_max: float) -> int | None:
    if score >= malignant_min:
        return 1
    if score <= benign_max:
        return 0
    return None


def _normalize_patient_id(pid: str) -> str:
    pid = str(pid).strip()
    m = re.search(r"(\d+)$", pid.replace("-", "").replace("_", ""))
    # Keep original plus bare digits variants for split matching
    return pid


def _patient_id_aliases(pid: str) -> set[str]:
    pid = str(pid).strip()
    aliases = {pid, pid.upper(), pid.lower()}
    # LIDC-IDRI-0001 <-> 0001 <-> 1
    m = re.search(r"(\d{1,4})$", pid)
    if m:
        digits = m.group(1)
        aliases.add(digits)
        aliases.add(digits.lstrip("0") or "0")
        aliases.add(f"LIDC-IDRI-{int(digits):04d}")
        aliases.add(f"LIDC-IDRI-{digits}")
    return aliases


def _load_patient_splits(splits: Any) -> dict[str, str]:
    patient_split: dict[str, str] = {}
    if not isinstance(splits, dict):
        return patient_split

    def _add(pid: Any, split_name: str) -> None:
        split_name = "validation" if split_name in ("val", "validation") else split_name
        for alias in _patient_id_aliases(str(pid)):
            patient_split[alias] = split_name

    if "train" in splits or "validation" in splits or "test" in splits or "val" in splits:
        for split_name in ("train", "validation", "val", "test"):
            key = split_name
            for pid in splits.get(key, []) or []:
                _add(pid, split_name)
    elif "splits" in splits and isinstance(splits["splits"], dict):
        for pid, split_name in splits["splits"].items():
            _add(pid, str(split_name))
    else:
        # flat map patient -> split
        for pid, split_name in splits.items():
            if isinstance(split_name, str) and split_name in {"train", "validation", "val", "test"}:
                _add(pid, split_name)
    return patient_split


def _iter_series_entries(manifest: Any) -> list[dict[str, Any]]:
    series_entries = manifest.get("series", manifest.get("studies", manifest))
    if isinstance(series_entries, dict) and "items" in series_entries:
        series_entries = series_entries["items"]
    if isinstance(series_entries, dict):
        return [{"series_uid": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in series_entries.items()]
    if isinstance(series_entries, list):
        return [e for e in series_entries if isinstance(e, dict)]
    return []


def _index_lidc_malignancy(lidc_dir: Path) -> dict[str, list[float]]:
    """
    Index malignancy scores from parsed LIDC JSON files.
    Keys tried: series_uid, and series_uid::nodule_id variants.
    """
    index: dict[str, list[float]] = {}
    if not lidc_dir.exists():
        return index

    for path in lidc_dir.glob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        series_uid = str(
            data.get("series_uid")
            or data.get("SeriesInstanceUID")
            or path.stem
        )
        scores_all: list[float] = []
        _walk_malignancy_scores(data, scores_all)
        if scores_all:
            index.setdefault(series_uid, []).extend(scores_all)

        # Also collect per reading session / nodule if structure allows
        sessions = data.get("reading_sessions") or data.get("sessions") or []
        if isinstance(sessions, list):
            for sess in sessions:
                if not isinstance(sess, dict):
                    continue
                nodules = sess.get("nodules") or sess.get("unblinded_read_nodule") or []
                if isinstance(nodules, dict):
                    nodules = list(nodules.values())
                for nod in nodules or []:
                    if not isinstance(nod, dict):
                        continue
                    nid = str(nod.get("nodule_id") or nod.get("id") or "")
                    local: list[float] = []
                    _walk_malignancy_scores(nod, local)
                    if local and nid:
                        index.setdefault(f"{series_uid}::{nid}", []).extend(local)
    return index


def _centroid_from_nodule(nod: dict[str, Any]) -> list[float] | None:
    for key in (
        "representative_centroid_patient_mm",
        "centroid_patient_mm",
        "centroid",
        "centroid_zyx",
        "center_mm",
    ):
        val = nod.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            if {"x", "y", "z"} <= set(val):
                return [float(val["x"]), float(val["y"]), float(val["z"])]
            if {"X", "Y", "Z"} <= set(val):
                return [float(val["X"]), float(val["Y"]), float(val["Z"])]
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return [float(val[0]), float(val[1]), float(val[2])]
    return None


def build_samples(
    data_root: str | Path,
    *,
    malignant_min: float = 4.0,
    benign_max: float = 2.0,
    min_confidence: Literal["high", "medium", "low"] = "low",
    use_physical_nodules: bool = True,
    return_stats: bool = False,
) -> list[NoduleSample] | tuple[list[NoduleSample], dict[str, int]]:
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
    patient_split = _load_patient_splits(splits)
    iterable = _iter_series_entries(manifest)
    lidc_index = _index_lidc_malignancy(root / "metadata" / "lidc")

    min_rank = CONFIDENCE_RANK.get(str(min_confidence).lower(), 0)
    samples: list[NoduleSample] = []
    stats = Counterish()

    for entry in iterable:
        series_uid = str(entry.get("series_uid") or entry.get("SeriesInstanceUID") or entry.get("id") or "")
        patient_id = str(
            entry.get("patient_id")
            or entry.get("PatientID")
            or entry.get("patient")
            or entry.get("subject_id")
            or series_uid
        )
        split = str(entry.get("split") or "unassigned")
        if split == "unassigned":
            for alias in _patient_id_aliases(patient_id) | _patient_id_aliases(series_uid):
                if alias in patient_split:
                    split = patient_split[alias]
                    break
        if split == "val":
            split = "validation"

        volume_rel = entry.get("volume") or entry.get("volume_path") or f"volumes/{series_uid}.nii.gz"
        volume_path = root / str(volume_rel)
        if not volume_path.exists():
            alt = root / "volumes" / f"{series_uid}.nii.gz"
            if alt.exists():
                volume_path = alt
            else:
                # fuzzy: any volume containing series uid fragment
                vol_dir = root / "volumes"
                if vol_dir.exists():
                    matches = list(vol_dir.glob(f"*{series_uid}*.nii.gz"))
                    if matches:
                        volume_path = matches[0]
        if not volume_path.exists():
            stats["skip_missing_volume"] += 1
            continue

        phys_path = root / "metadata" / "physical_nodules" / f"{series_uid}.json"
        if not phys_path.exists():
            phys_rel = entry.get("physical_nodules") or entry.get("physical_nodules_path")
            if phys_rel:
                phys_path = root / str(phys_rel)
        if not phys_path.exists():
            # fuzzy match by stem
            phys_dir = root / "metadata" / "physical_nodules"
            if phys_dir.exists():
                matches = list(phys_dir.glob(f"*{series_uid}*.json"))
                if matches:
                    phys_path = matches[0]
        if not phys_path.exists():
            stats["skip_missing_phys"] += 1
            continue

        phys = load_json(phys_path)
        nodules = phys.get("physical_nodules", phys.get("nodules", []))
        if not use_physical_nodules:
            nodules = phys.get("observations", nodules)
        if not nodules:
            stats["skip_empty_phys"] += 1
            continue

        for nod in nodules:
            if not isinstance(nod, dict):
                stats["skip_bad_nodule"] += 1
                continue
            confidence = str(nod.get("confidence", "medium")).lower()
            if CONFIDENCE_RANK.get(confidence, 1) < min_rank:
                stats["skip_confidence"] += 1
                continue

            centroid = _centroid_from_nodule(nod)
            if centroid is None:
                stats["skip_no_centroid"] += 1
                continue

            observations = nod.get("observations", nod.get("reader_observations", [nod]))
            score = _mean_malignancy(observations)
            if score is None:
                score = _mean_malignancy(nod)
            if score is None:
                # fall back to series-level lidc malignancy average
                series_scores = lidc_index.get(series_uid, [])
                # try reader nodule ids
                for obs in observations if isinstance(observations, list) else []:
                    if not isinstance(obs, dict):
                        continue
                    nid = str(obs.get("nodule_id") or obs.get("reader_nodule_id") or obs.get("id") or "")
                    if nid and f"{series_uid}::{nid}" in lidc_index:
                        series_scores = series_scores + lidc_index[f"{series_uid}::{nid}"]
                if series_scores:
                    score = float(sum(series_scores) / len(series_scores))
            if score is None:
                stats["skip_no_malignancy"] += 1
                continue

            label = _label_from_score(score, malignant_min, benign_max)
            if label is None:
                stats["skip_uncertain_label"] += 1
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
            stats["kept"] += 1
            stats[f"split_{split}"] += 1

    if return_stats:
        return samples, stats.as_dict()
    return samples


class Counterish:
    def __init__(self) -> None:
        self._d: dict[str, int] = {}

    def __getitem__(self, key: str) -> int:
        return self._d.get(key, 0)

    def __setitem__(self, key: str, value: int) -> None:
        self._d[key] = value

    def as_dict(self) -> dict[str, int]:
        return dict(self._d)


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
