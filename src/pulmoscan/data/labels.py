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


class _Counter:
    def __init__(self) -> None:
        self._d: dict[str, int] = {}

    def inc(self, key: str, n: int = 1) -> None:
        self._d[key] = self._d.get(key, 0) + n

    def as_dict(self) -> dict[str, int]:
        return dict(self._d)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk_malignancy_scores(obj: Any, out: list[float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in {"malignancy", "malignancy_score", "malignancyrating"}:
                if v is None or v == "":
                    continue
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    pass
            else:
                _walk_malignancy_scores(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_malignancy_scores(item, out)


def _norm_nid(nid: Any) -> str:
    return str(nid or "").strip().lower()


def _mean_or_none(scores: list[float]) -> float | None:
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _label_from_score(score: float, malignant_min: float, benign_max: float) -> int | None:
    if score >= malignant_min:
        return 1
    if score <= benign_max:
        return 0
    return None


def _patient_id_aliases(pid: str) -> set[str]:
    pid = str(pid).strip()
    aliases = {pid, pid.upper(), pid.lower()}
    m = re.search(r"(\d{1,4})$", pid)
    if m:
        digits = m.group(1)
        aliases.add(digits)
        aliases.add(digits.lstrip("0") or "0")
        aliases.add(f"LIDC-IDRI-{int(digits):04d}")
        aliases.add(f"LIDC-IDRI-{digits}")
    return aliases


def _strip_processed_prefix(rel: str) -> str:
    """Map manifest paths like data/processed/volumes/... -> volumes/..."""
    rel = str(rel).replace("\\", "/").lstrip("./")
    for prefix in ("data/processed/", "processed/", "data/"):
        if rel.startswith(prefix):
            return rel[len(prefix) :]
    return rel


def _resolve_under_root(root: Path, rel: str | None, fallbacks: list[Path] | None = None) -> Path | None:
    if rel:
        cleaned = _strip_processed_prefix(rel)
        candidates = [root / cleaned, root / rel, Path(rel)]
        for c in candidates:
            if c.exists():
                return c
    for fb in fallbacks or []:
        if fb.exists():
            return fb
    return None


def _load_patient_splits(splits_obj: Any) -> dict[str, str]:
    patient_split: dict[str, str] = {}

    def _add(pid: Any, split_name: str) -> None:
        split_name = "validation" if split_name in ("val", "validation") else str(split_name)
        for alias in _patient_id_aliases(str(pid)):
            patient_split[alias] = split_name

    if not isinstance(splits_obj, dict):
        return patient_split

    # Real schema: {"schema_version": "...", "splits": {"LIDC-IDRI-0046": "train", ...}}
    nested = splits_obj.get("splits")
    if isinstance(nested, dict) and nested and not any(
        k in nested for k in ("train", "validation", "val", "test")
    ):
        for pid, split_name in nested.items():
            if isinstance(split_name, str):
                _add(pid, split_name)
        return patient_split

    if "train" in splits_obj or "validation" in splits_obj or "test" in splits_obj or "val" in splits_obj:
        for split_name in ("train", "validation", "val", "test"):
            for pid in splits_obj.get(split_name, []) or []:
                _add(pid, split_name)
        return patient_split

    if isinstance(nested, dict):
        # {"splits": {"train": [...], ...}}
        for split_name in ("train", "validation", "val", "test"):
            for pid in nested.get(split_name, []) or []:
                _add(pid, split_name)
    return patient_split


def _iter_series_entries(manifest: Any) -> list[dict[str, Any]]:
    series_entries = manifest.get("series", manifest.get("studies", manifest))
    if isinstance(series_entries, dict) and "items" in series_entries:
        series_entries = series_entries["items"]
    if isinstance(series_entries, dict):
        # Maybe {"count": N, "items": [...]} already handled; else uid->info
        out = []
        for k, v in series_entries.items():
            if isinstance(v, dict):
                row = dict(v)
                row.setdefault("series_instance_uid", k)
                out.append(row)
        return out
    if isinstance(series_entries, list):
        return [e for e in series_entries if isinstance(e, dict)]
    return []


def _centroid_from_nodule(nod: dict[str, Any]) -> list[float] | None:
    for key in (
        "representative_centroid_patient_mm",
        "centroid_patient_mm",
        "centroid_patient",
        "centroid",
        "center_mm",
    ):
        val = nod.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            if {"x", "y", "z"} <= set(val.keys()):
                return [float(val["x"]), float(val["y"]), float(val["z"])]
            if {"X", "Y", "Z"} <= set(val.keys()):
                return [float(val["X"]), float(val["Y"]), float(val["Z"])]
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return [float(val[0]), float(val[1]), float(val[2])]
    return None


def _index_lidc_file(data: Any) -> dict[str, list[float]]:
    """
    Build maps for reading_sessions[].nodules[].characteristics.malignancy:
      session_index::nodule_id -> scores
      nodule_id -> scores
      session_index::#idx -> scores
    """
    index: dict[str, list[float]] = {}

    def _add(key: str, scores: list[float]) -> None:
        if not key or not scores:
            return
        index.setdefault(key, []).extend(scores)

    sessions = []
    if isinstance(data, dict):
        sessions = (
            data.get("reading_sessions")
            or data.get("sessions")
            or data.get("unblindedReadNodule")
            or []
        )
        if not sessions and "nodules" in data:
            sessions = [{"nodules": data.get("nodules"), "session_index": 0}]

    if isinstance(sessions, dict):
        sessions = list(sessions.values())

    for si, sess in enumerate(sessions or []):
        if not isinstance(sess, dict):
            continue
        session_index = int(sess.get("session_index", sess.get("id", si)))
        nodules = (
            sess.get("nodules")
            or sess.get("unblinded_read_nodule")
            or sess.get("unblindedReadNodule")
            or []
        )
        if isinstance(nodules, dict):
            nodules = list(nodules.values())
        for idx, nod in enumerate(nodules or []):
            if not isinstance(nod, dict):
                continue
            raw_nid = nod.get("nodule_id") or nod.get("noduleID") or nod.get("id") or ""
            nid = _norm_nid(raw_nid)
            local: list[float] = []
            chars = nod.get("characteristics") or nod.get("nodule_characteristics") or {}
            if isinstance(chars, dict) and chars.get("malignancy") is not None:
                try:
                    local.append(float(chars["malignancy"]))
                except (TypeError, ValueError):
                    pass
            if not local:
                _walk_malignancy_scores(chars, local)
            _add(f"{session_index}::#{idx}", local)
            if nid:
                _add(nid, local)
                _add(f"{session_index}::{nid}", local)
                _add(_norm_nid(raw_nid), local)
    return index


def _series_uid_from_lidc_doc(data: Any, fallback: str = "") -> str:
    if not isinstance(data, dict):
        return fallback
    for key in (
        "series_instance_uid",
        "series_uid",
        "SeriesInstanceUID",
        "seriesInstanceUid",
    ):
        if data.get(key):
            return str(data[key])
    # sometimes nested under study/series
    for nest in ("series", "study", "meta", "metadata"):
        sub = data.get(nest)
        if isinstance(sub, dict):
            for key in ("series_instance_uid", "series_uid", "SeriesInstanceUID"):
                if sub.get(key):
                    return str(sub[key])
    return fallback


def _series_uid_from_text(text: str) -> str:
    m = re.search(r'"series_instance_uid"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    m = re.search(r'"SeriesInstanceUID"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else ""


def build_lidc_path_index(root: Path) -> dict[str, Path]:
    """Map series_uid -> lidc json path.

    Lazy/cheap at startup:
    - reuse manifests/lidc_path_index.json if present
    - otherwise scan files with a lightweight regex (no full JSON parse)
    - write cache for next run
    """
    lidc_dir = root / "metadata" / "lidc"
    cache_path = root / "manifests" / "lidc_path_index.json"
    index: dict[str, Path] = {}
    if not lidc_dir.exists():
        return index

    if cache_path.exists():
        try:
            raw = load_json(cache_path)
            for uid, rel in (raw or {}).items():
                p = root / rel if not Path(rel).is_absolute() else Path(rel)
                if p.exists():
                    index[str(uid)] = p
            if index:
                return index
        except Exception:
            index = {}

    for path in lidc_dir.rglob("*.json"):
        try:
            # Read a small prefix first; fall back to full text if needed
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                chunk = f.read(4096)
                uid = _series_uid_from_text(chunk)
                if not uid:
                    uid = _series_uid_from_text(chunk + f.read(65536))
        except Exception:
            continue
        if uid:
            index[uid] = path

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            uid: str(path.relative_to(root)).replace("\\", "/")
            for uid, path in index.items()
        }
        cache_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except Exception:
        pass
    return index


def _load_lidc_index_for_series(
    root: Path,
    lidc_rel: str | None,
    series_uid: str,
    lidc_path_index: dict[str, Path] | None = None,
) -> dict[str, list[float]]:
    fallbacks: list[Path] = []
    if lidc_path_index and series_uid in lidc_path_index:
        fallbacks.append(lidc_path_index[series_uid])
    if (root / "metadata" / "lidc").exists():
        fallbacks.extend((root / "metadata" / "lidc").rglob(f"*{series_uid}*.json"))

    path = _resolve_under_root(root, lidc_rel, fallbacks=fallbacks)
    if path is None and lidc_path_index and series_uid in lidc_path_index:
        path = lidc_path_index[series_uid]
    if path is None or not path.exists():
        return {}
    try:
        return _index_lidc_file(load_json(path))
    except Exception:
        return {}


def _score_from_observations(
    observations: list[Any],
    lidc_index: dict[str, list[float]],
) -> float | None:
    scores: list[float] = []
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        local: list[float] = []
        _walk_malignancy_scores(obs, local)
        if local:
            scores.extend(local)
            continue
        nid = _norm_nid(obs.get("nodule_id") or obs.get("id") or "")
        si = obs.get("session_index")
        keys = []
        if si is not None and nid:
            keys.append(f"{int(si)}::{nid}")
        if nid:
            keys.append(nid)
        for key in keys:
            if key in lidc_index:
                scores.extend(lidc_index[key])
                break
    return _mean_or_none(scores)


def build_samples(
    data_root: str | Path,
    *,
    malignant_min: float = 4.0,
    benign_max: float = 2.0,
    min_confidence: Literal["high", "medium", "low"] = "low",
    use_physical_nodules: bool = True,
    return_stats: bool = False,
) -> list[NoduleSample] | tuple[list[NoduleSample], dict[str, int]]:
    root = Path(data_root)
    manifest_path = root / "manifests" / "dataset_manifest.json"
    splits_path = root / "manifests" / "patient_splits.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Point PULMOSCAN_DATA_ROOT at your processed LIDC root."
        )

    manifest = load_json(manifest_path)
    splits_obj = load_json(splits_path) if splits_path.exists() else {}
    patient_split = _load_patient_splits(splits_obj)
    iterable = _iter_series_entries(manifest)
    lidc_path_index = build_lidc_path_index(root)
    stats = _Counter()
    stats.inc("lidc_files_indexed", len(lidc_path_index))

    min_rank = CONFIDENCE_RANK.get(str(min_confidence).lower(), 0)
    samples: list[NoduleSample] = []

    for entry in iterable:
        series_uid = str(
            entry.get("series_instance_uid")
            or entry.get("series_uid")
            or entry.get("SeriesInstanceUID")
            or entry.get("id")
            or ""
        )
        patient_id = str(
            entry.get("patient_id")
            or entry.get("PatientID")
            or entry.get("patient")
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

        volume_path = _resolve_under_root(
            root,
            entry.get("volume_path") or entry.get("volume"),
            fallbacks=[root / "volumes" / f"{series_uid}.nii.gz"],
        )
        if volume_path is None:
            stats.inc("skip_missing_volume")
            continue

        phys_path = _resolve_under_root(
            root,
            entry.get("physical_nodule_metadata_path")
            or entry.get("physical_nodules")
            or entry.get("physical_nodules_path"),
            fallbacks=[root / "metadata" / "physical_nodules" / f"{series_uid}.json"],
        )
        if phys_path is None:
            stats.inc("skip_missing_phys")
            continue

        lidc_index = _load_lidc_index_for_series(
            root,
            entry.get("lidc_metadata_path") or entry.get("lidc_metadata"),
            series_uid,
            lidc_path_index=lidc_path_index,
        )
        if not lidc_index:
            stats.inc("series_without_lidc_malignancy")

        phys = load_json(phys_path)
        nodules = phys.get("physical_nodules", phys.get("nodules", []))
        if not use_physical_nodules:
            nodules = phys.get("observations", nodules)
        if not nodules:
            stats.inc("skip_empty_phys")
            continue

        for nod in nodules:
            if not isinstance(nod, dict):
                stats.inc("skip_bad_nodule")
                continue

            confidence = str(nod.get("confidence", "medium")).lower()
            if CONFIDENCE_RANK.get(confidence, 1) < min_rank:
                stats.inc("skip_confidence")
                continue

            centroid = _centroid_from_nodule(nod)
            if centroid is None:
                stats.inc("skip_no_centroid")
                continue

            observations = nod.get("observations", nod.get("reader_observations", []))
            if not isinstance(observations, list) or not observations:
                observations = [nod]

            score = _score_from_observations(observations, lidc_index)
            if score is None:
                score = _mean_or_none([])
                # direct fields on nodule
                direct: list[float] = []
                _walk_malignancy_scores(nod, direct)
                score = _mean_or_none(direct)
            if score is None:
                stats.inc("skip_no_malignancy")
                continue

            label = _label_from_score(score, malignant_min, benign_max)
            if label is None:
                stats.inc("skip_uncertain_label")
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
            stats.inc("kept")
            stats.inc(f"split_{split}")

    if return_stats:
        return samples, stats.as_dict()
    return samples


def summarize_samples(samples: list[NoduleSample]) -> dict[str, Any]:
    by_split: dict[str, list[NoduleSample]] = {}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)
    summary: dict[str, Any] = {}
    for split, items in by_split.items():
        n_pos = sum(1 for x in items if x.label == 1)
        summary[split] = {
            "n": len(items),
            "malignant": n_pos,
            "benign": len(items) - n_pos,
            "patients": len({x.patient_id for x in items}),
        }
    return summary
