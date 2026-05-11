"""Inspection des jeux : vues caméra vs satellite, masques, adéquation orthophoto."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from parking_capacity.datasets_satellite.apklot import (
    classify_apklot_path,
    find_apklot_labelme_dirs,
    find_apklot_voc_roots_with_views,
)
from parking_capacity.datasets_satellite.converters import dir_size_bytes
from parking_capacity.datasets_satellite.download_utils import project_data_datasets_dir
from parking_capacity.datasets_satellite.registry import load_registry


def _repo_root(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root).parent.parent


def _resolve_registered_path(project_root: Optional[Path], info: Dict[str, Any], key: str) -> Path:
    p = Path(info[key])
    if p.is_absolute():
        return p.resolve()
    return (_repo_root(project_root) / p).resolve()


def _image_exts() -> Tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def _count_images_apklot_raw(raw: Path, *, max_files: int = 6000) -> Tuple[int, Counter]:
    exts = {e.lower() for e in _image_exts()}
    view_counts: Counter = Counter()
    n = 0
    for p in raw.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        n += 1
        view_counts[classify_apklot_path(p)] += 1
        if n >= max_files:
            break
    return n, view_counts


def _count_images_generic(raw: Path, *, max_files: int = 6000) -> int:
    exts = {e.lower() for e in _image_exts()}
    n = 0
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
            if n >= max_files:
                break
    return n


def _glob_sample_images(root: Path, limit: int = 36) -> List[Path]:
    out: List[Path] = []
    exts = _image_exts()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def _resolution_sample(paths: List[Path], max_samples: int = 24) -> Dict[str, Any]:
    widths: List[int] = []
    heights: List[int] = []
    for p in paths[:max_samples]:
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except OSError:
            continue
    if not widths:
        return {"samples": 0}
    return {
        "samples": len(widths),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "mean_mpixels": sum(w * h for w, h in zip(widths, heights)) / len(widths) / 1e6,
    }


def _mask_presence_apklot(raw: Path) -> Dict[str, bool]:
    voc_roots = find_apklot_voc_roots_with_views(raw)
    voc_masks = any((v[0] / "SegmentationClass").is_dir() for v in voc_roots)
    lm = find_apklot_labelme_dirs(raw)
    return {"voc_segmentation_class": voc_masks, "labelme_json_dirs": len(lm)}


def _suitability_apklot(view_counts: Counter, masks: Dict[str, Any]) -> str:
    sat = view_counts.get("satellite", 0)
    cam = view_counts.get("camera", 0)
    if sat >= 10 and (masks.get("voc_segmentation_class") or masks.get("labelme_json_dirs", 0) > 0):
        return "high"
    if sat >= 1:
        return "medium"
    if cam > 0 and sat == 0:
        return "low"
    if view_counts.get("unknown", 0) > 0 and sat == 0 and cam == 0:
        return "unknown"
    return "none"


def inspect_dataset(dataset: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Rapport pour ``parking-capacity inspect-dataset``."""
    reg = load_registry(project_root)
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        return {"error": f"Dataset inconnu dans le registre : {dataset}"}

    raw_n = _resolve_registered_path(project_root, info, "raw_path")
    prep_n = _resolve_registered_path(project_root, info, "prepared_path")

    out: Dict[str, Any] = {
        "dataset": dataset,
        "dataset_type": info.get("dataset_type"),
        "registry": {
            "source_url": info.get("source_url"),
            "status": info.get("status"),
            "notes": info.get("notes"),
        },
        "paths": {"raw": str(raw_n), "prepared": str(prep_n)},
    }

    if dataset == "apklot":
        out["upstream_reference"] = {
            "paper": "https://www.mdpi.com/2076-3417/10/15/5364",
            "repository_layout": (
                "Branches/releases : dépôt unique ``master`` ; données volumineuses via Git LFS. "
                "Dossiers documentés : ``1. Satellite`` (captures Google Maps API, vue plongeante) "
                "et ``2. Camera`` + LabelMe (vue caméra / surveillance). "
                "Sans ``git lfs pull``, souvent seuls des pointeurs ou la partie caméra sont présents."
            ),
        }
        if raw_n.is_dir():
            voc_views = find_apklot_voc_roots_with_views(raw_n)
            out["voc_layouts"] = [{"path": str(v[0]), "view": v[1]} for v in voc_views]
            n_img, vcnt = _count_images_apklot_raw(raw_n)
            masks = _mask_presence_apklot(raw_n)
            out["raw_images_scanned"] = n_img
            out["raw_images_by_view_path"] = dict(vcnt)
            out["mask_sources"] = masks
            out["suitability_for_satellite_parking"] = _suitability_apklot(vcnt, masks)
            out["resolution_sample_raw"] = _resolution_sample(_glob_sample_images(raw_n, 40))
            dom = max(vcnt, key=lambda k: vcnt[k]) if vcnt else None
            out["dominant_path_view"] = dom
        else:
            out["suitability_for_satellite_parking"] = "unknown"

        meta_path = prep_n / "dataset_prepare_meta.json"
        if meta_path.is_file():
            out["prepare_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

        uni = prep_n / "parking_capacity_dataset" / "images"
        if uni.is_dir():
            imgs = [p for p in uni.iterdir() if p.is_file()][:48]
            out["prepared_unified_image_files"] = sum(1 for _ in uni.iterdir() if _.is_file())
            out["resolution_sample_prepared"] = _resolution_sample(imgs)

        return out

    dt = info.get("dataset_type") or "satellite"
    out["camera_vs_satellite"] = {
        "note": "Jeu aerial/satellite classique (pas de sous-arbre APKLOT caméra).",
        "dataset_type": dt,
    }
    if raw_n.is_dir():
        out["raw_images_scanned"] = _count_images_generic(raw_n)
        out["raw_size_bytes"] = dir_size_bytes(raw_n)
        out["resolution_sample_raw"] = _resolution_sample(_glob_sample_images(raw_n, 40))
    out["suitability_for_satellite_parking"] = "high" if dt in ("satellite", "aerial", "mixed") else "medium"
    return out
