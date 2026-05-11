"""Sélection des modules vision à exécuter."""

from __future__ import annotations

VALID_BACKENDS = (
    "none",
    "segformer_generic",
    "geometry_only",
    "auto",
    "yolo_parking",
    "groundingdino_sam",
    "future_custom",
)


def normalize_visual_backend(name: str) -> str:
    n = (name or "auto").strip().lower()
    if n not in VALID_BACKENDS:
        return "auto"
    return n


def runs_geometry(backend: str) -> bool:
    b = normalize_visual_backend(backend)
    if b == "none":
        return False
    if b == "segformer_generic":
        return True
    if b == "geometry_only":
        return True
    if b == "auto":
        return True
    if b in ("yolo_parking", "groundingdino_sam", "future_custom"):
        return True
    return False


def runs_segformer(backend: str) -> bool:
    b = normalize_visual_backend(backend)
    if b == "none" or b == "geometry_only":
        return False
    if b == "segformer_generic":
        return True
    if b == "auto":
        return True
    if b in ("yolo_parking", "groundingdino_sam", "future_custom"):
        return True
    return False


def runs_yolo(backend: str) -> bool:
    return normalize_visual_backend(backend) == "yolo_parking"
