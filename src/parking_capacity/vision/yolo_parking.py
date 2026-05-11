"""YOLO / détection parking — squelette sans poids obligatoires."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ParkingDetection:
    """Une détection normalisée (boîte ou masque)."""

    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_name: str = "parking_space"
    mask_rle: Optional[str] = None


@dataclass
class YoloParkingResult:
    detections: List[ParkingDetection] = field(default_factory=list)
    model_loaded: bool = False
    error: Optional[str] = None
    raw_backend: str = "none"


def load_yolo_parking_weights(weights: Path) -> Any:
    """
    Charge des poids YOLO (ultralytics) si disponible.
    Lève une ImportError claire si le paquet n'est pas installé.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ultralytics non installé : `pip install ultralytics` pour YOLO parking."
        ) from e
    if not weights.is_file():
        raise FileNotFoundError(f"Poids YOLO introuvables : {weights}")
    return YOLO(str(weights))


def infer_yolo_parking(
    image: Image.Image,
    weights: Optional[Path] = None,
    *,
    conf_th: float = 0.25,
    model: Any = None,
) -> YoloParkingResult:
    """
    Inférence segmentation / bbox parking.

    Sans ``weights`` ni ``model`` : retour vide exploitable (pas d'erreur bloquante).
    """
    if model is None and weights is None:
        return YoloParkingResult(
            detections=[],
            model_loaded=False,
            error=None,
            raw_backend="no_weights",
        )
    try:
        m = model if model is not None else load_yolo_parking_weights(Path(weights))
    except (ImportError, FileNotFoundError, OSError) as e:
        return YoloParkingResult([], False, str(e), "load_failed")

    try:
        arr = np.asarray(image.convert("RGB"))
        out = m.predict(arr, conf=conf_th, verbose=False)
        dets: List[ParkingDetection] = []
        for r in out:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xy = b.xyxy.cpu().numpy().flatten().tolist()
                cf = float(b.conf.cpu().numpy().flatten()[0])
                dets.append(ParkingDetection(xyxy=tuple(xy[:4]), confidence=cf))
        return YoloParkingResult(dets, True, None, "ultralytics")
    except Exception as e:  # noqa: BLE001
        return YoloParkingResult([], True, str(e), "inference_error")


def yolo_capacity_hint(result: YoloParkingResult, *, m2_per_slot: float = 26.0, m2_per_pixel: float) -> Optional[int]:
    """Ordre de grandeur places à partir des boîtes (aire cumulée / m² par place)."""
    if not result.detections:
        return None
    area_px = 0.0
    for d in result.detections:
        x1, y1, x2, y2 = d.xyxy
        area_px += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    m2 = area_px * m2_per_pixel
    if m2 <= 1:
        return None
    return max(1, int(round(m2 / m2_per_slot)))
