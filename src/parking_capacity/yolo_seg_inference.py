"""Inférence Segmentation YOLOv8 (Ultralytics) sur une image ou une puce."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


def predict_parking_mask(
    image: Union[Image.Image, Path, str],
    weights: Path,
    *,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Retourne un masque booléen H×W (union des instances segmentation parking, classe 0).
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("pip install ultralytics") from e

    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    else:
        pil = image.convert("RGB")
    w, h = pil.size

    model = YOLO(str(weights))
    kwargs = dict(imgsz=imgsz, conf=conf_thres, verbose=False)
    if device:
        kwargs["device"] = device
    results = model.predict(pil, **kwargs)
    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        return np.zeros((h, w), dtype=bool)
    md = r.masks.data.cpu().numpy()
    if md.ndim == 3:
        combined = (md.max(axis=0) > 0.5).astype(np.bool_)
    else:
        combined = (md > 0.5).astype(np.bool_)
    if combined.shape[0] != h or combined.shape[1] != w:
        import cv2

        combined = cv2.resize(combined.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return combined
