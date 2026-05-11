"""Inférence multi-échelle (plusieurs demi-côtés de puce) et fusion des masques."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import httpx
import numpy as np

from parking_capacity.imagery_wms import OrthoChip, fetch_ortho_chip

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def resize_mask_to(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Réduit/agrandit un masque H×W (nearest)."""
    th, tw = shape_hw
    if mask.shape == shape_hw:
        return mask
    if cv2 is None:
        raise RuntimeError("opencv requis pour resize_mask_to")
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)


def fuse_masks_max(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Union logique (recall)."""
    if not masks:
        raise ValueError("masks vide")
    ref = masks[0].shape
    stack = []
    for m in masks:
        stack.append(resize_mask_to(m.astype(np.uint8), ref).astype(bool))
    out = stack[0].copy()
    for m in stack[1:]:
        out |= m
    return out


def fuse_masks_mean_threshold(masks: Sequence[np.ndarray], thr: float = 0.5) -> np.ndarray:
    """Moyenne des masques {0,1} puis seuil — compromis."""
    if not masks:
        raise ValueError("masks vide")
    ref = masks[0].shape
    acc = np.zeros(ref, dtype=np.float32)
    for m in masks:
        acc += resize_mask_to(m.astype(np.uint8), ref).astype(np.float32)
    acc /= len(masks)
    return acc >= thr


def multiscale_chips_and_masks(
    lon: float,
    lat: float,
    predict_fn: Callable[[OrthoChip], np.ndarray],
    *,
    half_sides_m: Sequence[float] = (25.0, 50.0, 80.0),
    chip_pixels: int = 640,
    client: Optional[httpx.Client] = None,
    cache_dir: Optional[Path] = None,
    refresh_imagery: bool = False,
) -> Tuple[List[OrthoChip], List[np.ndarray], np.ndarray]:
    """
    Télécharge des puces à plusieurs échelles (même résolution px), infère chaque masque.

    ``predict_fn`` prend un ``OrthoChip`` et renvoie un masque booléen H×W.

    Fusion : union (max) des masques redimensionnés sur la plus grande emprise (dernière échelle).
    Note : alignement géométrique approximatif (centrage commun), voir doc produit.
    """
    chips: List[OrthoChip] = []
    masks: List[np.ndarray] = []
    close_client = False
    if client is None:
        client = httpx.Client(timeout=120.0)
        close_client = True
    try:
        for hs in half_sides_m:
            chip = fetch_ortho_chip(
                lon,
                lat,
                half_side_m=float(hs),
                width_px=chip_pixels,
                height_px=chip_pixels,
                client=client,
                cache_dir=cache_dir,
                refresh_imagery=refresh_imagery,
            )
            chips.append(chip)
            masks.append(predict_fn(chip))
    finally:
        if close_client:
            client.close()

    ref_shape = masks[-1].shape
    fused = fuse_masks_max([resize_mask_to(m, ref_shape) for m in masks])
    return chips, masks, fused


def save_debug_multiscale_fusion(
    rgb_ref: "Image.Image",
    fused_mask: np.ndarray,
    path: Path,
) -> None:
    from PIL import Image as I

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.array(rgb_ref.convert("RGB")).astype(np.float32)
    ov = np.zeros_like(rgb)
    ov[:, :, 1] = fused_mask.astype(np.float32) * 255.0
    out = np.clip(rgb * 0.55 + ov * 0.45, 0, 255).astype(np.uint8)
    I.fromarray(out).save(path)
