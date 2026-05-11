"""Pipeline inférence « deep satellite » : segmentation + exclusion GIS + estimation surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from parking_capacity.gis_fusion import GisFusionResult
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.vision_estimate import chip_m2_per_pixel_from_chip, segment_parking_on_chip


@dataclass
class SatelliteDeepResult:
    parking_mask_hw: np.ndarray
    usable_mask_hw: np.ndarray
    theoretical_spaces: int
    parking_area_m2: float
    usable_area_m2: float
    extras: Dict[str, Any]


def _fuse_usable(
    parking_hw: np.ndarray,
    gis: Optional[GisFusionResult],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Parking ∩ ¬bâtiments (masques alignés H×W)."""
    info: Dict[str, Any] = {}
    usable = parking_hw.astype(np.bool_)
    if gis is None:
        return usable, info
    b = gis.building_mask_hw
    if b is not None and b.shape == usable.shape:
        usable = usable & (~b.astype(np.bool_))
        info["buildings_subtracted"] = True
    else:
        info["buildings_subtracted"] = False
    return usable, info


def theoretical_spaces_from_mask(
    usable_mask_hw: np.ndarray,
    chip: OrthoChip,
    m2_per_space: float,
) -> tuple[float, int]:
    m2px = chip_m2_per_pixel_from_chip(chip)
    area_px = float(usable_mask_hw.sum())
    area_m2 = area_px * m2px
    cap = int(max(0, area_m2 / m2_per_space)) if m2_per_space > 0 else 0
    return area_m2, cap


def run_satellite_deep_inference(
    chip: OrthoChip,
    *,
    gis: Optional[GisFusionResult] = None,
    m2_per_space: float = 26.0,
    device: Optional[str] = None,
) -> SatelliteDeepResult:
    seg = segment_parking_on_chip(chip, m2_per_space=m2_per_space, device=device)
    if seg is None:
        raise RuntimeError("segment_parking_on_chip a retourné None (vision désactivée).")
    usable, fuse_info = _fuse_usable(seg.parking_mask_hw, gis)
    usable_m2, cap = theoretical_spaces_from_mask(usable, chip, m2_per_space)
    park_m2 = float(seg.parking_mask_hw.sum()) * chip_m2_per_pixel_from_chip(chip)
    return SatelliteDeepResult(
        parking_mask_hw=seg.parking_mask_hw,
        usable_mask_hw=usable,
        theoretical_spaces=cap,
        parking_area_m2=park_m2,
        usable_area_m2=usable_m2,
        extras={
            "vision_estimate": seg.estimate,
            "fusion": fuse_info,
            "m2_per_space": m2_per_space,
        },
    )


def save_debug_segmentation_mask(mask_hw: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (mask_hw.astype(np.float32) * 255).astype(np.uint8)
    Image.fromarray(u8).save(path)


def save_debug_parking_polygon(path: Path, chip: OrthoChip, mask_hw: np.ndarray) -> None:
    """Contour principal du masque sur fond image (aperçu rapide)."""
    try:
        import cv2
    except ImportError:
        save_debug_segmentation_mask(mask_hw, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.array(chip.image.convert("RGB"))
    m = (mask_hw.astype(np.uint8) * 255)
    cts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cts, -1, (0, 255, 0), 2)
    Image.fromarray(rgb).save(path)


def save_debug_slot_detection(path: Path, chip: OrthoChip, slot_overlay: np.ndarray) -> None:
    """``slot_overlay`` : image H×W×3 ou masque uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if slot_overlay.ndim == 2:
        rgb = np.array(chip.image.convert("RGB"))
        colored = np.zeros_like(rgb)
        colored[:, :, 1] = slot_overlay
        out = (rgb.astype(np.float32) * 0.6 + colored.astype(np.float32) * 0.4).astype(np.uint8)
        Image.fromarray(out).save(path)
    else:
        Image.fromarray(slot_overlay.astype(np.uint8)).save(path)


def save_debug_gis_segmentation_fusion(
    path: Path,
    chip: OrthoChip,
    parking_hw: np.ndarray,
    usable_hw: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.array(chip.image.convert("RGB")).astype(np.float32)
    red = np.zeros_like(rgb)
    red[:, :, 0] = parking_hw.astype(np.float32) * 255
    green = np.zeros_like(rgb)
    green[:, :, 1] = usable_hw.astype(np.float32) * 255
    out = np.clip(rgb * 0.5 + red * 0.25 + green * 0.25, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def write_satellite_debug_bundle(
    out_dir: Path,
    chip: OrthoChip,
    result: SatelliteDeepResult,
    slot_overlay: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Écrit les PNG listés dans la section Debug du cahier des charges."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "debug_segmentation_mask": str(out_dir / "debug_segmentation_mask.png"),
        "debug_parking_polygon": str(out_dir / "debug_parking_polygon.png"),
        "debug_gis_segmentation_fusion": str(out_dir / "debug_gis_segmentation_fusion.png"),
    }
    save_debug_segmentation_mask(result.parking_mask_hw, Path(paths["debug_segmentation_mask"]))
    save_debug_parking_polygon(Path(paths["debug_parking_polygon"]), chip, result.parking_mask_hw)
    save_debug_gis_segmentation_fusion(
        Path(paths["debug_gis_segmentation_fusion"]),
        chip,
        result.parking_mask_hw,
        result.usable_mask_hw,
    )
    if slot_overlay is not None:
        p = out_dir / "debug_slot_detection.png"
        save_debug_slot_detection(p, chip, slot_overlay)
        paths["debug_slot_detection"] = str(p)
    return paths
