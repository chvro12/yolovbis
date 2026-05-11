"""Fusion masque segmentation YOLO / binaire + masques GIS (bâtiments, routes)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from parking_capacity.gis_fusion import GisFusionResult
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.parking_polygon import (
    contours_to_polygons,
    mask_preprocess,
    raster_polygon_to_wgs84_geojson,
    save_debug_polygon_extraction,
)


@dataclass
class SegmentationGisFusionResult:
    parking_mask_hw: np.ndarray
    usable_mask_hw: np.ndarray
    building_excluded_hw: Optional[np.ndarray]
    road_excluded_hw: Optional[np.ndarray]
    polygons_px: list
    usable_area_m2: float
    theoretical_spaces: int
    geojson: Dict[str, Any]


def mask_bool_from_yolo_result(pred_mask_hw: np.ndarray) -> np.ndarray:
    """Accepte float [0,1], uint8 ou bool H×W."""
    if pred_mask_hw.dtype == np.bool_:
        return pred_mask_hw
    if pred_mask_hw.dtype in (np.float32, np.float64):
        return pred_mask_hw > 0.5
    return pred_mask_hw > 127


def fuse_segmentation_and_gis(
    chip: OrthoChip,
    segmentation_mask_hw: np.ndarray,
    gis: Optional[GisFusionResult],
    *,
    m2_per_space: float = 26.0,
    subtract_roads_from_parking: bool = True,
    morph_close: int = 5,
    min_area_px: int = 120,
) -> SegmentationGisFusionResult:
    """
    segmentation_mask_hw : masque parking prédit (H×W aligné sur la puce).

    Étapes : binarisation → (optionnel) exclusion routes → exclusion bâtiments → surface utile.
    """
    from parking_capacity.imagery_wms import chip_m2_per_pixel

    raw_parking = mask_bool_from_yolo_result(segmentation_mask_hw)
    pm = raw_parking.copy()
    if pm.shape != (chip.height_px, chip.width_px):
        raise ValueError(
            f"Masque {pm.shape} incompatible avec la puce {chip.height_px}x{chip.width_px}"
        )

    road_mask = None
    if subtract_roads_from_parking and gis is not None:
        r = gis.bdtopo_road_mask_hw
        if r is None:
            r = gis.osm_road_mask_hw
        if r is None:
            r = gis.road_mask_gis_hw
        if r is not None and r.shape == pm.shape:
            road_mask = r.astype(np.bool_)
            pm = pm & (~road_mask)

    building_mask = None
    if gis is not None and gis.building_mask_hw is not None:
        building_mask = gis.building_mask_hw.astype(np.bool_)
        if building_mask.shape == pm.shape:
            pm = pm & (~building_mask)

    proc = mask_preprocess(pm, close_kernel=morph_close, min_area_px=min_area_px)
    polys = contours_to_polygons(proc, simplify_eps_px=2.0)

    m2px = chip_m2_per_pixel(chip)
    usable_px = float((proc > 127).sum()) if proc.ndim == 2 else float(pm.sum())
    usable_m2 = usable_px * m2px
    theo = int(max(0, usable_m2 / m2_per_space)) if m2_per_space > 0 else 0

    gj = raster_polygon_to_wgs84_geojson(
        polys,
        width_px=chip.width_px,
        height_px=chip.height_px,
        minx=chip.minx,
        miny=chip.miny,
        maxx=chip.maxx,
        maxy=chip.maxy,
    )

    return SegmentationGisFusionResult(
        parking_mask_hw=raw_parking,
        usable_mask_hw=(proc > 127).astype(np.bool_) if proc.ndim == 2 else pm.astype(np.bool_),
        building_excluded_hw=gis.building_mask_hw if gis else None,
        road_excluded_hw=road_mask,
        polygons_px=polys,
        usable_area_m2=float(usable_m2),
        theoretical_spaces=theo,
        geojson=gj,
    )


def write_fusion_debug_bundle(
    chip: OrthoChip,
    fusion: SegmentationGisFusionResult,
    out_dir: Path,
    *,
    seg_raw_hw: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    mseg = out_dir / "debug_segmentation_mask.png"
    Image.fromarray((fusion.parking_mask_hw.astype(np.uint8) * 255)).save(mseg)
    paths["debug_segmentation_mask"] = str(mseg)

    fus = out_dir / "debug_segmentation_gis_fusion.png"
    rgb = np.array(chip.image.convert("RGB")).astype(np.float32)
    green = np.zeros_like(rgb)
    green[:, :, 1] = fusion.usable_mask_hw.astype(np.float32) * 255.0
    red = np.zeros_like(rgb)
    red[:, :, 0] = fusion.parking_mask_hw.astype(np.float32) * 180.0
    out = np.clip(rgb * 0.55 + green * 0.35 + red * 0.1, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(fus)
    paths["debug_segmentation_gis_fusion"] = str(fus)

    poly_png = out_dir / "debug_polygon_extraction.png"
    proc_u8 = (fusion.usable_mask_hw.astype(np.uint8) * 255)
    if proc_u8.ndim == 2:
        save_debug_polygon_extraction(chip.image, proc_u8, fusion.polygons_px, poly_png)
        paths["debug_polygon_extraction"] = str(poly_png)

    ua = out_dir / "debug_usable_area.png"
    Image.fromarray((fusion.usable_mask_hw.astype(np.uint8) * 255)).save(ua)
    paths["debug_usable_area"] = str(ua)

    gj_path = out_dir / "final_parking_polygon.geojson"
    gj_path.write_text(json.dumps(fusion.geojson, indent=2), encoding="utf-8")
    paths["final_parking_polygon.geojson"] = str(gj_path)

    return paths
