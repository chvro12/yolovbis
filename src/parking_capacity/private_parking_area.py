"""Masque et métriques de surface privée exploitable pour capacité de garage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from parking_capacity.gis_fusion import GisFusionResult
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.surface_classification import SurfaceClassification

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


@dataclass
class PrivateParkingArea:
    """Surface privée candidate pour le stationnement théorique."""

    parcel_mask: Optional[np.ndarray]
    usable_mask: Optional[np.ndarray]
    building_mask: Optional[np.ndarray]
    road_mask: Optional[np.ndarray]
    parcel_area_m2: float = 0.0
    usable_area_m2: float = 0.0
    building_area_m2: float = 0.0
    road_area_m2: float = 0.0
    usable_ratio: float = 0.0
    source: str = "none"
    notes: list[str] | None = None

    def to_debug_dict(self) -> dict:
        return {
            "source": self.source,
            "parcel_area_m2": round(self.parcel_area_m2, 2),
            "usable_area_m2": round(self.usable_area_m2, 2),
            "building_area_m2": round(self.building_area_m2, 2),
            "road_area_m2": round(self.road_area_m2, 2),
            "usable_ratio": round(self.usable_ratio, 4),
            "notes": list(self.notes or []),
        }


def lonlat_polyline_to_pixels(points: list, chip: OrthoChip) -> Optional[np.ndarray]:
    """Convertit un anneau lon/lat en coordonnées pixel relatives à la puce EPSG:3857."""
    try:
        pts = []
        earth_r = 6378137.0
        span_x = max(chip.maxx - chip.minx, 1e-9)
        span_y = max(chip.maxy - chip.miny, 1e-9)
        for lon, lat in points:
            mx = math.radians(float(lon)) * earth_r
            my = math.log(math.tan(math.pi / 4.0 + math.radians(float(lat)) / 2.0)) * earth_r
            u = (mx - chip.minx) / span_x * chip.width_px
            v = (chip.maxy - my) / span_y * chip.height_px
            pts.append((u, v))
        if len(pts) < 3:
            return None
        return np.asarray(pts, dtype=np.float32)
    except Exception:
        return None


def mask_from_polygon_px(poly_px: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if poly_px is None or not _HAS_CV2 or poly_px.shape[0] < 3:
        return None
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px.astype(np.int32)], 1)
    return mask.astype(bool)


def compute_private_parking_area(
    chip: OrthoChip,
    surface: SurfaceClassification,
    *,
    parcel_polygon_lonlat: Optional[list],
    fusion: Optional[GisFusionResult],
) -> PrivateParkingArea:
    """Calcule ``parcelle stricte ∩ surface garable``, hors bâtiments et routes publiques."""
    h, w = surface.parking_eligible_mask.shape[:2]
    m2_per_px = max((chip.maxx - chip.minx) * (chip.maxy - chip.miny) / max(chip.width_px * chip.height_px, 1), 0.0)
    notes: list[str] = []

    poly_px = lonlat_polyline_to_pixels(parcel_polygon_lonlat, chip) if parcel_polygon_lonlat else None
    parcel_mask = mask_from_polygon_px(poly_px, (h, w))
    if parcel_mask is None:
        return PrivateParkingArea(
            parcel_mask=None,
            usable_mask=None,
            building_mask=None,
            road_mask=None,
            source="missing_parcel",
            notes=["parcelle_absente_ou_non_rasterisable"],
        )

    building_mask = None
    road_mask = None
    hardstand = surface.asphalt_mask.astype(bool)
    generic_eligible = surface.parking_eligible_mask.astype(bool)
    candidate = hardstand & parcel_mask.astype(bool)
    generic_private_m2 = float((generic_eligible & parcel_mask.astype(bool)).sum()) * m2_per_px
    hardstand_private_m2 = float(candidate.sum()) * m2_per_px
    if generic_private_m2 > max(250.0, hardstand_private_m2 * 1.8):
        notes.append("surface_generique_non_bitumee_exclue")
    if fusion is not None:
        bmask = fusion.building_mask_hw
        if bmask is not None and bmask.shape == candidate.shape:
            building_mask = bmask.astype(bool)
            candidate &= ~building_mask
        rmask = fusion.road_mask_gis_hw
        if rmask is not None and rmask.shape == candidate.shape:
            road_mask = rmask.astype(bool)
            if _HAS_CV2 and road_mask.any():
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                road_mask = cv2.dilate(road_mask.astype(np.uint8), k, iterations=1).astype(bool)
            candidate &= ~road_mask
    if _HAS_CV2 and candidate.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)

    parcel_area = float(parcel_mask.sum()) * m2_per_px
    usable_area = float(candidate.sum()) * m2_per_px
    building_area = float((building_mask & parcel_mask).sum()) * m2_per_px if building_mask is not None else 0.0
    road_area = float((road_mask & parcel_mask).sum()) * m2_per_px if road_mask is not None else 0.0
    if usable_area < 60.0:
        notes.append("surface_privee_garable_trop_faible")

    return PrivateParkingArea(
        parcel_mask=parcel_mask,
        usable_mask=candidate,
        building_mask=building_mask,
        road_mask=road_mask,
        parcel_area_m2=parcel_area,
        usable_area_m2=usable_area,
        building_area_m2=building_area,
        road_area_m2=road_area,
        usable_ratio=usable_area / max(parcel_area, 1.0),
        source="cadastre_surface_gis",
        notes=notes,
    )
