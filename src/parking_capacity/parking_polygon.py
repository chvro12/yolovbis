"""Masques → polygones parking (pixels), nettoyage, GeoJSON WGS84."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None  # type: ignore

try:
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover
    mapping = None  # type: ignore


def mask_preprocess(
    mask_bool: np.ndarray,
    *,
    close_kernel: int = 5,
    min_area_px: int = 80,
    hole_kernel: int = 0,
) -> np.ndarray:
    """Fermeture morphologique, suppression petites composantes, trous optionnels."""
    if cv2 is None:
        return mask_bool.astype(np.uint8) * 255
    m = (mask_bool.astype(np.uint8) * 255).copy()
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if hole_kernel > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_kernel, hole_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(m)
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area_px:
            cv2.drawContours(canvas, [c], -1, 255, thickness=cv2.FILLED)
    return canvas


def contours_to_polygons(
    mask_u8: np.ndarray,
    *,
    simplify_eps_px: float = 2.0,
) -> List[List[Tuple[float, float]]]:
    """Contours externes → polygones (simplification Douglas-Peucker)."""
    if cv2 is None:
        raise RuntimeError("opencv requis")
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[Tuple[float, float]]] = []
    for c in cnts:
        if len(c) < 3:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(simplify_eps_px, 0.001 * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        poly = approx.reshape(-1, 2).astype(np.float64)
        polys.append([(float(x), float(y)) for x, y in poly])
    return polys


def raster_polygon_to_wgs84_geojson(
    polygons_xy_px: Sequence[Sequence[Tuple[float, float]]],
    *,
    width_px: int,
    height_px: int,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    crs_xy: str = "EPSG:3857",
) -> Dict[str, Any]:
    """Convertit polygones pixel → WGS84 (polygone multipolygon GeoJSON)."""
    if mapping is None or Transformer is None:
        raise RuntimeError("shapely + pyproj requis pour GeoJSON géoréférencé")
    tf = Transformer.from_crs(crs_xy, "EPSG:4326", always_xy=True)

    def px_to_m(wx: float, wy_top: float) -> Tuple[float, float]:
        mx = minx + (wx / max(width_px, 1)) * (maxx - minx)
        my = maxy - (wy_top / max(height_px, 1)) * (maxy - miny)
        return mx, my

    geoms = []
    for poly in polygons_xy_px:
        if len(poly) < 3:
            continue
        ring_m = [px_to_m(x, y) for x, y in poly]
        lonlat = [tf.transform(mx, my) for mx, my in ring_m]
        # shapely Polygon expects lon lat
        from shapely.geometry import Polygon as ShPoly

        g = ShPoly(lonlat)
        if not g.is_valid:
            g = g.buffer(0)
        if getattr(g, "is_empty", False):
            continue
        geoms.append(g)
    if not geoms:
        return {"type": "FeatureCollection", "features": []}
    try:
        u = unary_union(geoms)
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(u), "properties": {}}]}
    except Exception:
        feats = [
            {"type": "Feature", "geometry": mapping(g), "properties": {"part_index": i}}
            for i, g in enumerate(geoms)
            if g is not None and not getattr(g, "is_empty", True)
        ]
        return {"type": "FeatureCollection", "features": feats}


def save_debug_polygon_extraction(
    rgb: Image.Image,
    mask_u8: np.ndarray,
    polys: Sequence[Sequence[Tuple[float, float]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.array(rgb.convert("RGB")).copy()
    if cv2 is not None:
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(base, [pts], True, (0, 255, 0), 2)
    Image.fromarray(base).save(path)
