"""Rasterisation de géométries WGS84 sur une puce orthophoto (bbox Web Mercator)."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, shape

from parking_capacity.imagery_wms import OrthoChip

_to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def lonlat_to_pixel_xy(chip: OrthoChip, lon: float, lat: float) -> Tuple[float, float]:
    """Centre image : origine haut-gauche, y vers le bas (PIL)."""
    mx, my = _to3857.transform(lon, lat)
    w, h = float(chip.width_px), float(chip.height_px)
    px = (mx - chip.minx) / max(chip.maxx - chip.minx, 1e-9) * w
    py = h - (my - chip.miny) / max(chip.maxy - chip.miny, 1e-9) * h
    return float(px), float(py)


def _coords_to_pixels(chip: OrthoChip, coords: Sequence[Sequence[float]]) -> np.ndarray:
    """Nx2 float32 en pixels pour OpenCV."""
    pts: List[Tuple[float, float]] = []
    for c in coords:
        if len(c) < 2:
            continue
        lon, lat = float(c[0]), float(c[1])
        px, py = lonlat_to_pixel_xy(chip, lon, lat)
        pts.append((px, py))
    if len(pts) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def rasterize_polygons_on_chip(
    chip: OrthoChip,
    polygons_wgs84: Iterable[Any],
    *,
    line_width_px: int = 0,
) -> np.ndarray:
    """Masque bool H×W : remplissage polygones (Shapely ou GeoJSON-like dict)."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return np.zeros((chip.height_px, chip.width_px), dtype=bool)

    h, w = chip.height_px, chip.width_px
    mask = np.zeros((h, w), dtype=np.uint8)
    for g in polygons_wgs84:
        geom = shape(g) if isinstance(g, dict) else g
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue
        for poly in polys:
            ext = np.array(
                _coords_to_pixels(chip, poly.exterior.coords),
                dtype=np.int32,
            )
            if ext.shape[0] >= 3:
                cv2.fillPoly(mask, [ext], 1)
            for inter in poly.interiors:
                inn = np.array(_coords_to_pixels(chip, inter.coords), dtype=np.int32)
                if inn.shape[0] >= 3:
                    cv2.fillPoly(mask, [inn], 0)
    if line_width_px > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(1, line_width_px // 2))
    return mask.astype(bool)


def rasterize_lines_on_chip(
    chip: OrthoChip,
    lines_wgs84: Iterable[Any],
    *,
    thickness_px: int = 6,
) -> np.ndarray:
    """Masque bool H×W : lignes épaissies (routes OSM / tronçons)."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return np.zeros((chip.height_px, chip.width_px), dtype=bool)

    h, w = chip.height_px, chip.width_px
    mask = np.zeros((h, w), dtype=np.uint8)
    t = max(1, int(thickness_px))
    for g in lines_wgs84:
        if isinstance(g, dict):
            geom = shape(g)
        elif isinstance(g, LineString):
            geom = g
        else:
            continue
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for ln in lines:
            arr = _coords_to_pixels(chip, ln.coords)
            if arr.shape[0] < 2:
                continue
            cv2.polylines(mask, [arr.astype(np.int32)], isClosed=False, color=1, thickness=t)
    return mask.astype(bool)


def geojson_feature_to_geom(feat: dict[str, Any]) -> Any:
    """Extrait une géométrie Shapely depuis un Feature GeoJSON."""
    g = feat.get("geometry")
    if not g:
        return None
    try:
        return shape(g)
    except Exception:
        return None


def point_in_chip_fraction(chip: OrthoChip, lon: float, lat: float) -> Tuple[float, float]:
    """Position normalisée (0-1) du point dans la puce ; peut dépasser si hors cadre."""
    px, py = lonlat_to_pixel_xy(chip, lon, lat)
    return px / max(chip.width_px, 1), py / max(chip.height_px, 1)


def chip_to_bbox4326(chip: OrthoChip) -> Tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) pour la bbox de la puce orthophoto."""
    from pyproj import Transformer

    to4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    corners = (
        (chip.minx, chip.miny),
        (chip.maxx, chip.miny),
        (chip.maxx, chip.maxy),
        (chip.minx, chip.maxy),
    )
    lons: List[float] = []
    lats: List[float] = []
    for mx, my in corners:
        lo, la = to4326.transform(mx, my)
        lons.append(lo)
        lats.append(la)
    pad = 1e-5
    return min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad
