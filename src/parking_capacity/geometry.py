"""Projections, buffers, boîtes englobantes et tests d'intersection."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

# Web Mercator pour buffers et surfaces en m² (approximation acceptable sur petites emprises).
CRS_METRIC = "EPSG:3857"
CRS_WGS84 = "EPSG:4326"

_to_metric = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True)
_to_wgs = Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True)


def to_metric(geom: BaseGeometry) -> BaseGeometry:
    return transform(_to_metric.transform, geom)


def to_wgs84(geom: BaseGeometry) -> BaseGeometry:
    return transform(_to_wgs.transform, geom)


def buffer_point_m(lon: float, lat: float, radius_m: float) -> BaseGeometry:
    """Polygone buffer autour du point, en WGS84."""
    pt = Point(lon, lat)
    return to_wgs84(to_metric(pt).buffer(radius_m))


def bbox_around_point_m(lon: float, lat: float, half_side_m: float) -> tuple[float, float, float, float]:
    """
    BBOX en EPSG:3857 (minx, miny, maxx, maxy) centrée sur le point,
    carré de côté 2 * half_side_m.
    """
    mx, my = _to_metric.transform(lon, lat)
    return (
        mx - half_side_m,
        my - half_side_m,
        mx + half_side_m,
        my + half_side_m,
    )


def intersection_area_m2(a: BaseGeometry, b: BaseGeometry) -> float:
    """Aire d'intersection en m² (géométries WGS84 en entrée)."""
    ai = to_metric(a)
    bi = to_metric(b)
    inter = ai.intersection(bi)
    return float(inter.area)


def polygon_area_m2_lonlat_ring(ring: list[tuple[float, float]]) -> float:
    """Aire d'un polygone (lon, lat) en m² via projection Web Mercator."""
    if len(ring) < 4:
        return 0.0
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            from shapely import make_valid

            poly = make_valid(poly)
        if poly.is_empty:
            return 0.0
        if not isinstance(poly, Polygon):
            polys = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon)]
            if not polys:
                return 0.0
            poly = max(polys, key=lambda p: p.area)
        return float(to_metric(poly).area)
    except Exception:
        return 0.0


def geojson_mapping(geom: BaseGeometry) -> dict:
    return mapping(geom)


@dataclass
class BBox4326:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


def bbox4326_around_point_deg(lon: float, lat: float, delta_deg: float) -> BBox4326:
    """Petite bbox degrés (usage Overpass / fallback)."""
    return BBox4326(
        min_lon=lon - delta_deg,
        min_lat=lat - delta_deg,
        max_lon=lon + delta_deg,
        max_lat=lat + delta_deg,
    )


def overpass_bbox_str(b: BBox4326) -> str:
    return f"{b.min_lat},{b.min_lon},{b.max_lat},{b.max_lon}"
