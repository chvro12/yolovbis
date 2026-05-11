"""Agrégation géométrique des parkings OSM et somme des capacités."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from shapely import make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from parking_capacity.geometry import intersection_area_m2, to_metric
from parking_capacity.overpass import OsmParkingElement


def _parse_capacity(tags: dict[str, str]) -> int | None:
    for key in ("capacity", "capacity:disabled", "capacity:parent"):
        v = tags.get(key)
        if not v:
            continue
        try:
            return int(float(v))
        except ValueError:
            continue
    return None


def element_to_polygon(el: OsmParkingElement) -> Polygon | None:
    ring = el.geometry_lonlat
    if len(ring) < 4:
        return None
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or not isinstance(poly, Polygon):
            # make_valid peut retourner GeometryCollection
            if hasattr(poly, "geoms"):
                polys = [g for g in poly.geoms if isinstance(g, Polygon)]
                if not polys:
                    return None
                poly = max(polys, key=lambda p: p.area)
            else:
                return None
        return poly
    except Exception:
        return None


@dataclass
class ParkingClassification:
    element: OsmParkingElement
    polygon: Polygon
    capacity: int | None
    on_parcel: bool
    in_buffer_only: bool


def classify_parkings(
    elements: Iterable[OsmParkingElement],
    parcel_union: BaseGeometry | None,
    point_buffer: BaseGeometry,
    *,
    min_intersection_m2: float = 25.0,
) -> list[ParkingClassification]:
    """
    on_parcel : intersection parcelle >= min_intersection_m2
    in_buffer_only : intersecte le buffer point mais pas assez la parcelle (ou pas de parcelle)
    """
    out: list[ParkingClassification] = []
    for el in elements:
        poly = element_to_polygon(el)
        if poly is None or poly.is_empty:
            continue
        cap = _parse_capacity(el.tags)

        on_parcel = False
        if parcel_union is not None and not parcel_union.is_empty:
            try:
                ia = intersection_area_m2(poly, parcel_union)
                on_parcel = ia >= min_intersection_m2
            except Exception:
                on_parcel = False

        in_buf = False
        try:
            if intersection_area_m2(poly, point_buffer) >= min_intersection_m2:
                in_buf = True
        except Exception:
            in_buf = False

        in_buffer_only = in_buf and not on_parcel
        if not on_parcel and not in_buf:
            continue

        out.append(
            ParkingClassification(
                element=el,
                polygon=poly,
                capacity=cap,
                on_parcel=on_parcel,
                in_buffer_only=in_buffer_only,
            )
        )
    return out


def sum_capacity(rows: list[ParkingClassification], *, on_parcel: bool | None) -> tuple[int, int]:
    """
    Retourne (somme_capacity, nb_avec_tag).
    Si on_parcel est True : seulement sur parcelle.
    Si False : seulement buffer_only.
    Si None : tous les rows passés.
    """
    s = 0
    n = 0
    for r in rows:
        if on_parcel is True and not r.on_parcel:
            continue
        if on_parcel is False and not r.in_buffer_only:
            continue
        if on_parcel is None:
            pass
        if r.capacity is None:
            continue
        s += r.capacity
        n += 1
    return s, n


def classification_polygon_area_m2(c: ParkingClassification) -> float:
    """Surface du polygone parking en m²."""
    return float(to_metric(c.polygon).area)


def surface_capacity_range_for_untagged(
    rows: list[ParkingClassification],
    *,
    on_parcel: bool | None,
    m2_per_space_mid: float = 28.0,
    m2_per_space_min: float = 25.0,
    m2_per_space_max: float = 32.0,
) -> tuple[int, int, int, float]:
    """
    Estime places (mid, min, max) à partir de la surface des parkings **sans** tag capacity.
    Retourne (mid, min_places, max_places, total_area_m2).
    """
    total_a = 0.0
    for r in rows:
        if on_parcel is True and not r.on_parcel:
            continue
        if on_parcel is False and not r.in_buffer_only:
            continue
        if on_parcel is None and not (r.on_parcel or r.in_buffer_only):
            continue
        if r.capacity is not None:
            continue
        total_a += classification_polygon_area_m2(r)
    if total_a <= 0:
        return 0, 0, 0, 0.0
    mid = int(max(0, round(total_a / m2_per_space_mid)))
    mn = int(max(0, round(total_a / m2_per_space_max)))
    mx = int(max(0, round(total_a / m2_per_space_min)))
    return mid, mn, mx, total_a


def count_polygons(rows: list[ParkingClassification], *, on_parcel: bool | None) -> int:
    c = 0
    for r in rows:
        if on_parcel is True and not r.on_parcel:
            continue
        if on_parcel is False and not r.in_buffer_only:
            continue
        c += 1
    return c
