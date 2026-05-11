"""Requêtes Overpass pour les parkings OSM."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from parking_capacity.cache_http import post_with_cache

OVERPASS_DEFAULT_URL = "https://overpass-api.de/api/interpreter"
logger = logging.getLogger(__name__)


@dataclass
class OsmParkingElement:
    osm_type: str  # way | relation
    osm_id: int
    tags: dict[str, str]
    geometry_lonlat: list[tuple[float, float]]  # (lon, lat) rings ou lignes fermées


class OverpassError(RuntimeError):
    pass


def _ring_closed(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return []
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return coords


def _coords_from_geometry(geom: list[dict[str, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for n in geom:
        lat = float(n.get("lat"))
        lon = float(n.get("lon"))
        out.append((lon, lat))
    return out


def parse_overpass_elements(data: dict[str, Any]) -> list[OsmParkingElement]:
    elements = data.get("elements") or []
    result: list[OsmParkingElement] = []

    for el in elements:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        if tags.get("amenity") != "parking":
            continue

        et = el.get("type")
        eid = int(el.get("id", 0))

        if et == "way":
            geom = el.get("geometry")
            if not geom:
                continue
            ring = _ring_closed(_coords_from_geometry(geom))
            if len(ring) < 4:
                continue
            result.append(OsmParkingElement("way", eid, tags, ring))

        elif et == "relation":
            # Overpass avec 'out body geom' peut fournir des membres avec géométrie
            geom = el.get("geometry")
            if geom:
                ring = _ring_closed(_coords_from_geometry(geom))
                if len(ring) >= 4:
                    result.append(OsmParkingElement("relation", eid, tags, ring))
                    continue
            members = el.get("members") or []
            rings: list[list[tuple[float, float]]] = []
            for m in members:
                if m.get("type") != "way":
                    continue
                mg = m.get("geometry")
                if not mg:
                    continue
                r = _ring_closed(_coords_from_geometry(mg))
                if len(r) >= 4:
                    rings.append(r)
            if not rings:
                continue
            # Garde le plus grand anneau extérieur (approximation multipolygone)
            best = max(rings, key=lambda r: len(r))
            result.append(OsmParkingElement("relation", eid, tags, best))

    return result


def build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # around: utilise des mètres pour le rayon
    return f"""[out:json][timeout:90];
(
  way["amenity"="parking"](around:{radius_m},{lat},{lon});
  relation["amenity"="parking"](around:{radius_m},{lat},{lon});
);
out body geom;
"""


def build_overpass_bbox_parking_capacity(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> str:
    """BBox Overpass : (sud, ouest, nord, est) = (min_lat, min_lon, max_lat, max_lon)."""
    south, west, north, east = min_lat, min_lon, max_lat, max_lon
    return f"""[out:json][timeout:180];
(
  way["amenity"="parking"]["capacity"]({south},{west},{north},{east});
  relation["amenity"="parking"]["capacity"]({south},{west},{north},{east});
);
out body geom;
"""


def build_overpass_parking_space_count(lat: float, lon: float, radius_m: int) -> str:
    # « out skel » ne renvoie pas les tags : le comptage Python sur tags amenity échouait toujours.
    # « out tags » suffit pour compter les micro-cartographies parking_space (cf. OSM web).
    return f"""[out:json][timeout:90];
(
  node["amenity"="parking_space"](around:{radius_m},{lat},{lon});
  way["amenity"="parking_space"](around:{radius_m},{lat},{lon});
  relation["amenity"="parking_space"](around:{radius_m},{lat},{lon});
);
out tags;
"""


def _post_overpass(
    base_url: str,
    query: str,
    client: httpx.Client,
    *,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    r = post_with_cache(
        client,
        base_url,
        query,
        cache_dir=cache_dir,
        max_retries=max_retries,
    )
    return r.json()


def query_parkings(
    lat: float,
    lon: float,
    *,
    radius_m: int = 120,
    base_url: str = OVERPASS_DEFAULT_URL,
    client: httpx.Client | None = None,
    delay_s: float = 0.0,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> tuple[list[OsmParkingElement], dict[str, Any]]:
    if delay_s > 0:
        time.sleep(delay_s)
    q = build_overpass_query(lat, lon, radius_m)
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=120.0)
    try:
        data = _post_overpass(base_url, q, client, cache_dir=cache_dir, max_retries=max_retries)
    except httpx.HTTPError as e:
        raise OverpassError(str(e)) from e
    finally:
        if own_client:
            client.close()

    if "remark" in data and not data.get("elements"):
        raise OverpassError(f"Overpass remark: {data.get('remark')}")

    elems = parse_overpass_elements(data)
    return elems, data


def query_parking_space_count(
    lat: float,
    lon: float,
    *,
    radius_m: int = 50,
    base_url: str = OVERPASS_DEFAULT_URL,
    client: httpx.Client | None = None,
    delay_s: float = 0.0,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> tuple[int, dict[str, Any]]:
    if delay_s > 0:
        time.sleep(delay_s)
    q = build_overpass_parking_space_count(lat, lon, radius_m)
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=120.0)
    try:
        data = _post_overpass(base_url, q, client, cache_dir=cache_dir, max_retries=max_retries)
    except httpx.HTTPError as e:
        raise OverpassError(str(e)) from e
    finally:
        if own_client:
            client.close()
    n = 0
    for el in data.get("elements") or []:
        tags = el.get("tags") or {}
        if tags.get("amenity") == "parking_space":
            n += 1
        elif not tags and el.get("type") in ("node", "way", "relation"):
            # Anciennes réponses « out skel » (sans tags) : le filtre Overpass garantit amenity=parking_space.
            n += 1
    return n, data


def query_parkings_bbox_capacity(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    base_url: str = OVERPASS_DEFAULT_URL,
    client: httpx.Client | None = None,
    delay_s: float = 0.0,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> tuple[list[OsmParkingElement], dict[str, Any]]:
    """Parkings avec tag capacity dans une bbox WGS84 (min_lon, min_lat, max_lon, max_lat)."""
    if delay_s > 0:
        time.sleep(delay_s)
    q = build_overpass_bbox_parking_capacity(min_lon, min_lat, max_lon, max_lat)
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=120.0)
    try:
        data = _post_overpass(base_url, q, client, cache_dir=cache_dir, max_retries=max_retries)
    except httpx.HTTPError as e:
        raise OverpassError(str(e)) from e
    finally:
        if own_client:
            client.close()
    if "remark" in data and not data.get("elements"):
        logger.warning("Overpass bbox remark: %s", data.get("remark"))
    elems = parse_overpass_elements(data)
    elems = [e for e in elems if _capacity_from_tags(e.tags) is not None]
    return elems, data


def _capacity_from_tags(tags: dict[str, str]) -> int | None:
    for key in ("capacity", "capacity:disabled", "capacity:parent"):
        v = tags.get(key)
        if not v:
            continue
        try:
            c = int(float(v))
            if c > 0:
                return c
        except ValueError:
            continue
    return None
