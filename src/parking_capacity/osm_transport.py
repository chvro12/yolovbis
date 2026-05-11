"""Overpass : transport, accès, parkings détaillés (complément ``overpass.py``)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from parking_capacity.cache_http import post_with_cache

logger = logging.getLogger(__name__)


def build_osm_transport_query(lat: float, lon: float, radius_m: int) -> str:
    """
    Requête union autour du point : voirie, accès parking, parkings surfaciques, bâtiments.

    Références tags : ``service=parking_aisle``, ``highway=service``, ``amenity=parking*``.
    """
    r = int(radius_m)
    # timeout élevé : unions nombreuses
    return f"""[out:json][timeout:120];
(
  way["amenity"="parking"](around:{r},{lat},{lon});
  relation["amenity"="parking"](around:{r},{lat},{lon});
  way["amenity"="parking_space"](around:{r},{lat},{lon});
  node["amenity"="parking_space"](around:{r},{lat},{lon});
  way["amenity"="parking_entrance"](around:{r},{lat},{lon});
  node["amenity"="parking_entrance"](around:{r},{lat},{lon});
  way["highway"]["service"="parking_aisle"](around:{r},{lat},{lon});
  way["highway"="service"]["service"="driveway"](around:{r},{lat},{lon});
  way["highway"="service"](around:{r},{lat},{lon});
  way["highway"="residential"](around:{r},{lat},{lon});
  way["highway"="living_street"](around:{r},{lat},{lon});
  way["highway"="unclassified"](around:{r},{lat},{lon});
  way["highway"="tertiary"](around:{r},{lat},{lon});
  way["highway"="secondary"](around:{r},{lat},{lon});
  way["highway"="primary"](around:{r},{lat},{lon});
  way["highway"="trunk"](around:{r},{lat},{lon});
  way["highway"]["access"](around:{r},{lat},{lon});
  way["building"](around:{r},{lat},{lon});
  way["parking"="surface"](around:{r},{lat},{lon});
  way["parking"="street_side"](around:{r},{lat},{lon});
);
out body geom;
"""


@dataclass
class TransportSummary:
    n_highway_ways: int = 0
    n_parking_aisle: int = 0
    n_service: int = 0
    n_residential_tertiary_plus: int = 0
    n_parking_amenity: int = 0
    n_building_ways: int = 0
    named_highways_sample: List[str] = field(default_factory=list)


@dataclass
class TransportFetchResult:
    elements: List[Dict[str, Any]]
    raw: Dict[str, Any]
    summary: TransportSummary
    line_geoms: List[Any] = field(default_factory=list)  # Shapely LineString-ready coords
    building_geoms: List[Any] = field(default_factory=list)


def _coords_from_geometry(geom: List[Dict[str, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for n in geom or []:
        lat = float(n.get("lat"))
        lon = float(n.get("lon"))
        out.append((lon, lat))
    return out


def _summarize_and_collect(data: Dict[str, Any]) -> TransportFetchResult:
    elems = [e for e in (data.get("elements") or []) if isinstance(e, dict)]
    summary = TransportSummary()
    lines: List[Any] = []
    buildings: List[Any] = []
    names_seen: set[str] = set()

    for el in elems:
        tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
        et = el.get("type")
        hw = tags.get("highway")
        amen = tags.get("amenity")
        building = tags.get("building")

        if hw:
            summary.n_highway_ways += 1
            if tags.get("service") == "parking_aisle":
                summary.n_parking_aisle += 1
            if hw == "service":
                summary.n_service += 1
            if hw in ("residential", "living_street", "unclassified", "tertiary", "secondary", "primary", "trunk"):
                summary.n_residential_tertiary_plus += 1
            name = tags.get("name")
            if name and name not in names_seen and len(summary.named_highways_sample) < 12:
                names_seen.add(name)
                summary.named_highways_sample.append(name)

        if amen in ("parking", "parking_space", "parking_entrance"):
            summary.n_parking_amenity += 1

        if building and building.lower() not in ("no", "false"):
            summary.n_building_ways += 1

        geom = el.get("geometry")
        if not geom:
            continue
        coords = _coords_from_geometry(geom)
        if len(coords) < 2:
            continue
        if et == "way" and hw:
            lines.append({"type": "LineString", "coordinates": [[c[0], c[1]] for c in coords]})
        elif et == "way" and building:
            # bâtiment : contour fermé
            if len(coords) >= 3:
                ring = list(coords)
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                buildings.append({"type": "Polygon", "coordinates": [ring]})

    return TransportFetchResult(elements=elems, raw=data, summary=summary, line_geoms=lines, building_geoms=buildings)


def query_transport_around(
    lat: float,
    lon: float,
    *,
    radius_m: int = 80,
    base_url: str,
    client: httpx.Client,
    delay_s: float = 0.0,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> TransportFetchResult:
    if delay_s > 0:
        time.sleep(delay_s)
    q = build_osm_transport_query(lat, lon, radius_m)
    r = post_with_cache(client, base_url, q, cache_dir=cache_dir, max_retries=max_retries)
    data = r.json()
    if "remark" in data and not data.get("elements"):
        logger.warning("Overpass transport remark: %s", data.get("remark"))
    return _summarize_and_collect(data)


def chip_intersects_any_named_highway(
    chip_bounds_lonlat: Tuple[float, float, float, float],
    named_samples: List[str],
) -> bool:
    """Heuristique : si le nom d'une voie OSM contient un token du buffer d'adresse (non utilisé ici)."""
    # réservé à des extensions (matching BAN street name)
    return False
