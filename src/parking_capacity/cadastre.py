"""Parcelles cadastrales via APICarto IGN.

L’API Carto (`https://apicarto.ign.fr`) expose plusieurs modules (cadastre, urbanisme,
risques, etc.). Ce module n’utilise que **parcelle** (polygone cadastral au point).
Les autres couches (PLU, zones inondables, …) ont des schémas et limites propres :
voir https://apicarto.ign.fr/api/doc et ``docs/gis_providers.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import httpx
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from parking_capacity.cache_http import get_with_cache

APICARTO_PARCELLE_URL = "https://apicarto.ign.fr/api/cadastre/parcelle"

# Décalages (m) : le point BAN peut tomber sur une limite / hors parcelle alors qu’un voisin proche intersecte.
# Profil par défaut allégé pour le bulk : 3 anneaux au lieu de 6 → ~19 appels max au lieu de ~46 dans le pire cas.
_DEFAULT_JITTER_M: Tuple[float, ...] = (0.0, 5.0, 15.0)


@dataclass
class ParcelleHit:
    """Une parcelle avec identifiants et géométrie Shapely (WGS84)."""

    geometry: BaseGeometry
    id_parcelle: str | None
    section: str | None
    numero: str | None
    commune: str | None
    raw: dict[str, Any]


def iter_parcel_query_points(
    lon: float,
    lat: float,
    *,
    jitter_m: Tuple[float, ...] = _DEFAULT_JITTER_M,
) -> Iterator[Tuple[float, float]]:
    """Centre puis petits déplacements en mètres (approximation WGS84) pour retrouver une parcelle."""
    cos_lat = math.cos(math.radians(lat))
    scale_n = 1.0 / 111_320.0
    scale_e = 1.0 / (111_320.0 * max(cos_lat, 0.2))
    seen: set[Tuple[float, float]] = set()
    for d in jitter_m:
        dn = d * scale_n
        de = d * scale_e
        candidates = [
            (lon, lat),
            (lon + de, lat),
            (lon - de, lat),
            (lon, lat + dn),
            (lon, lat - dn),
            (lon + de * 0.70710678, lat + dn * 0.70710678),
            (lon - de * 0.70710678, lat + dn * 0.70710678),
            (lon + de * 0.70710678, lat - dn * 0.70710678),
            (lon - de * 0.70710678, lat - dn * 0.70710678),
        ]
        for lo, la in candidates:
            key = (round(lo, 8), round(la, 8))
            if key in seen:
                continue
            seen.add(key)
            yield (lo, la)


def _parse_feature_collection(fc: Any) -> List[ParcelleHit]:
    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        return []
    hits: list[ParcelleHit] = []
    for feat in fc.get("features") or []:
        g = feat.get("geometry")
        if not g:
            continue
        shp = shape(g)
        if shp.is_empty:
            continue
        props = feat.get("properties") or {}
        pid = props.get("id") or props.get("uid") or props.get("idu")
        hits.append(
            ParcelleHit(
                geometry=shp,
                id_parcelle=str(pid) if pid is not None else None,
                section=props.get("section"),
                numero=props.get("numero"),
                commune=props.get("commune") or props.get("insee"),
                raw=feat,
            )
        )
    return hits


def _fetch_parcelles_at_point(
    lon: float,
    lat: float,
    *,
    client: httpx.Client,
    cache_dir: Optional[Path] = None,
) -> list[ParcelleHit]:
    geom = {"type": "Point", "coordinates": [lon, lat]}
    params = {"geom": json.dumps(geom, separators=(",", ":"))}
    r = get_with_cache(client, APICARTO_PARCELLE_URL, params=params, cache_dir=cache_dir)
    return _parse_feature_collection(r.json())


def fetch_parcelles(
    lon: float,
    lat: float,
    *,
    client: httpx.Client | None = None,
    jitter_m: Iterable[float] | None = None,
    cache_dir: Optional[Path] = None,
) -> list[ParcelleHit]:
    """
    Interroge APICarto parcelle pour un point (lon, lat) en WGS84.

    Si la réponse est vide (point sur limite, décalage BAN, etc.), réessaie avec de légers
    décalages en mètres autour du point. ``cache_dir`` active le cache disque GET (utile en bulk).
    """
    jit = tuple(jitter_m) if jitter_m is not None else _DEFAULT_JITTER_M

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=30.0)
    try:
        for lo, la in iter_parcel_query_points(lon, lat, jitter_m=jit):
            hits = _fetch_parcelles_at_point(lo, la, client=client, cache_dir=cache_dir)
            if hits:
                return hits
        return []
    finally:
        if own_client:
            client.close()


def merge_parcelles_geometries(hits: list[ParcelleHit]):
    """Union des géométries parcelle (WGS84)."""
    from shapely.ops import unary_union

    if not hits:
        return None
    return unary_union([h.geometry for h in hits])
