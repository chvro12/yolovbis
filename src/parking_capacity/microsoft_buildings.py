"""Microsoft Building Footprints : import local GeoJSON + requête par bbox (index spatial)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from shapely.geometry import box, mapping, shape

from parking_capacity.gis_rasterize import chip_to_bbox4326, rasterize_polygons_on_chip
from parking_capacity.imagery_wms import OrthoChip

logger = logging.getLogger(__name__)

_MAX_BYTES_DEFAULT = 120_000_000


def _iter_features(path: Path) -> Iterator[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        for f in data.get("features") or []:
            if isinstance(f, dict):
                yield f
    elif data.get("type") == "Feature":
        yield data


def load_building_geometries_for_bbox(
    source: Path,
    bbox4326: tuple[float, float, float, float],
    *,
    max_bytes: int = _MAX_BYTES_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Charge les géométries GeoJSON dont la bbox intersecte ``bbox4326``.

    ``source`` : fichier .geojson ou répertoire (premier ``*.geojson`` trouvé, non récursif).
    """
    p = Path(source)
    if p.is_dir():
        cands = sorted(p.glob("*.geojson"))
        if not cands:
            cands = sorted(p.glob("*.json"))
        if not cands:
            logger.warning("Aucun GeoJSON dans %s", p)
            return []
        p = cands[0]
    if not p.is_file():
        return []
    if p.stat().st_size > max_bytes:
        logger.warning(
            "Fichier trop volumineux pour chargement en mémoire (%s o > %s). "
            "Fournissez un extrait régional découpé.",
            p.stat().st_size,
            max_bytes,
        )
        return []

    min_lon, min_lat, max_lon, max_lat = bbox4326
    aoi = box(min_lon, min_lat, max_lon, max_lat)
    geoms: List[Any] = []
    for feat in _iter_features(p):
        g = feat.get("geometry")
        if not g:
            continue
        try:
            geom = shape(g)
        except Exception:
            continue
        if geom.is_empty or not geom.intersects(aoi):
            continue
        geoms.append(mapping(geom.intersection(aoi)))
    return geoms


def query_buildings_mask_for_chip(chip: OrthoChip, source: Path) -> Optional[np.ndarray]:
    bbox = chip_to_bbox4326(chip)
    geoms = load_building_geometries_for_bbox(source, bbox)
    if not geoms:
        return None
    return rasterize_polygons_on_chip(chip, geoms)
