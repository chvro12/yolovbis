"""Client WFS Géoplateforme (BD TOPO v3) + cache GeoJSON local."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


def wfs_bbox_param(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    """BBOX WFS 2.0 pour le service IGN Géoplateforme en EPSG:4326.

    Le serveur attend **minLat, minLon, maxLat, maxLon** (sud, ouest, nord, est), pas lon,lat.
    Voir tests réels : ``min_lon,min_lat,...`` renvoie 0 entité alors que l'ordre lat,lon est correct.
    """
    return f"{min_lat},{min_lon},{max_lat},{max_lon},urn:ogc:def:crs:EPSG::4326"


def _cache_key(
    wfs_url: str,
    typename: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    max_count: int,
) -> str:
    s = f"{wfs_url}|{typename}|{min_lon:.6f}|{min_lat:.6f}|{max_lon:.6f}|{max_lat:.6f}|{max_count}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:28]


def wfs_get_feature_geojson(
    client: httpx.Client,
    wfs_base_url: str,
    typename: str,
    bbox4326: Tuple[float, float, float, float],
    *,
    max_features: int = 500,
    cache_dir: Optional[Path] = None,
    srsname: str = "EPSG:4326",
    output_format: str = "application/json",
) -> Dict[str, Any]:
    """
    GetFeature WFS 2.0.0 ; réponse attendue : GeoJSON FeatureCollection.

    ``bbox4326`` : (min_lon, min_lat, max_lon, max_lat).
    """
    min_lon, min_lat, max_lon, max_lat = bbox4326
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / "ign_wfs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(wfs_base_url, typename, min_lon, min_lat, max_lon, max_lat, max_features)
        safe_t = typename.replace(":", "_")
        cache_path = cache_dir / f"{safe_t}_{key}.geojson"
        if cache_path.is_file():
            logger.debug("WFS cache hit %s", cache_path.name)
            return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": typename,
        "COUNT": str(max_features),
        "SRSNAME": srsname,
        "BBOX": wfs_bbox_param(min_lon, min_lat, max_lon, max_lat),
        "outputFormat": output_format,
    }
    r = client.get(wfs_base_url, params=params, timeout=180.0)
    r.raise_for_status()
    data = r.json()
    if cache_path is not None:
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("WFS cache écrit %s", cache_path.name)
    return data


def geojson_feature_collection_to_features(fc: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = fc.get("features")
    if isinstance(feats, list):
        return [f for f in feats if isinstance(f, dict)]
    return []


def wfs_ping(client: httpx.Client, wfs_base_url: str, typename: str) -> bool:
    """Vérifie que le endpoint WFS répond (GetCapabilities léger)."""
    try:
        r = client.get(
            wfs_base_url,
            params={"SERVICE": "WFS", "REQUEST": "GetCapabilities", "VERSION": "2.0.0"},
            timeout=60.0,
        )
        r.raise_for_status()
        text = r.text[:5000] if r.text else ""
        return "WFS_Capabilities" in text or "wfs:WFS_Capabilities" in text
    except Exception as e:  # noqa: BLE001
        logger.warning("WFS ping échoué : %s", e)
        return False
