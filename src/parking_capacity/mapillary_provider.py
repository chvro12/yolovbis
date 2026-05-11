"""Mapillary Graph API (optionnel) : images dans une bbox."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def mapillary_images_in_bbox(
    client: httpx.Client,
    bbox4326: tuple[float, float, float, float],
    *,
    access_token: str,
    graph_base: str = "https://graph.mapillary.com",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Liste d'images (champs réduits) intersectant la bbox ``min_lon,min_lat,max_lon,max_lat``.

    Nécessite un jeton avec les portées Graph API Mapillary.
    """
    min_lon, min_lat, max_lon, max_lat = bbox4326
    fields = "id,geometry,captured_at"
    url = (
        f"{graph_base.rstrip('/')}/images"
        f"?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&fields={fields}&limit={limit}&access_token={access_token}"
    )
    r = client.get(url, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    return list(data.get("data") or [])


def mapillary_ping(client: httpx.Client, token: str) -> bool:
    try:
        # bbox minuscule Paris
        mapillary_images_in_bbox(client, (2.33, 48.85, 2.331, 48.851), access_token=token, limit=1)
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("mapillary_ping: %s", e)
        return False
