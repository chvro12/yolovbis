"""Géocodage via l'API Adresse (Base Adresse Nationale)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

BAN_SEARCH_URL = "https://api-adresse.data.gouv.fr/search/"


@dataclass
class GeocodeResult:
    label: str
    lon: float
    lat: float
    score: float
    raw: dict[str, Any]


class GeocodeError(RuntimeError):
    pass


def geocode_address(
    address: str,
    *,
    client: httpx.Client | None = None,
    limit: int = 1,
) -> GeocodeResult:
    """Retourne le meilleur résultat BAN pour une chaîne d'adresse."""
    params = {"q": address.strip(), "limit": limit}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=30.0)

    try:
        r = client.get(BAN_SEARCH_URL, params=params)
        r.raise_for_status()
        data = r.json()
    finally:
        if own_client:
            client.close()

    feats = data.get("features") or []
    if not feats:
        raise GeocodeError(f"Aucun résultat BAN pour : {address!r}")

    f0 = feats[0]
    geom = f0.get("geometry") or {}
    coords = geom.get("coordinates")
    if not coords or len(coords) < 2:
        raise GeocodeError("Réponse BAN sans coordonnées valides")

    props = f0.get("properties") or {}
    return GeocodeResult(
        label=str(props.get("label") or address),
        lon=float(coords[0]),
        lat=float(coords[1]),
        score=float(props.get("score") or 0.0),
        raw=f0,
    )
