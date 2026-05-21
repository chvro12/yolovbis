"""Géocodage via l'API Adresse (Base Adresse Nationale)."""

from __future__ import annotations

import csv
import io
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import httpx

BAN_SEARCH_URL = "https://api-adresse.data.gouv.fr/search/"
BAN_SEARCH_CSV_URL = "https://api-adresse.data.gouv.fr/search/csv/"


@dataclass
class GeocodeResult:
    label: str
    lon: float
    lat: float
    score: float
    raw: dict[str, Any]


class GeocodeError(RuntimeError):
    pass


# Cache mémoire partagé (clef = adresse normalisée). Prewarm via batch_geocode_csv en bulk.
_GEOCODE_CACHE: Dict[str, GeocodeResult] = {}
_GEOCODE_CACHE_LOCK = threading.Lock()


def _normalize(address: str) -> str:
    return " ".join(address.strip().split())


def prime_geocode_cache(items: Dict[str, GeocodeResult]) -> None:
    with _GEOCODE_CACHE_LOCK:
        for k, v in items.items():
            _GEOCODE_CACHE[_normalize(k)] = v


def clear_geocode_cache() -> None:
    with _GEOCODE_CACHE_LOCK:
        _GEOCODE_CACHE.clear()


def geocode_address(
    address: str,
    *,
    client: httpx.Client | None = None,
    limit: int = 1,
) -> GeocodeResult:
    """Retourne le meilleur résultat BAN pour une chaîne d'adresse."""
    key = _normalize(address)
    cached = _GEOCODE_CACHE.get(key)
    if cached is not None:
        return cached

    params = {"q": key, "limit": limit}
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
    result = GeocodeResult(
        label=str(props.get("label") or address),
        lon=float(coords[0]),
        lat=float(coords[1]),
        score=float(props.get("score") or 0.0),
        raw=f0,
    )
    with _GEOCODE_CACHE_LOCK:
        _GEOCODE_CACHE[key] = result
    return result


def batch_geocode_csv(
    addresses: Iterable[str],
    *,
    client: httpx.Client | None = None,
    chunk_size: int = 500,
) -> Dict[str, GeocodeResult]:
    """Géocode en masse via POST /search/csv/ (1 appel = jusqu'à `chunk_size` adresses).

    Renvoie un dict ``adresse normalisée → GeocodeResult`` pour les lignes géocodées avec succès.
    Les adresses sans résultat (score bas ou erreur API) sont simplement omises ; le pipeline
    retombera sur ``geocode_address`` (qui requêtera /search/ unitairement) pour celles-là.
    """
    uniq: List[str] = []
    seen: set[str] = set()
    for a in addresses:
        k = _normalize(a)
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    if not uniq:
        return {}

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=120.0, follow_redirects=True)

    out: Dict[str, GeocodeResult] = {}
    try:
        for start in range(0, len(uniq), chunk_size):
            batch = uniq[start : start + chunk_size]
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["adresse"])
            for a in batch:
                w.writerow([a])
            files = {"data": ("addresses.csv", buf.getvalue().encode("utf-8"), "text/csv")}
            data = {"columns": "adresse"}
            r = client.post(BAN_SEARCH_CSV_URL, data=data, files=files)
            r.raise_for_status()
            reader = csv.DictReader(io.StringIO(r.text))
            for row in reader:
                addr = row.get("adresse") or ""
                key = _normalize(addr)
                lon_s = row.get("longitude") or row.get("result_longitude")
                lat_s = row.get("latitude") or row.get("result_latitude")
                score_s = row.get("result_score")
                label = row.get("result_label") or addr
                if not key or not lon_s or not lat_s:
                    continue
                try:
                    lon = float(lon_s)
                    lat = float(lat_s)
                    score = float(score_s) if score_s else 0.0
                except (TypeError, ValueError):
                    continue
                out[key] = GeocodeResult(
                    label=str(label),
                    lon=lon,
                    lat=lat,
                    score=score,
                    raw={"_source": "ban_csv_batch", **{k: v for k, v in row.items() if k}},
                )
    finally:
        if own_client:
            client.close()

    return out
