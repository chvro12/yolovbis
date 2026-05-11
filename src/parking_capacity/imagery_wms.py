"""Téléchargement d'une puce orthophoto via WMS Géoplateforme + cache fichier."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
from PIL import Image

# Service WMS-Raster Géoplateforme (voir aide cartes.gouv.fr).
DEFAULT_WMS_BASE = "https://data.geopf.fr/wms-r"
DEFAULT_WMS_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO"

# Incrémenter si la logique de requête change (invalide les anciens fichiers).
WMS_CACHE_VERSION = "v1"

logger = logging.getLogger(__name__)


@dataclass
class OrthoChip:
    """Image RGB et géométrie de la bbox en Web Mercator (mètres)."""

    image: Image.Image
    minx: float
    miny: float
    maxx: float
    maxy: float
    width_px: int
    height_px: int
    layer: str


def wms_cache_key(
    lon: float,
    lat: float,
    *,
    half_side_m: float,
    width_px: int,
    height_px: int,
    wms_base: str,
    layer: str,
    imagery_profile: str = "bdortho",
    analysis_radius_m: Optional[float] = None,
) -> str:
    rpart = f"|rm{float(analysis_radius_m):.2f}" if analysis_radius_m is not None else ""
    s = (
        f"{WMS_CACHE_VERSION}|{imagery_profile}|{wms_base}|{layer}|"
        f"{lon:.7f}|{lat:.7f}|{half_side_m:.4f}|{width_px}|{height_px}{rpart}"
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def _cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "wms_ortho" / f"{key}.png"


def fetch_ortho_chip(
    lon: float,
    lat: float,
    *,
    half_side_m: float = 50.0,
    width_px: int = 512,
    height_px: int = 512,
    wms_base: str = DEFAULT_WMS_BASE,
    layer: str = DEFAULT_WMS_LAYER,
    client: httpx.Client | None = None,
    cache_dir: Optional[Path] = None,
    refresh_imagery: bool = False,
    imagery_profile: str = "bdortho",
    analysis_radius_m: Optional[float] = None,
) -> OrthoChip:
    """
    Récupère une image GetMap WMS 1.3.0 en EPSG:3857 centrée sur (lon, lat).

    Si ``cache_dir`` est défini et ``refresh_imagery`` est faux, tente de lire
    ``<cache_dir>/wms_ortho/<clé>.png`` ; sinon télécharge et enregistre.
    """
    from parking_capacity.geometry import bbox_around_point_m

    minx, miny, maxx, maxy = bbox_around_point_m(lon, lat, half_side_m)

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": "EPSG:3857",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": str(width_px),
        "HEIGHT": str(height_px),
        "FORMAT": "image/png",
    }
    url = f"{wms_base.rstrip('/')}?{urlencode(params)}"

    key = wms_cache_key(
        lon,
        lat,
        half_side_m=half_side_m,
        width_px=width_px,
        height_px=height_px,
        wms_base=wms_base,
        layer=layer,
        imagery_profile=imagery_profile,
        analysis_radius_m=analysis_radius_m,
    )
    cpath: Optional[Path] = None
    if cache_dir is not None and not refresh_imagery:
        cpath = _cache_path(Path(cache_dir), key)
        if cpath.is_file():
            logger.info("WMS cache hit %s", cpath.name)
            img = Image.open(cpath).convert("RGB")
            return OrthoChip(
                image=img,
                minx=minx,
                miny=miny,
                maxx=maxx,
                maxy=maxy,
                width_px=width_px,
                height_px=height_px,
                layer=layer,
            )
        logger.info("WMS cache miss %s (téléchargement)", cpath.name if cpath else key)
    elif cache_dir is not None and refresh_imagery:
        logger.info("WMS refresh forcé (--refresh-imagery) pour %.5f,%.5f", lat, lon)
    elif cache_dir is None:
        logger.debug("WMS sans cache fichier pour %.5f,%.5f", lat, lon)

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        r = client.get(url)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if cache_dir is not None:
            cpath = _cache_path(Path(cache_dir), key)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            img.save(cpath, format="PNG")
            logger.info("WMS cache écrit %s", cpath.name)
    finally:
        if own_client:
            client.close()

    return OrthoChip(
        image=img,
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        width_px=width_px,
        height_px=height_px,
        layer=layer,
    )


def chip_m2_per_pixel(chip: OrthoChip) -> float:
    """Surface au sol m² par pixel (approx., plan horizontal)."""
    dx = (chip.maxx - chip.minx) / chip.width_px
    dy = (chip.maxy - chip.miny) / chip.height_px
    return float(dx * dy)
