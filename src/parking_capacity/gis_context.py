"""Compatibilité : la fusion GIS vit dans ``gis_fusion``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from parking_capacity.gis_fusion import GisFusionResult, build_gis_fusion
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.providers_config import GisProvidersConfig


def fetch_chip_gis_augmentation(
    chip: OrthoChip,
    lat: float,
    lon: float,
    *,
    radius_m: int,
    cfg: GisProvidersConfig,
    client: httpx.Client,
    cache_dir: Optional[Path],
    overpass_delay_s: float,
    access_distance_threshold_m: Optional[float] = None,
) -> GisFusionResult:
    """Alias historique : délègue à ``build_gis_fusion``."""
    thr = (
        float(access_distance_threshold_m)
        if access_distance_threshold_m is not None
        else float(cfg.access_distance_threshold_m)
    )
    return build_gis_fusion(
        chip,
        lat,
        lon,
        radius_m=radius_m,
        cfg=cfg,
        client=client,
        cache_dir=cache_dir,
        overpass_delay_s=overpass_delay_s,
        access_distance_threshold_m=thr,
    )
