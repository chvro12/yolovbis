from __future__ import annotations

import math

import numpy as np
from PIL import Image

from parking_capacity.gis_fusion import GisFusionResult
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.private_parking_area import compute_private_parking_area
from parking_capacity.surface_classification import SurfaceClassification


EARTH_R = 6378137.0


def _lonlat_from_webmercator(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / EARTH_R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / EARTH_R)) - math.pi / 2.0)
    return lon, lat


def test_private_area_excludes_buildings_and_roads() -> None:
    chip = OrthoChip(
        image=Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
        minx=0.0,
        miny=0.0,
        maxx=100.0,
        maxy=100.0,
        width_px=100,
        height_px=100,
        layer="test",
    )
    parking = np.ones((100, 100), dtype=bool)
    surface = SurfaceClassification(
        asphalt_mask=parking,
        roof_mask=np.zeros_like(parking),
        road_mask=np.zeros_like(parking),
        vegetation_mask=np.zeros_like(parking),
        shadow_mask=np.zeros_like(parking),
        building_edge_mask=np.zeros_like(parking),
        parking_eligible_mask=parking,
    )
    building = np.zeros_like(parking)
    building[40:60, 40:60] = True
    road = np.zeros_like(parking)
    road[:, :10] = True
    fusion = GisFusionResult(building_mask_hw=building, road_mask_gis_hw=road)
    parcel = [
        _lonlat_from_webmercator(10.0, 10.0),
        _lonlat_from_webmercator(90.0, 10.0),
        _lonlat_from_webmercator(90.0, 90.0),
        _lonlat_from_webmercator(10.0, 90.0),
        _lonlat_from_webmercator(10.0, 10.0),
    ]

    area = compute_private_parking_area(chip, surface, parcel_polygon_lonlat=parcel, fusion=fusion)

    assert area.source == "cadastre_surface_gis"
    assert area.parcel_area_m2 > 6000
    assert area.building_area_m2 > 300
    assert area.road_area_m2 > 0
    assert 0 < area.usable_area_m2 < area.parcel_area_m2 - area.building_area_m2
