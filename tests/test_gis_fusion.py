"""Tests fusion GIS (BD TOPO prioritaire, routes OSM/IGN, surface utile)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from parking_capacity.gis_fusion import GisFusionResult, compute_fusion_area_metrics
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.semantic_layer import compute_semantic_context
from parking_capacity.surface_classification import SurfaceClassification
from parking_capacity.vehicle_detection import VehicleDetectionResult


def _tiny_chip(h: int = 64, w: int = 64) -> OrthoChip:
    return OrthoChip(
        image=Image.new("RGB", (w, h), color=(140, 140, 140)),
        minx=0.0,
        miny=0.0,
        maxx=100.0,
        maxy=100.0,
        width_px=w,
        height_px=h,
        layer="test",
    )


def test_bdtopo_building_mask_prioritaire_sur_heuristique():
    chip = _tiny_chip()
    h, w = chip.height_px, chip.width_px
    bd = np.zeros((h, w), dtype=bool)
    bd[10:30, 10:50] = True

    asphalt = np.ones((h, w), dtype=bool)
    roof = np.zeros((h, w), dtype=bool)
    road = np.zeros((h, w), dtype=bool)
    veg = np.zeros((h, w), dtype=bool)
    shadow = np.zeros((h, w), dtype=bool)
    edge = np.zeros((h, w), dtype=bool)
    eligible = asphalt & ~roof & ~road & ~veg & ~shadow
    surf = SurfaceClassification(
        asphalt_mask=asphalt,
        roof_mask=roof,
        road_mask=road,
        vegetation_mask=veg,
        shadow_mask=shadow,
        building_edge_mask=edge,
        parking_eligible_mask=eligible,
        asphalt_likelihood=0.5,
        roof_likelihood=0.0,
        road_likelihood=0.0,
        vegetation_likelihood=0.0,
        shadow_likelihood=0.0,
        building_edge_likelihood=0.0,
    )
    rgb = np.asarray(chip.image)
    m_per_px = math.sqrt(100.0 * 100.0 / (h * w))
    veh = VehicleDetectionResult()
    sem = compute_semantic_context(
        rgb,
        surf,
        m_per_px=m_per_px,
        vehicles=veh,
        bdtopo_buildings_mask=bd,
        building_mask_source="bdtopo",
        gis_road_connection=True,
        road_network_score_gis=0.8,
        road_source="osm",
        max_plausible_slots_cap=39,
    )
    assert sem.building_mask_source == "bdtopo"
    assert sem.road_connection_detected is True
    assert sem.building_area_m2 > 0


def test_route_gis_active_connexion():
    chip = _tiny_chip()
    h, w = chip.height_px, chip.width_px
    asphalt = np.ones((h, w), dtype=bool)
    surf = SurfaceClassification(
        asphalt_mask=asphalt,
        roof_mask=np.zeros((h, w), dtype=bool),
        road_mask=np.zeros((h, w), dtype=bool),
        vegetation_mask=np.zeros((h, w), dtype=bool),
        shadow_mask=np.zeros((h, w), dtype=bool),
        building_edge_mask=np.zeros((h, w), dtype=bool),
        parking_eligible_mask=asphalt.copy(),
        asphalt_likelihood=0.5,
        roof_likelihood=0.0,
        road_likelihood=0.0,
        vegetation_likelihood=0.0,
        shadow_likelihood=0.0,
        building_edge_likelihood=0.0,
    )
    rgb = np.asarray(chip.image)
    m_per_px = 1.0
    veh = VehicleDetectionResult()
    sem = compute_semantic_context(
        rgb,
        surf,
        m_per_px=m_per_px,
        vehicles=veh,
        gis_road_connection=True,
        access_distance_m_gis=12.0,
        road_network_score_gis=0.75,
        road_source="bdtopo",
        building_mask_source="heuristic",
    )
    assert sem.road_connection_detected is True
    assert sem.access_distance_m == 12.0


def test_exclusion_batiment_usable_area():
    chip = _tiny_chip(32, 32)
    h, w = 32, 32
    bd = np.zeros((h, w), dtype=bool)
    bd[4:20, 4:20] = True
    fusion = GisFusionResult(building_mask_hw=bd, building_mask_source="bdtopo")
    eligible = np.ones((h, w), dtype=bool)
    road = np.zeros((h, w), dtype=bool)
    excl, usable_m2, final_m2, mask = compute_fusion_area_metrics(chip, eligible, road, fusion)
    assert excl > 0
    assert usable_m2 < float(eligible.sum()) * (100.0 * 100.0 / (h * w))


def test_neufchatel_mock_cap_plafond():
    """Plafond produit ≤ 39 même si surface théorique immense (mock)."""
    chip = _tiny_chip(32, 32)
    h, w = 32, 32
    asphalt = np.ones((h, w), dtype=bool)
    surf = SurfaceClassification(
        asphalt_mask=asphalt,
        roof_mask=np.zeros((h, w), dtype=bool),
        road_mask=np.ones((h, w), dtype=bool),
        vegetation_mask=np.zeros((h, w), dtype=bool),
        shadow_mask=np.zeros((h, w), dtype=bool),
        building_edge_mask=np.zeros((h, w), dtype=bool),
        parking_eligible_mask=asphalt.copy(),
        asphalt_likelihood=0.9,
        roof_likelihood=0.0,
        road_likelihood=0.2,
        vegetation_likelihood=0.0,
        shadow_likelihood=0.0,
        building_edge_likelihood=0.0,
    )
    rgb = np.asarray(chip.image)
    m_per_px = 0.5
    veh = VehicleDetectionResult()
    sem = compute_semantic_context(
        rgb,
        surf,
        m_per_px=m_per_px,
        vehicles=veh,
        gis_road_connection=True,
        road_network_score_gis=0.9,
        road_source="bdtopo+osm",
        building_mask_source="osm",
        max_plausible_slots_cap=39,
    )
    assert sem.road_connection_detected is True
    assert sem.plausible_capacity_ceiling is not None
    assert sem.plausible_capacity_ceiling <= 39
