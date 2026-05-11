"""Règles produit : capacité théorique (``parking_capacity_estimation``) ≠ comptage véhicules."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.parking_scenarios import analyze_parking_scenarios
from parking_capacity.pipeline_resolve import pick_primary_capacity
from parking_capacity.semantic_layer import clamp_capacity_to_semantic_bounds
from parking_capacity.site_classification import infer_site_type


def _chip_rgb(arr: np.ndarray, side_m: float = 80.0) -> OrthoChip:
    h, w = arr.shape[:2]
    return OrthoChip(
        image=Image.fromarray(arr),
        minx=0.0,
        miny=0.0,
        maxx=side_m,
        maxy=side_m,
        width_px=w,
        height_px=h,
        layer="synth",
    )


def test_infer_site_type_public_parking_from_amenity():
    st = infer_site_type(
        "2 Boulevard Industriel 76270 Neufchâtel-en-Bray",
        osm_amenity_tags={"parking"},
    )
    assert st == "public_parking"


def test_infer_site_type_clinic_keyword():
    assert infer_site_type("Parking Clinique du Moulin", osm_amenity_tags=set()) == "clinic"


def test_pick_primary_osm_capacity_before_all():
    p, s, _, _ = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=True,
        cap_p=42,
        cap_b=0,
        tagged_p=1,
        tagged_b=0,
        ban_score=0.9,
        vision_est=None,
        vision_primary_places=500,
        geometry_places=200,
        geometry_confidence="strong",
        ml_int=80,
        ml_mode_l="fallback",
        mid_a=0,
        mn_a=0,
        mx_a=0,
        ta=0.0,
        osm_parking_space_count=10,
        visual_evidence_level="strong",
        visual_specialized_effective=True,
        scenario_primary_capacity=30,
        scenario_primary_source="scenario_unmarked_surface",
        scenario_primary_confidence="medium",
    )
    assert p == 42 and s == "osm_parcelle"


def test_pick_primary_vision_marked_visible_before_strong_geometry():
    p, s, _, prov = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.9,
        vision_est=None,
        vision_primary_places=48,
        geometry_places=200,
        geometry_confidence="strong",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=0,
        mn_a=0,
        mx_a=0,
        ta=0.0,
        osm_parking_space_count=0,
        visual_evidence_level="strong",
        visual_specialized_effective=True,
        scenario_primary_capacity=30,
        scenario_primary_source="scenario_unmarked_surface",
        scenario_primary_confidence="medium",
    )
    assert p == 48 and s == "vision_marked_visible"
    assert "places_marquées_visibles" in prov


def test_clamp_function_still_supports_explicit_floor():
    cap, notes = clamp_capacity_to_semantic_bounds(5, ceiling=100, floor=50)
    assert cap == 50
    assert notes


def test_neufchatel_synthetic_cap_or_refuse():
    pytest.importorskip("cv2")
    arr = np.full((512, 512, 3), [80, 130, 70], dtype=np.uint8)
    arr[60:230, 80:240] = [170, 90, 70]
    arr[60:230, 270:430] = [170, 90, 70]
    arr[290:325, 100:420] = [115, 115, 115]
    chip = _chip_rgb(arr, side_m=60.0)
    r = analyze_parking_scenarios(chip, site_type="clinic")
    primary = r.primary_estimate
    if primary and primary.capacity_estimate is not None:
        if r.semantic and r.semantic.plausible_capacity_ceiling is not None:
            assert primary.capacity_estimate <= r.semantic.plausible_capacity_ceiling + 1
