"""Résolution capacité : géométrie vs vision spécialisée vs surface seule."""

from __future__ import annotations

from parking_capacity.pipeline_resolve import pick_primary_capacity
from parking_capacity.vision_estimate import VisionEstimate


def _v(spaces: int = 35) -> VisionEstimate:
    return VisionEstimate(
        parking_pixel_fraction=0.05,
        parking_area_m2=900.0,
        estimated_spaces=spaces,
        confidence="medium",
        device="cpu",
        notes="",
    )


def test_weak_geometry_falls_back_to_geometry_as_hint():
    p, src, conf, prov = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.95,
        vision_est=None,
        vision_primary_places=None,
        geometry_places=19,
        geometry_confidence="weak",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=200,
        mn_a=180,
        mx_a=220,
        ta=5000.0,
        osm_parking_space_count=0,
        visual_evidence_level="weak",
    )
    assert p == 19
    assert src == "parking_geometry"
    assert conf == "low"


def test_osm_parking_space_count_beats_weak_geometry():
    p, src, conf, prov = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.95,
        vision_est=None,
        vision_primary_places=None,
        geometry_places=19,
        geometry_confidence="weak",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=200,
        mn_a=180,
        mx_a=220,
        ta=5000.0,
        osm_parking_space_count=86,
        visual_evidence_level="weak",
    )
    assert p == 86
    assert src == "osm_parking_space_count"
    assert conf == "medium"


def test_no_primary_when_only_surface_signal():
    """Sans géométrie, sans OSM tag, sans vision spécialisée : pas de primary."""
    v = _v(35)
    p, src, _, prov = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.6,
        vision_est=v,
        vision_primary_places=None,  # SegFormer non spécialisé : count = None
        geometry_places=None,
        geometry_confidence="none",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=200,
        mn_a=180,
        mx_a=220,
        ta=5000.0,
        osm_parking_space_count=0,
        visual_evidence_level="weak",
        visual_specialized_effective=False,
    )
    assert p is None
    assert src is None
    assert "aucune" in prov


def test_specialized_vision_used_when_specialized_effective():
    v = _v(50)
    p, src, _, _ = pick_primary_capacity(
        source_priority="aerial",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.95,
        vision_est=v,
        vision_primary_places=50,
        geometry_places=None,
        geometry_confidence="none",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=0,
        mn_a=0,
        mx_a=0,
        ta=0.0,
        osm_parking_space_count=0,
        visual_evidence_level="medium",
        visual_specialized_effective=True,
    )
    assert p == 50
    assert src == "vision_marked_visible"


def test_segformer_generic_count_never_promoted_to_primary():
    """Sans spécialisation effective : vision_primary_places n'est PAS retenu."""
    v = _v(50)
    p, src, _, _ = pick_primary_capacity(
        source_priority="aerial",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.95,
        vision_est=v,
        vision_primary_places=50,
        geometry_places=None,
        geometry_confidence="none",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=0,
        mn_a=0,
        mx_a=0,
        ta=0.0,
        osm_parking_space_count=0,
        visual_evidence_level="weak",
        visual_specialized_effective=False,
    )
    assert p is None
    assert src is None


def test_medium_geometry_beats_unspecialized_vision():
    v = _v(50)
    p, src, _, _ = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.95,
        vision_est=v,
        vision_primary_places=50,
        geometry_places=19,
        geometry_confidence="medium",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=200,
        mn_a=180,
        mx_a=220,
        ta=5000.0,
        osm_parking_space_count=0,
        visual_evidence_level="medium",
        visual_specialized_effective=False,
    )
    assert p == 19
    assert src == "parking_geometry"


def test_strong_geometry_gives_high_confidence():
    p, src, conf, _ = pick_primary_capacity(
        source_priority="hybrid",
        has_osm_capacity=False,
        cap_p=0,
        cap_b=0,
        tagged_p=0,
        tagged_b=0,
        ban_score=0.95,
        vision_est=None,
        vision_primary_places=None,
        geometry_places=120,
        geometry_confidence="strong",
        ml_int=None,
        ml_mode_l="fallback",
        mid_a=0,
        mn_a=0,
        mx_a=0,
        ta=0.0,
        osm_parking_space_count=0,
        visual_evidence_level="strong",
    )
    assert p == 120
    assert src == "parking_geometry"
    assert conf == "high"
