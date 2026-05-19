"""Tests des règles de cohérence — chaque règle vérifiée isolément."""

from __future__ import annotations

from parking_capacity.consistency_checks import (
    ConsistencyFlag,
    run_consistency_checks,
    summarize_flags,
)


def _base(**overrides):
    """Construit un dict de signaux neutres (aucun flag ne se déclenche)."""
    d = {
        "estimated_capacity": 5,
        "min_capacity": 3,
        "max_capacity": 7,
        "vehicle_count": 3,
        "parking_area_detected_m2": 100.0,
        "area_total_m2": 100.0,
        "parking_outside_buildings_ratio": 0.6,
        "site_typology_min": 3,
        "site_typology_max": 12,
        "site_typology_confidence": "medium",
        "site_typology_family": "vet_clinic",
        "primary_source": "scenario_unmarked_surface",
        "primary_confidence": "medium",
        "plausible_capacity_ceiling": 30,
        "nearby_public_capacity_estimate": None,
        "slots_total_count": 5,
        "slot_detection_method": "heuristic_vehicles_rows",
        "n_parkings_parcelle": 0,
        "capacity_osm_parcelle": 0,
    }
    d.update(overrides)
    return d


def _names(flags):
    return {f.name for f in flags}


def test_no_flags_on_neutral_prediction():
    flags = run_consistency_checks(_base())
    assert flags == []


def test_returns_empty_when_no_estimate():
    flags = run_consistency_checks(_base(estimated_capacity=None))
    assert flags == []


def test_invented_surface_flag_fires():
    # MONTAUBAN-like : 13 places prédites, 0 véhicule, ~0 m² surface, aucun parking OSM.
    flags = run_consistency_checks(_base(
        estimated_capacity=13,
        vehicle_count=0,
        parking_area_detected_m2=0.0,
        area_total_m2=0.0,
        n_parkings_parcelle=0,
    ))
    assert "invented_surface" in _names(flags)
    f = next(f for f in flags if f.name == "invented_surface")
    assert f.severity == "high"


def test_invented_surface_does_not_fire_with_vehicles():
    flags = run_consistency_checks(_base(
        estimated_capacity=13,
        vehicle_count=4,
        parking_area_detected_m2=0.0,
    ))
    assert "invented_surface" not in _names(flags)


def test_counting_buildings_flag_fires():
    # CERISE-like : grosse prédiction sur quasi pas de surface hors-bâti.
    flags = run_consistency_checks(_base(
        estimated_capacity=39,
        parking_outside_buildings_ratio=0.08,
    ))
    assert "counting_buildings" in _names(flags)


def test_counting_buildings_does_not_fire_when_outside_ratio_healthy():
    flags = run_consistency_checks(_base(
        estimated_capacity=39,
        parking_outside_buildings_ratio=0.7,
    ))
    assert "counting_buildings" not in _names(flags)


def test_typology_exceeded_flag_fires():
    flags = run_consistency_checks(_base(
        estimated_capacity=50,
        site_typology_max=20,
    ))
    assert "typology_exceeded" in _names(flags)


def test_typology_exceeded_requires_confident_typology():
    # Typologie incertaine → on ne flag pas (évite faux positifs).
    flags = run_consistency_checks(_base(
        estimated_capacity=50,
        site_typology_max=20,
        site_typology_confidence="none",
    ))
    assert "typology_exceeded" not in _names(flags)


def test_typology_underestimated_flag_fires():
    flags = run_consistency_checks(_base(
        estimated_capacity=1,
        site_typology_min=10,
        site_typology_confidence="strong",
    ))
    assert "typology_underestimated" in _names(flags)


def test_relies_on_nearby_public_flag_is_info_only():
    flags = run_consistency_checks(_base(
        estimated_capacity=0,
        nearby_public_capacity_estimate=25,
    ))
    f = next(f for f in flags if f.name == "relies_on_nearby_public")
    assert f.severity == "info"


def test_low_confidence_large_value_flag_fires():
    flags = run_consistency_checks(_base(
        estimated_capacity=20,
        primary_confidence="low",
    ))
    assert "low_confidence_large_value" in _names(flags)


def test_low_confidence_small_value_does_not_fire():
    # Petite prédiction même avec confiance basse → pas de flag (erreur attendue petite).
    flags = run_consistency_checks(_base(
        estimated_capacity=3,
        primary_confidence="low",
    ))
    assert "low_confidence_large_value" not in _names(flags)


def test_ceiling_saturated_flag_fires():
    # ALLONZIER-pré-fix-like : estimation collée au plafond, min=max.
    flags = run_consistency_checks(_base(
        estimated_capacity=39,
        min_capacity=39,
        max_capacity=39,
        plausible_capacity_ceiling=39,
    ))
    assert "ceiling_saturated" in _names(flags)


def test_ceiling_not_saturated_when_min_below_max():
    flags = run_consistency_checks(_base(
        estimated_capacity=39,
        min_capacity=30,
        max_capacity=39,
        plausible_capacity_ceiling=39,
    ))
    assert "ceiling_saturated" not in _names(flags)


def test_marked_slots_source_no_detection_fires():
    flags = run_consistency_checks(_base(
        estimated_capacity=10,
        primary_source="private_marked_slots",
        slots_total_count=0,
        slot_detection_method="heuristic_vehicles_rows",
    ))
    assert "marked_slots_source_no_detection" in _names(flags)


def test_osm_tagged_divergence_fires():
    # OSM dit 50 places, on prédit 10 → écart > 50%.
    flags = run_consistency_checks(_base(
        estimated_capacity=10,
        capacity_osm_parcelle=50,
    ))
    assert "osm_tagged_divergence" in _names(flags)


def test_osm_tagged_no_divergence_when_close():
    flags = run_consistency_checks(_base(
        estimated_capacity=22,
        capacity_osm_parcelle=20,
    ))
    assert "osm_tagged_divergence" not in _names(flags)


def test_summarize_flags_counts_and_severity():
    flags = [
        ConsistencyFlag("a", "high", "x"),
        ConsistencyFlag("b", "medium", "y"),
        ConsistencyFlag("c", "info", "z"),
    ]
    s = summarize_flags(flags)
    assert s["high_count"] == 1
    assert s["medium_count"] == 1
    assert s["info_count"] == 1
    assert s["max_severity"] == "high"
    assert s["needs_review"] is True


def test_summarize_flags_empty():
    s = summarize_flags([])
    assert s["count"] == 0
    assert s["max_severity"] == "none"
    assert s["needs_review"] is False


def test_summarize_flags_only_info_does_not_need_review():
    s = summarize_flags([ConsistencyFlag("a", "info", "x")])
    assert s["max_severity"] == "info"
    assert s["needs_review"] is False
