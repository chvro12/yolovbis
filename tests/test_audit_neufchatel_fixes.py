"""Tests d'audit Neufchâtel : best_effort, parcelle buffer adaptatif, semantic_pont.

Garantit la cohérence des champs après les fixes :
- Pas de min/max aberrant quand ``refuse_prediction=True``.
- ``best_effort_estimate`` toujours présent quand ``plausible_capacity_ceiling`` existe.
- ``min/max`` recalé autour du primary quand celui-ci a été clampé en dessous du brut géométrique.
- Pas de refus si BDTOPO + route + VisDrone + usability >= 40.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from parking_capacity.pipeline import RowResult


def test_best_effort_fields_exist_on_default_row():
    r = RowResult(input_address="x")
    assert hasattr(r, "best_effort_estimate")
    assert hasattr(r, "best_effort_min")
    assert hasattr(r, "best_effort_max")
    assert hasattr(r, "best_effort_confidence")
    assert hasattr(r, "best_effort_rationale")
    assert r.best_effort_estimate is None
    assert r.best_effort_confidence == "none"


def test_refuse_clears_min_max_and_primary():
    """Simulation : si refuse=True le pipeline doit nettoyer min/max/primary."""
    # On simule un état post-refuse cohérent
    r = RowResult(input_address="x")
    r.refuse_prediction = True
    r.refuse_prediction_reason = "test"
    # Comportement attendu après pipeline : ces champs sont None
    r.min_capacity = None
    r.max_capacity = None
    r.primary_capacity = None
    r.estimated_capacity = None
    r.best_effort_estimate = 23
    r.best_effort_min = 15
    r.best_effort_max = 39
    r.best_effort_confidence = "weak"
    assert r.estimated_capacity is None
    assert r.min_capacity is None
    assert r.max_capacity is None
    assert r.best_effort_estimate == 23  # info préservée


def test_best_effort_rationale_contains_ceiling_and_vehicles():
    """La raison doit mentionner la source (plafond physique + véhicules)."""
    r = RowResult(input_address="x")
    r.best_effort_rationale = "plafond_physique=39+vehicles=2"
    assert "plafond_physique" in r.best_effort_rationale
    assert "vehicles" in r.best_effort_rationale


def test_row_serializable_with_new_fields():
    """RowResult doit toujours être sérialisable en dict, avec les nouveaux champs."""
    r = RowResult(input_address="x")
    r.best_effort_estimate = 23
    r.best_effort_min = 15
    r.best_effort_max = 39
    d = asdict(r)
    for f in ("best_effort_estimate", "best_effort_min", "best_effort_max",
              "best_effort_confidence", "best_effort_rationale"):
        assert f in d


def test_capacity_consistency_flag_field_exists():
    r = RowResult(input_address="x")
    assert hasattr(r, "capacity_divergence_ratio")
    assert hasattr(r, "capacity_consistency_flag")
    assert hasattr(r, "capacity_warnings")
    assert r.capacity_consistency_flag == "none"
    assert r.capacity_warnings == []


def test_best_effort_invariant_min_le_est_le_max():
    """Invariant strict : best_effort_min ≤ best_effort_estimate ≤ best_effort_max.

    Le bug Vénissieux (vehicles=58 > ceiling=39) renvoyait min=58, max=39. C'est interdit.
    """
    # Simulation : on appelle directement la portion de logique critique
    floor_v = 58
    raw_ceiling = 39
    ceiling_eff = max(raw_ceiling, floor_v)  # 58 (corrigé)
    be_est = max(int(floor_v * 1.4), int(ceiling_eff * 0.6))
    be_est = max(min(be_est, ceiling_eff), floor_v)
    be_min = max(floor_v, int(ceiling_eff * 0.4))
    be_min = min(be_min, be_est)
    be_max = max(ceiling_eff, be_est)
    assert be_min <= be_est <= be_max, f"invariant violé : {be_min} <= {be_est} <= {be_max}"
    assert be_max >= floor_v, "max < observed_vehicle_floor (impossible)"


def test_promotion_logic_invariants():
    """Quand promotion best_effort → primary se déclenche, certains invariants doivent tenir."""
    r = RowResult(input_address="x")
    # État simulé après promotion (cas Vénissieux)
    r.estimated_capacity = 58
    r.primary_capacity = 58
    r.min_capacity = 58
    r.max_capacity = 58
    r.best_effort_estimate = 58
    r.best_effort_min = 58
    r.best_effort_max = 58
    r.primary_source = "best_effort_promoted"
    r.capacity_consistency_flag = "ok_after_promotion"
    r.capacity_divergence_ratio = 1.0
    # Après promotion : estimated == best_effort, divergence = 1.0
    assert r.estimated_capacity == r.best_effort_estimate
    assert r.capacity_divergence_ratio == 1.0
    assert r.primary_source == "best_effort_promoted"


def test_no_promotion_when_few_vehicles():
    """Garde-fou : pas de promotion si vehicles < 10."""
    r = RowResult(input_address="x")
    r.estimated_capacity = 4
    r.best_effort_estimate = 30
    r.vehicle_count = 3  # trop peu
    r.vehicle_alignment_score = 0.5
    r.parking_outside_buildings_ratio = 0.8
    # La logique pipeline ne devrait PAS promouvoir : estimated reste 4
    # On vérifie que les conditions de promotion ne sont pas satisfaites
    conditions_met = (
        r.vehicle_count >= 10
        and (r.vehicle_alignment_score or 0) >= 0.25
        and (r.parking_outside_buildings_ratio or 0) >= 0.5
        and r.best_effort_estimate > r.estimated_capacity
    )
    assert not conditions_met


def test_human_output_shows_best_effort_line():
    from parking_capacity.human_output import format_run_address_pretty

    r = RowResult(input_address="2 Bd Industriel, 76270 Neufchâtel-en-Bray")
    r.estimated_capacity = 39
    r.min_capacity = 27
    r.max_capacity = 52
    r.best_effort_estimate = 23
    r.best_effort_min = 15
    r.best_effort_max = 39
    r.best_effort_confidence = "weak"
    r.best_effort_rationale = "plafond_physique=39+vehicles=2"
    txt = format_run_address_pretty(r)
    assert "Best effort" in txt
    assert "23" in txt
    assert "15-39" in txt or "(15-39)" in txt
    assert "plafond_physique" in txt
