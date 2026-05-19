"""Tests typologie SIRENE/OSM + calibration capacity."""

from __future__ import annotations

import pytest

from parking_capacity.site_typology import (
    SiteTypology,
    apply_typology_to_estimate,
    classify_site,
)
from parking_capacity.sirene_client import _parking_relevance, _address_match_score


def test_classify_ape_vet_clinic():
    t = classify_site(ape_code="75.00Z")
    assert t.family == "vet_clinic"
    assert t.confidence == "medium"
    assert t.expected_capacity_min == 3
    assert t.expected_capacity_max == 20


def test_classify_ape_supermarket():
    t = classify_site(ape_code="47.11D")
    assert t.family == "supermarket"
    assert t.expected_capacity_min == 40
    assert t.expected_capacity_max == 150


def test_classify_ape_hospital():
    t = classify_site(ape_code="86.10Z")
    assert t.family == "hospital_large"
    assert t.expected_capacity_min == 100


def test_classify_residential_holding_low_priority():
    """Un APE 68.20 (location logement) doit donner 'residential' fourchette modeste."""
    t = classify_site(ape_code="68.20B")
    assert t.family == "residential"
    assert t.expected_capacity_max <= 50


def test_classify_unknown_returns_none_confidence():
    t = classify_site(ape_code="99.99Z")
    # On retombe potentiellement sur OSM si pas d'APE
    assert t.family == "unknown" or t.confidence in ("none", "weak")


def test_classify_osm_only():
    t = classify_site(osm_amenity="hospital")
    assert t.family == "hospital_large"
    assert t.confidence == "medium"


def test_classify_ape_osm_concordant_gives_strong():
    """APE 75.00 + OSM amenity=veterinary → strong confidence."""
    t = classify_site(ape_code="75.00Z", osm_amenity="veterinary")
    assert t.confidence == "strong"
    assert t.family == "vet_clinic"


def test_parking_relevance_vet_high():
    """Un vétérinaire est très pertinent pour la détection parking."""
    rel_vet = _parking_relevance("75.00Z")
    rel_holding = _parking_relevance("68.20B")
    assert rel_vet > rel_holding
    assert rel_vet >= 0.9
    assert rel_holding <= 0.3


def test_parking_relevance_supermarket_high():
    assert _parking_relevance("47.11D") >= 0.9


def test_parking_relevance_default_neutral():
    assert 0.3 < _parking_relevance("99.99Z") < 0.7


def test_address_match_score_basic():
    s1 = _address_match_score("2 Bd Industriel 76270 Neufchâtel-en-Bray",
                              "2 BOULEVARD INDUSTRIEL 76270 NEUFCHATEL-EN-BRAY")
    s2 = _address_match_score("2 Bd Industriel Neufchâtel",
                              "10 RUE DIFFERENT PARIS")
    assert s1 > 0.5
    assert s2 < 0.3


def test_apply_typology_to_estimate_vet():
    """Cas Neufchâtel : 9 vehicles + occupation 0.4 = ~22 places."""
    t = SiteTypology(
        family="vet_clinic", label="Clinique vétérinaire",
        expected_capacity_min=3, expected_capacity_max=20,
        expected_occupation_rate=0.4, confidence="medium",
    )
    est, lo, hi, _ = apply_typology_to_estimate(
        t, vehicle_count=9, plausible_ceiling=39,
    )
    assert lo is not None and est is not None and hi is not None
    assert lo <= est <= hi
    assert 15 <= est <= 25  # cohérent avec vérité Neufchâtel


def test_apply_typology_supermarket():
    """Cas Vénissieux : 31 vehicles + occupation 0.55 = ~56 places."""
    t = SiteTypology(
        family="supermarket", label="Supermarché",
        expected_capacity_min=40, expected_capacity_max=150,
        expected_occupation_rate=0.55, confidence="medium",
    )
    est, lo, hi, _ = apply_typology_to_estimate(
        t, vehicle_count=31, plausible_ceiling=39,
    )
    # On veut au moins 40 (typology min) même si ceiling=39 (ceiling-eff sera relevé par vehicles)
    assert lo == 40  # typology min prend le dessus
    assert est >= 40


def test_apply_typology_returns_none_if_no_typology():
    t = SiteTypology()  # confidence=none par défaut
    est, _, _, _ = apply_typology_to_estimate(t, vehicle_count=5, plausible_ceiling=20)
    assert est is None
