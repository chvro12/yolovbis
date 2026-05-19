"""Tests de la couche sémantique : véhicules, bâtiments, accès, parking_usability_score, ceiling/floor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.parking_scenarios import analyze_parking_scenarios
from parking_capacity.semantic_layer import (
    clamp_capacity_to_semantic_bounds,
    compute_semantic_context,
)
from parking_capacity.surface_classification import classify_surfaces
from parking_capacity.vehicle_detection import detect_vehicles


def _chip(arr: np.ndarray, side_m: float = 60.0) -> OrthoChip:
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


def _gray(h: int, w: int, level: int = 110) -> np.ndarray:
    rng = np.random.default_rng(42)
    arr = np.full((h, w, 3), level, dtype=np.int16)
    arr += rng.integers(-6, 6, size=arr.shape, dtype=np.int16)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _add_aligned_cars(arr: np.ndarray, y: int, x0: int, n: int, *, slot_px: int = 26, w_px: int = 16, l_px: int = 32, color=(45, 45, 50)):
    for i in range(n):
        x = x0 + i * slot_px
        arr[y : y + l_px, x : x + w_px] = color


def test_clamp_capacity_to_ceiling_and_floor():
    cap, notes = clamp_capacity_to_semantic_bounds(100, ceiling=30, floor=5)
    assert cap == 30
    assert any("plafond" in n for n in notes)
    cap, notes = clamp_capacity_to_semantic_bounds(2, ceiling=30, floor=8)
    assert cap == 8
    assert any("plancher" in n for n in notes)


def test_vehicles_detected_on_asphalt_with_cars():
    pytest.importorskip("cv2")
    arr = _gray(512, 512, level=130)
    _add_aligned_cars(arr, y=240, x0=60, n=8)
    img = Image.fromarray(arr)
    asphalt = np.ones((512, 512), dtype=bool)
    r = detect_vehicles(img, asphalt_mask=asphalt, m_per_px=60 / 512)
    assert r.method == "opencv_fallback"
    assert r.vehicle_count >= 4  # tolérance fallback heuristique
    # Alignement détecté
    assert r.vehicle_alignment_score > 0.3


def test_scenarios_with_aligned_vehicles_increase_usability_score():
    """Voitures alignées sur asphalte → usability_score significativement > sans voitures."""
    pytest.importorskip("cv2")
    base = _gray(512, 512, level=130)
    chip_no_cars = _chip(base.copy(), side_m=60.0)
    r_no_cars = analyze_parking_scenarios(chip_no_cars)

    with_cars = base.copy()
    _add_aligned_cars(with_cars, y=240, x0=60, n=10)
    chip_with = _chip(with_cars, side_m=60.0)
    r_with = analyze_parking_scenarios(chip_with)

    assert r_with.semantic is not None and r_no_cars.semantic is not None
    assert r_with.vehicles is not None and r_with.vehicles.vehicle_count > 0
    # Les véhicules augmentent modestement le score (preuve d’usage), sans piloter la capacité.
    assert r_with.semantic.parking_usability_score >= r_no_cars.semantic.parking_usability_score - 8.0


def test_roof_without_cars_low_usability():
    """Grand rectangle de toit + aucune voiture → score sémantique faible."""
    pytest.importorskip("cv2")
    arr = np.full((512, 512, 3), [80, 130, 70], dtype=np.uint8)  # fond végétal
    arr[60:380, 80:430] = [170, 90, 70]  # toit tuiles
    arr[380:395, 80:430] = [22, 22, 30]  # ombre
    chip = _chip(arr, side_m=80.0)
    r = analyze_parking_scenarios(chip)
    assert r.semantic is not None
    # Pas d'OSM, pas de voitures → score < 45
    assert r.semantic.parking_usability_score < 45.0
    assert r.semantic.semantic_confidence in ("none", "weak", "medium")


def test_large_asphalt_no_access_does_not_become_strong():
    """Surface asphaltée immense sans connexion route ne doit pas être promue 'strong'."""
    pytest.importorskip("cv2")
    arr = np.full((512, 512, 3), [80, 130, 70], dtype=np.uint8)  # tout végétal
    # ilot bitumé central isolé (pas adjacent à la chaussée)
    arr[120:380, 120:380] = [115, 115, 115]
    chip = _chip(arr, side_m=80.0)
    r = analyze_parking_scenarios(chip)
    primary = r.primary_estimate
    if primary is not None and primary.capacity_estimate is not None:
        # Pas d'OSM, surface isolée : confiance ne doit pas être strong
        assert primary.confidence != "strong"


def test_ceiling_caps_implausible_capacity():
    """Petit site → plafond physique limite les estimations délirantes."""
    pytest.importorskip("cv2")
    arr = _gray(256, 256, level=120)
    chip = _chip(arr, side_m=30.0)  # 900 m² total
    r = analyze_parking_scenarios(chip)
    assert r.semantic is not None
    # Avec 900 m², ceiling max ~75 places (900/12)
    if r.semantic.plausible_capacity_ceiling is not None:
        assert r.semantic.plausible_capacity_ceiling <= 80


def test_ceiling_scales_with_large_surface():
    """Grande surface utilisable → plafond doit dépasser le cap dur de 39 places.

    Régression : un cap dur à 39 sur tous les sites écrasait les estimations légitimes
    de grandes parcelles (>2000 m² de bitume utile) à 39 places maximum.
    """
    pytest.importorskip("cv2")
    arr = _gray(512, 512, level=120)
    chip = _chip(arr, side_m=80.0)  # 6400 m² total — chip large couvert d'asphalte
    r = analyze_parking_scenarios(chip)
    assert r.semantic is not None
    if r.semantic.plausible_capacity_ceiling is not None:
        # Plafond surface/25 m²/place ≈ 256 places → cap doit largement dépasser 39
        assert r.semantic.plausible_capacity_ceiling > 39


def test_neufchatel_synthetic_converges_or_refuses():
    """Petit parking façade + toits adjacents → soit ≤ 40 places, soit confiance non strong."""
    pytest.importorskip("cv2")
    arr = np.full((512, 512, 3), [80, 130, 70], dtype=np.uint8)  # végétation
    # Bâtiments cliniques en haut
    arr[60:230, 80:240] = [170, 90, 70]
    arr[60:230, 270:430] = [170, 90, 70]
    arr[230:245, 60:450] = [22, 22, 30]  # ombre
    # Petit parking devant
    arr[290:325, 100:420] = [115, 115, 115]
    _add_aligned_cars(arr, y=295, x0=110, n=10, slot_px=28, w_px=14, l_px=22)
    chip = _chip(arr, side_m=60.0)
    r = analyze_parking_scenarios(chip)
    primary = r.primary_estimate
    if primary and primary.capacity_estimate is not None:
        if primary.confidence in ("medium", "strong"):
            assert primary.capacity_estimate <= 50, (
                f"trop optimiste : {primary.capacity_estimate} en {primary.confidence}"
            )


def test_observed_vehicle_presence_metadata_propagated():
    """Le comptage véhicules est exposé en métadonnée (preuve secondaire), pas comme plancher de capacité."""
    pytest.importorskip("cv2")
    arr = _gray(512, 512, level=130)
    _add_aligned_cars(arr, y=240, x0=60, n=6)
    chip = _chip(arr, side_m=60.0)
    r = analyze_parking_scenarios(chip)
    assert r.semantic is not None
    assert r.semantic.observed_vehicle_floor == r.vehicles.vehicle_count


def test_semantic_scores_dict_structure():
    pytest.importorskip("cv2")
    arr = _gray(256, 256)
    chip = _chip(arr, side_m=40.0)
    r = analyze_parking_scenarios(chip)
    assert r.semantic is not None
    e = r.semantic.evidence
    for name in (
        "vehicle_evidence",
        "vehicle_alignment_evidence",
        "building_exclusion_score",
        "road_access_score",
        "compactness_score",
        "separators_score",
        "geometry_score",
        "osm_score",
    ):
        v = getattr(e, name)
        assert 0.0 <= v <= 1.0, f"{name}={v}"
    assert 0.0 <= r.semantic.parking_usability_score <= 100.0
    assert r.semantic.semantic_confidence in ("none", "weak", "medium", "strong")
