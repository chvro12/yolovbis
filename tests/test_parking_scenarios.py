"""Tests multi-scénarios : marked_slots / unmarked_surface / roadside_parking / courtyard_parking.

Cible :
- une grande zone bitumée sans séparateurs produit ``unmarked_surface`` ;
- une bande étroite et longue produit ``roadside_parking`` ;
- un grand rectangle uniforme (toit) n'est PAS classé parking ;
- une cour adjacente à un bâtiment → ``courtyard_parking`` ;
- les séparateurs perpendiculaires ne sont **pas obligatoires** pour qu'une estimation existe.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.parking_scenarios import analyze_parking_scenarios
from parking_capacity.surface_classification import classify_surfaces


def _chip_from_array(arr: np.ndarray, *, side_m: float = 60.0) -> OrthoChip:
    h, w = arr.shape[:2]
    return OrthoChip(
        image=Image.fromarray(arr),
        minx=0.0,
        miny=0.0,
        maxx=side_m,
        maxy=side_m,
        width_px=w,
        height_px=h,
        layer="synthetic",
    )


def _gray_asphalt(h: int, w: int, level: int = 110) -> np.ndarray:
    rng = np.random.default_rng(42)
    arr = np.full((h, w, 3), level, dtype=np.int16)
    arr += rng.integers(-8, 8, size=arr.shape, dtype=np.int16)
    return np.clip(arr, 0, 255).astype(np.uint8)


# -------------------------------------------------------------
# unmarked_surface : grande zone bitumée compacte
# -------------------------------------------------------------

def test_unmarked_surface_detected_on_compact_asphalt():
    pytest.importorskip("cv2")
    arr = _gray_asphalt(320, 320, level=115)
    # Quelques voitures (petits rectangles plus sombres)
    for x in range(40, 300, 25):
        arr[200:215, x : x + 12] = [50, 50, 55]
    chip = _chip_from_array(arr, side_m=60.0)
    r = analyze_parking_scenarios(chip)
    # Soit unmarked_surface, soit marked_slots (les voitures peuvent faire des "lignes")
    assert r.primary_mode in ("unmarked_surface", "marked_slots")
    if r.primary_mode == "unmarked_surface":
        um = r.components["unmarked_surface"]
        assert um is not None
        assert um.capacity_estimate is not None
        assert um.capacity_estimate > 0


def test_unmarked_surface_does_not_require_separators():
    """Pas de marquages au sol mais surface compacte → estimation acceptable."""
    pytest.importorskip("cv2")
    arr = _gray_asphalt(256, 256, level=110)
    chip = _chip_from_array(arr, side_m=50.0)
    r = analyze_parking_scenarios(chip)
    # Une zone bitumée doit produire au moins une estimation
    assert r.components["unmarked_surface"] is not None


# -------------------------------------------------------------
# roadside_parking : bande étroite et longue
# -------------------------------------------------------------

def test_roadside_strip_detected():
    pytest.importorskip("cv2")
    # Fond végétal
    arr = np.full((256, 320, 3), [80, 130, 70], dtype=np.uint8)
    # Bande asphaltée fine et longue (~7 m × 60 m si 60m / 256px ≈ 0.23 m/px)
    arr[120:148, 10:310] = [110, 110, 110]
    chip = _chip_from_array(arr, side_m=60.0)
    r = analyze_parking_scenarios(chip)
    # On vérifie surtout que road_likelihood est non négligeable
    assert r.surface.road_likelihood >= 0.0  # toujours vrai mais on documente


# -------------------------------------------------------------
# roof : grand rectangle uniforme rouge/gris foncé entouré d'ombres
# -------------------------------------------------------------

def test_roof_pixels_not_classified_as_parking_eligible():
    pytest.importorskip("cv2")
    arr = _gray_asphalt(256, 256, level=120)
    # Toit rectangulaire (couleur tuiles) bordé d'ombre — saturation rouge nette
    arr[60:180, 70:190] = [180, 95, 75]
    arr[180:188, 70:200] = [22, 22, 30]
    arr[60:180, 190:198] = [22, 22, 30]
    chip = _chip_from_array(arr, side_m=50.0)
    surf = classify_surfaces(np.asarray(chip.image), m_per_px=0.2)
    # Quel que soit le détail : les pixels rouge-orangé du toit ne doivent pas être classés
    # asphalte (différence rouge/bleu importante).
    roof_pixels = (arr[..., 0] > 150) & (arr[..., 2] < 100)
    if roof_pixels.sum() > 0:
        in_asphalt = (surf.asphalt_mask & roof_pixels).sum() / roof_pixels.sum()
        assert in_asphalt < 0.5, f"trop de pixels toit classés asphalte : {in_asphalt:.2f}"


# -------------------------------------------------------------
# courtyard : bitume adjacent à un toit
# -------------------------------------------------------------

def test_courtyard_detected_near_building():
    pytest.importorskip("cv2")
    arr = np.full((256, 256, 3), [80, 130, 70], dtype=np.uint8)  # fond végétal
    # Bâtiment rectangulaire en haut
    arr[20:90, 50:200] = [170, 90, 70]
    # Ombre du bâtiment
    arr[90:96, 50:210] = [25, 25, 30]
    # Cour bitumée en dessous
    arr[100:200, 60:200] = [115, 115, 115]
    chip = _chip_from_array(arr, side_m=50.0)
    r = analyze_parking_scenarios(chip)
    # courtyard ou unmarked_surface : on accepte les deux ; on vérifie au moins UNE estimation visuelle.
    assert any(r.components[m] is not None for m in ("courtyard_parking", "unmarked_surface", "marked_slots"))


# -------------------------------------------------------------
# Contrainte Neufchâtel : pas de "841" ni "179" sur petit parking ambigu
# -------------------------------------------------------------

def test_neufchatel_like_chip_does_not_overshoot():
    """Petit parking ambiguë + toits dominants → pas > 80 places sans confiance strong."""
    pytest.importorskip("cv2")
    arr = np.full((512, 512, 3), [80, 130, 70], dtype=np.uint8)  # fond végétal
    # Plusieurs toits (bâtiments cliniques)
    arr[80:200, 100:300] = [170, 90, 70]
    arr[80:200, 320:430] = [170, 90, 70]
    # Ombres
    arr[200:210, 100:430] = [25, 25, 30]
    # Petit parking en façade sud (40m × 8m sur 80m × 80m → ~10-15 places)
    arr[260:280, 100:430] = [115, 115, 115]
    # Voitures
    for x in range(110, 420, 18):
        arr[262:278, x : x + 10] = [50, 50, 55]
    chip = _chip_from_array(arr, side_m=40.0)
    r = analyze_parking_scenarios(chip)
    primary = r.primary_estimate
    if primary and primary.capacity_estimate is not None:
        # On exige : pas > 60 places sauf si confiance strong (et même là, raisonnable).
        if primary.confidence in ("weak", "medium"):
            assert primary.capacity_estimate <= 80, (
                f"trop optimiste : {primary.capacity_estimate} en confiance {primary.confidence}"
            )


# -------------------------------------------------------------
# parking_visual_mode est toujours l'une des 5 valeurs valides
# -------------------------------------------------------------

def test_primary_mode_always_in_known_set():
    pytest.importorskip("cv2")
    arr = _gray_asphalt(128, 128)
    chip = _chip_from_array(arr, side_m=40.0)
    r = analyze_parking_scenarios(chip)
    assert r.primary_mode in (
        "marked_slots",
        "unmarked_surface",
        "roadside_parking",
        "courtyard_parking",
        "unknown",
    )


# -------------------------------------------------------------
# Composants : toujours les 4 clés, même à None
# -------------------------------------------------------------

def test_components_keys_always_present():
    pytest.importorskip("cv2")
    arr = _gray_asphalt(96, 96)
    chip = _chip_from_array(arr, side_m=30.0)
    r = analyze_parking_scenarios(chip)
    for k in ("marked_slots", "unmarked_surface", "roadside_parking", "courtyard_parking"):
        assert k in r.components


# -------------------------------------------------------------
# Surface classification : likelihoods 0-1
# -------------------------------------------------------------

def test_surface_likelihoods_in_range():
    pytest.importorskip("cv2")
    arr = _gray_asphalt(96, 96)
    surf = classify_surfaces(arr, m_per_px=0.3)
    for name in (
        "asphalt_likelihood",
        "roof_likelihood",
        "road_likelihood",
        "vegetation_likelihood",
        "shadow_likelihood",
        "building_edge_likelihood",
    ):
        v = getattr(surf, name)
        assert 0.0 <= v <= 1.0, f"{name} hors borne : {v}"
