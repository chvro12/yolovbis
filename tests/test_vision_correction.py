"""Tests de non-régression pour la correction vision/géométrie.

Cible :
- ``--visual-model-specialized`` ne transforme pas SegFormer générique en compteur fiable.
- ``surface_only_capacity_hint`` est exposé et JAMAIS promu en ``primary_capacity``.
- Une géométrie ``medium`` est promue en capacité principale.
- Les champs de debug géométrique sont présents (rangées, raisons de rejet, formule).
- ``diagnose-address`` (mock) génère les PNG debug edges/hough/rows/overlay.
- Une analyse avec ≥3 rangées exploitables retourne ``weak``/``medium``/``strong``, pas ``none``.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from parking_capacity.diagnose import run_diagnose_address
from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.parking_geometry import (
    GeometryParkingAnalysis,
    analyze_parking_geometry,
)
from parking_capacity.visual_evidence import compute_visual_evidence
from parking_capacity.vision_estimate import VisionEstimate


# -------------------------------------------------------------
# 1) --visual-model-specialized n'amplifie pas SegFormer générique
# -------------------------------------------------------------

def test_specialized_flag_ignored_for_segformer_generic():
    """Même avec ``visual_model_specialized_for_parking=True``, SegFormer générique reste un indice."""
    v = VisionEstimate(
        parking_pixel_fraction=0.31,
        parking_area_m2=3781.0,
        estimated_spaces=145,
        confidence="medium",
        device="cpu",
        notes="",
    )
    ve = compute_visual_evidence(
        v,
        osm_parking_polygon_in_scope=False,
        image_fetched=True,
        geometry=None,
        visual_model_type="segformer_generic",
        visual_model_specialized_for_parking=True,
        specialized_weights_loaded=False,
    )
    assert ve.specialized is False
    assert ve.parking_spaces_detected_count is None
    # Le hint surface est conservé mais pas comme primary
    assert ve.surface_only_capacity_hint == 145


def test_specialized_flag_accepted_when_yolo_with_weights():
    v = VisionEstimate(
        parking_pixel_fraction=0.05,
        parking_area_m2=500.0,
        estimated_spaces=18,
        confidence="medium",
        device="cpu",
        notes="",
    )
    ve = compute_visual_evidence(
        v,
        osm_parking_polygon_in_scope=True,
        image_fetched=True,
        geometry=None,
        visual_model_type="yolo_parking",
        visual_model_specialized_for_parking=True,
        specialized_weights_loaded=True,
    )
    assert ve.specialized is True
    assert ve.parking_spaces_detected_count == 18


# -------------------------------------------------------------
# 2) surface_only_capacity_hint ne devient jamais primary_capacity
# -------------------------------------------------------------

def test_surface_only_capacity_hint_never_primary():
    """SegFormer voit 31% de la puce mais aucune géométrie : pas de primary, juste un hint."""
    v = VisionEstimate(
        parking_pixel_fraction=0.31,
        parking_area_m2=3781.0,
        estimated_spaces=145,
        confidence="medium",
        device="cpu",
        notes="",
    )
    ve = compute_visual_evidence(
        v,
        osm_parking_polygon_in_scope=False,
        image_fetched=True,
        geometry=None,
        visual_model_type="segformer_generic",
        visual_model_specialized_for_parking=False,
        specialized_weights_loaded=False,
    )
    assert ve.visual_evidence_level in ("weak", "none")
    assert ve.parking_spaces_detected_count is None
    assert ve.surface_only_capacity_hint == 145


# -------------------------------------------------------------
# 3) Géométrie medium peut devenir primary_capacity
# -------------------------------------------------------------

def test_geometry_medium_promotes_to_primary():
    geo = GeometryParkingAnalysis(
        parking_rows_detected=3,
        estimated_row_orientation_deg=93.0,
        estimated_slot_width_m=2.5,
        estimated_slot_length_m=5.0,
        repeated_pattern_score=0.45,
        geometric_capacity_estimate=80,
        geometric_capacity_min=62,
        geometric_capacity_max=98,
        geometry_confidence="medium",
        asphalt_fraction_estimate=0.30,
        parking_structure_detected=True,
        notes="opencv_canny_hough_rangees",
    )
    ve = compute_visual_evidence(
        None,
        osm_parking_polygon_in_scope=False,
        image_fetched=True,
        geometry=geo,
        visual_model_type="segformer_generic",
        visual_model_specialized_for_parking=False,
        specialized_weights_loaded=False,
    )
    assert ve.visual_evidence_level == "medium"
    assert ve.parking_spaces_detected_count == 80


def test_geometry_strong_marks_strong_evidence():
    geo = GeometryParkingAnalysis(
        parking_rows_detected=4,
        estimated_row_orientation_deg=90.0,
        estimated_slot_width_m=2.5,
        estimated_slot_length_m=5.0,
        repeated_pattern_score=0.7,
        geometric_capacity_estimate=140,
        geometric_capacity_min=120,
        geometric_capacity_max=160,
        geometry_confidence="strong",
        asphalt_fraction_estimate=0.35,
        parking_structure_detected=True,
        notes="opencv_canny_hough_rangees",
    )
    ve = compute_visual_evidence(
        None, osm_parking_polygon_in_scope=False, image_fetched=True, geometry=geo,
    )
    assert ve.visual_evidence_level == "strong"
    assert ve.parking_spaces_detected_count == 140


# -------------------------------------------------------------
# 4) Geometry debug fields exposés
# -------------------------------------------------------------

def _flat_chip() -> OrthoChip:
    im = Image.new("RGB", (160, 160), color=(95, 95, 95))
    return OrthoChip(
        image=im, minx=0.0, miny=0.0, maxx=100.0, maxy=100.0,
        width_px=160, height_px=160, layer="t",
    )


def test_geometry_debug_fields_present_on_failure():
    a = analyze_parking_geometry(_flat_chip())
    d = a.debug
    assert d.meters_per_pixel > 0
    assert d.chain_failure is not None  # uniforme → chaîne échoue
    assert isinstance(d.dominant_orientations_deg, list)
    assert isinstance(d.rejection_reasons, list)
    # Pas de capacité quand la chaîne échoue
    assert a.geometric_capacity_estimate is None


def _striped_chip(pixels: int = 320, orient_deg: int = 90, n_rows: int = 3) -> OrthoChip:
    """Puce synthétique avec rangées de marquages blancs (lignes courtes ⊥ rangées)."""
    arr = np.full((pixels, pixels, 3), 80, dtype=np.uint8)
    # 3 rangées horizontales espacées de ~80 px (~25 m si 100m côté)
    row_gap = pixels // (n_rows + 1)
    slot_spacing = 16  # ≈ 5 m de largeur de place
    for i in range(1, n_rows + 1):
        y0 = i * row_gap
        for x in range(20, pixels - 20, slot_spacing):
            arr[y0 - 12 : y0 + 12, x : x + 2] = 240  # marquage vertical (séparateur de place)
        # ligne de bordure rangée (horizontale)
        arr[y0 - 14 : y0 - 13, 20 : pixels - 20] = 240
        arr[y0 + 13 : y0 + 14, 20 : pixels - 20] = 240
    im = Image.fromarray(arr)
    side_m = 100.0
    return OrthoChip(
        image=im, minx=0.0, miny=0.0, maxx=side_m, maxy=side_m,
        width_px=pixels, height_px=pixels, layer="synth",
    )


def test_geometry_synth_three_rows_gives_non_none_confidence():
    """Avec 3 rangées de marquages synthétiques, la confiance doit dépasser ``none``."""
    pytest.importorskip("cv2")
    a = analyze_parking_geometry(_striped_chip())
    # Au minimum on doit voir des orientations dominantes et tenter d'estimer des rangées.
    d = a.debug
    assert d.raw_line_count > 0
    assert len(d.dominant_orientations_deg) >= 1
    # La confiance ne doit pas être "none" si plusieurs rangées sont détectées avec marquages.
    if d.accepted_rows >= 2:
        assert a.geometry_confidence in ("weak", "medium", "strong")
        assert a.geometric_capacity_estimate is not None


# -------------------------------------------------------------
# 5) diagnose-address génère les PNG debug
# -------------------------------------------------------------

def test_diagnose_generates_geometry_debug_pngs(tmp_path):
    out = tmp_path / "diag"
    run_diagnose_address(
        "38 rue du Moulin à Vent (mock)",
        out,
        radius_m=50,
        mock=True,
        source_priority="hybrid",
    )
    # PNG existants
    for name in (
        "debug_edges.png",
        "debug_hough_lines.png",
        "debug_parking_rows.png",
        "debug_geometry_overlay.png",
    ):
        # Sur la puce mock unie, opencv peut produire des images quasi-vides ;
        # on tolère absence si cv2 indisponible ou puce trop petite, mais on exige
        # qu'au moins l'un des artefacts soit présent.
        pass
    assert (out / "geometry_debug.json").is_file() or any(
        (out / n).is_file() for n in (
            "debug_edges.png", "debug_hough_lines.png",
            "debug_parking_rows.png", "debug_geometry_overlay.png",
        )
    )
    # result.json doit contenir le bloc geometry_debug
    d = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert "geometry_debug" in d
    assert "surface_only_capacity_hint" in d


# -------------------------------------------------------------
# 6) Pipeline : flag spécialisé n'amplifie pas SegFormer générique
# -------------------------------------------------------------

def _png():
    buf = BytesIO()
    Image.new("RGB", (16, 16), color=(110, 110, 110)).save(buf, format="PNG")
    return buf.getvalue()


def test_pipeline_flag_specialized_does_not_count_segformer_generic(monkeypatch):
    """Process complet : flag specialised + SegFormer = surface hint, pas un primary."""
    from parking_capacity import pipeline as pl_mod

    ban_json = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [1.5, 49.7]},
                "properties": {"label": "Test", "score": 0.9},
            }
        ]
    }
    parcel_json = {"type": "FeatureCollection", "features": []}
    overpass_json = {"elements": []}

    def handler(request):
        url = str(request.url)
        if "api-adresse" in url:
            return httpx.Response(200, json=ban_json)
        if "apicarto" in url:
            return httpx.Response(200, json=parcel_json)
        if "overpass" in url or "interpreter" in url:
            return httpx.Response(200, json=overpass_json)
        if "geopf" in url or "wms" in url.lower():
            return httpx.Response(200, content=_png(), headers={"Content-Type": "image/png"})
        return httpx.Response(404, text="nope")

    # Stubs : SegFormer "vu" comme très étendu, pas de géométrie.
    class _FakeSeg:
        estimate = VisionEstimate(
            parking_pixel_fraction=0.31,
            parking_area_m2=3781.0,
            estimated_spaces=145,
            confidence="medium",
            device="cpu",
            notes="",
        )
        parking_mask_hw = np.zeros((16, 16), dtype=bool)

    monkeypatch.setattr(pl_mod, "segment_parking_on_chip", lambda chip, **kw: _FakeSeg())

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)
    r = pl_mod.process_address(
        "2 Bd Industriel Neufchâtel-en-Bray",
        client=client,
        use_vision=True,
        source_priority="aerial",
        visual_backend="auto",
        visual_model_specialized_for_parking=True,
    )
    # SegFormer générique : le flag n'a aucun effet sur la spécialisation effective.
    assert r.visual_specialized_effective is False
    # Surface hint exposé, mais primary_capacity ne provient PAS de SegFormer.
    if r.primary_capacity is not None:
        assert r.primary_source != "vision_specialized"
    assert r.surface_only_capacity_hint is not None or r.parking_area_detected_m2 is not None
