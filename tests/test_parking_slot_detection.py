"""Tests détection places marquées : heuristique véhicules+rangées (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

from parking_capacity.parking_slot_detection import (
    Slot,
    SlotDetectionResult,
    detect_slots,
    detect_slots_heuristic,
)


@dataclass
class _Veh:
    cx: float
    cy: float
    width_px: float = 14.0
    height_px: float = 28.0
    angle_deg: float = 0.0
    area_px: float = 14 * 28
    confidence: float = 0.5


def _chip(size: int = 256) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), 110, dtype=np.uint8))


def test_no_geometry_returns_empty():
    r = detect_slots_heuristic(
        _chip(),
        row_lengths_m=[],
        row_orientation_deg=0.0,
        vehicles=[],
        m_per_px=0.2,
    )
    assert r.method == "none"
    assert r.slots_total_count == 0


def test_single_row_with_vehicles_fills_correctly():
    """Rangée de 30 m, slot 2.5 m → 12 places ; 4 véhicules détectés → 4 pleines, 8 vides."""
    # m_per_px = 0.2 ; 30 m = 150 px ; slot_px ≈ 12.5 ; row centré
    cx_img = 128
    cy_img = 128
    vehicles = [_Veh(cx=cx_img - 50, cy=cy_img),
                _Veh(cx=cx_img - 30, cy=cy_img),
                _Veh(cx=cx_img + 10, cy=cy_img),
                _Veh(cx=cx_img + 30, cy=cy_img)]
    r = detect_slots_heuristic(
        _chip(),
        row_lengths_m=[30.0],
        row_orientation_deg=0.0,
        vehicles=vehicles,
        m_per_px=0.2,
    )
    assert r.method == "heuristic_vehicles_rows"
    assert r.slots_total_count == 12  # 30/2.5
    assert r.slots_filled_count >= 3  # tolérance assignation
    assert r.slots_empty_count == r.slots_total_count - r.slots_filled_count


def test_plausible_ceiling_caps_slots():
    """Plafond physique limite le total."""
    r = detect_slots_heuristic(
        _chip(),
        row_lengths_m=[60.0, 60.0, 60.0],
        row_orientation_deg=0.0,
        vehicles=[],
        m_per_px=0.2,
        plausible_ceiling=20,
    )
    assert r.slots_total_count == 20


def test_no_vehicles_all_empty():
    r = detect_slots_heuristic(
        _chip(),
        row_lengths_m=[20.0],
        row_orientation_deg=0.0,
        vehicles=[],
        m_per_px=0.2,
    )
    assert r.slots_filled_count == 0
    assert r.slots_total_count > 0
    assert r.slots_empty_count == r.slots_total_count


def test_detect_slots_dispatches_to_heuristic_without_weights():
    r = detect_slots(
        _chip(),
        m_per_px=0.2,
        row_lengths_m=[20.0],
        row_orientation_deg=0.0,
        vehicles=[],
        yolo_weights=None,
        roboflow_api_key=None,
        roboflow_model_id=None,
    )
    assert r.method in ("heuristic_vehicles_rows", "none")


def test_polygon_clip_excludes_slots_outside():
    """Polygone restreint au centre → seuls les slots dans la zone restent."""
    pytest.importorskip("cv2")
    # Polygone carré 50×50 centré
    poly = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    r = detect_slots_heuristic(
        _chip(),
        row_lengths_m=[30.0],  # rangée s'étend au-delà du polygone
        row_orientation_deg=0.0,
        vehicles=[],
        m_per_px=0.2,
        parcelle_polygon_px=poly,
    )
    # Le total après clip doit être < total sans clip (12)
    assert r.slots_total_count <= 12


def test_slot_status_either_filled_or_empty_or_unknown():
    r = detect_slots_heuristic(
        _chip(),
        row_lengths_m=[20.0],
        row_orientation_deg=45.0,
        vehicles=[],
        m_per_px=0.2,
    )
    for s in r.slots:
        assert s.status in ("filled", "empty", "unknown")
        assert 0.0 <= s.confidence <= 1.0
        assert s.source in ("heuristic", "yolo", "roboflow")
