"""Fumée : analyse géométrique sur une puce synthétique + structures de debug exposées."""

from __future__ import annotations

from PIL import Image

from parking_capacity.imagery_wms import OrthoChip
from parking_capacity.parking_geometry import (
    GeometryDebug,
    GeometryParkingAnalysis,
    analyze_parking_geometry,
    merge_geometry_analyses,
)


def _chip(side_m: float = 100.0, pixels: int = 128) -> OrthoChip:
    im = Image.new("RGB", (pixels, pixels), color=(90, 92, 88))
    return OrthoChip(
        image=im,
        minx=0.0,
        miny=0.0,
        maxx=side_m,
        maxy=side_m,
        width_px=pixels,
        height_px=pixels,
        layer="ORTHO",
    )


def test_analyze_parking_geometry_returns_dataclass():
    g = analyze_parking_geometry(_chip())
    assert g.geometry_confidence in ("none", "weak", "medium", "strong")
    assert isinstance(g.estimated_row_orientation_deg, (int, float))
    assert g.parking_rows_detected >= 0
    assert isinstance(g.debug, GeometryDebug)


def test_analyze_parking_geometry_exposes_debug_fields():
    g = analyze_parking_geometry(_chip())
    d = g.debug
    assert hasattr(d, "meters_per_pixel")
    assert hasattr(d, "raw_line_count")
    assert hasattr(d, "filtered_line_count")
    assert hasattr(d, "usable_line_count")
    assert hasattr(d, "dominant_orientations_deg")
    assert hasattr(d, "row_candidates")
    assert hasattr(d, "accepted_rows")
    assert hasattr(d, "rejected_rows")
    assert hasattr(d, "rejection_reasons")
    assert hasattr(d, "capacity_formula_used")
    # Sur une puce plate uniforme : chaîne échoue, capacity reste None.
    assert g.geometric_capacity_estimate is None
    assert d.chain_failure is not None


def _make_analysis(conf: str, cap: int | None, rep: float = 0.3, rows: int = 2) -> GeometryParkingAnalysis:
    return GeometryParkingAnalysis(
        parking_rows_detected=rows,
        estimated_row_orientation_deg=0.0,
        estimated_slot_width_m=2.5,
        estimated_slot_length_m=5.0,
        repeated_pattern_score=rep,
        geometric_capacity_estimate=cap,
        geometric_capacity_min=(cap and int(cap * 0.8)) or None,
        geometric_capacity_max=(cap and int(cap * 1.2)) or None,
        geometry_confidence=conf,
        asphalt_fraction_estimate=0.3,
        parking_structure_detected=conf in ("medium", "strong"),
        notes="test",
    )


def test_merge_geometry_prefers_higher_confidence():
    low = _make_analysis("weak", 10, rep=0.1, rows=1)
    med = _make_analysis("medium", 40, rep=0.4, rows=2)
    assert merge_geometry_analyses(low, med) is med
    assert merge_geometry_analyses(med, low) is med


def test_analyze_returns_min_max_when_capacity_present():
    a = _make_analysis("medium", 40)
    assert a.geometric_capacity_min is not None
    assert a.geometric_capacity_max is not None
    assert a.geometric_capacity_min <= 40 <= a.geometric_capacity_max
