"""Scores de fiabilité 0–100 (heuristiques produit)."""

from __future__ import annotations

from typing import Any, Optional

from parking_capacity.parking_geometry import GeometryParkingAnalysis


def _clamp01(x: float) -> float:
    return max(0.0, min(100.0, x))


def geometric_reliability_score(geo: Optional[GeometryParkingAnalysis]) -> float:
    if geo is None or geo.geometry_confidence == "none":
        return 0.0
    base = {"low": 35.0, "medium": 62.0, "high": 85.0}.get(geo.geometry_confidence, 10.0)
    base += 25.0 * float(geo.repeated_pattern_score)
    if geo.parking_structure_detected:
        base += 12.0
    return _clamp01(base)


def osm_reliability_score(
    *,
    has_tagged_capacity: bool,
    tagged_polygon_count: int,
    ban_score: float,
) -> float:
    s = 15.0
    if has_tagged_capacity:
        s += 55.0
    s += min(20.0, tagged_polygon_count * 4.0)
    s += max(0.0, (ban_score - 0.4) * 35.0)
    return _clamp01(s)


def visual_reliability_score(
    *,
    visual_evidence_level: str,
    specialized_parking: bool,
    segformer_ran: bool,
) -> float:
    if not segformer_ran:
        return 0.0
    if specialized_parking:
        m = {"none": 5.0, "weak": 30.0, "medium": 60.0, "strong": 85.0}
    else:
        # SegFormer générique = plafond faible (indice seulement)
        m = {"none": 0.0, "weak": 12.0, "medium": 22.0, "strong": 28.0}
    return _clamp01(m.get(visual_evidence_level, 0.0))


def ml_reliability_score(meta_ok: bool, meta: Optional[dict]) -> float:
    if not meta_ok or not meta:
        return 0.0
    try:
        n = int(meta.get("n_train_samples", 0))
        r2 = float(meta.get("val_r2", -99))
    except (TypeError, ValueError):
        return 5.0
    if n < 100 or r2 < 0:
        return 8.0
    s = 25.0 + min(40.0, n / 25.0) + max(0.0, min(35.0, r2 * 40.0))
    return _clamp01(s)


def capacity_range_quality_score(mid: Optional[int], mn: Optional[int], mx: Optional[int]) -> float:
    if mid is None or mn is None or mx is None or mid <= 0:
        return 0.0
    span = mx - mn
    if span <= 0:
        return 100.0
    ratio = span / max(mid, 1)
    # ratio 1 = excellent ; >4 = très mauvais
    q = 100.0 * max(0.0, 1.0 - max(0.0, ratio - 1.0) / 5.0)
    return _clamp01(q)


def overall_reliability_score(
    *,
    g: float,
    o: float,
    v: float,
    m: float,
    range_q: float,
) -> float:
    # pondération : OSM et géométrie > ML > vision générique
    overall = 0.34 * o + 0.32 * g + 0.18 * range_q + 0.10 * m + 0.06 * v
    return _clamp01(overall)
