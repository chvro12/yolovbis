"""Réduction des fourchettes aberrantes (surface seule + signaux géométriques)."""

from __future__ import annotations

from typing import Optional, Tuple

from parking_capacity.parking_geometry import GeometryParkingAnalysis


def narrow_area_ratio_range(
    mid: int,
    mn: int,
    mx: int,
    *,
    geometry: Optional[GeometryParkingAnalysis],
    osm_polygon_count: int,
) -> Tuple[int, int, float]:
    """
    Retourne (min', max', range_quality_score amélioré après resserrement).
    """
    span = mx - mn
    rq = 100.0 * max(0.0, 1.0 - max(0.0, span / max(mid, 1) - 1.2) / 6.0)

    if mid <= 0:
        return mn, mx, 0.0

    # Limite supérieure : ne pas dépasser ~2.2x la médiane si géométrie faible
    cap_hi = int(mid * 2.2 + 25)
    cap_lo = max(0, int(mid * 0.45 - 5))

    if geometry and geometry.geometry_confidence in ("low", "medium"):
        asp = geometry.asphalt_fraction_estimate
        geo_cap = geometry.geometric_capacity_estimate
        cap_hi = min(cap_hi, int(mid + (geo_cap or mid) * (0.9 + asp) + 40))
        cap_lo = max(cap_lo, max(0, int((geo_cap or mid) * 0.35)) if geo_cap else cap_lo)

    if osm_polygon_count <= 0:
        cap_hi = min(cap_hi, int(mid * 1.95 + 30))

    new_mn = max(mn, cap_lo)
    new_mx = min(mx, max(new_mn + 5, cap_hi))
    span2 = new_mx - new_mn
    rq2 = 100.0 * max(0.0, 1.0 - max(0.0, span2 / max(mid, 1) - 1.0) / 5.0)
    return new_mn, new_mx, min(100.0, max(rq, rq2))
