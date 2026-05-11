"""Estimation baseline (non-ML) : OSM, surface, places individuelles, vision, ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BaselineSignals:
    osm_capacity_tagged: int
    osm_parking_space_count: int
    area_mid_places: int
    area_min_places: int
    area_max_places: int
    area_total_m2: float
    vision_places: Optional[int]
    ml_places: Optional[int]


def choose_baseline_capacity(sig: BaselineSignals) -> tuple[Optional[int], str, str]:
    """
    Retourne (capacité entière ou None, méthode courte, note).
    Priorité : tag OSM > parking_space > surface > vision > ML.
    """
    if sig.osm_capacity_tagged > 0:
        return sig.osm_capacity_tagged, "osm_capacity", "Somme des tags capacity OSM fiables dans la zone."
    if sig.osm_parking_space_count > 0:
        return sig.osm_parking_space_count, "osm_parking_space_count", "Comptage amenity=parking_space OSM."
    if sig.area_mid_places > 0 and sig.area_total_m2 >= 80:
        return sig.area_mid_places, "area_ratio", f"Surface parking sans tag ~{sig.area_total_m2:.0f} m² / 28 m²·place⁻¹."
    if sig.vision_places is not None and sig.vision_places > 0:
        return sig.vision_places, "vision", "SegFormer + heuristique m²/place sur orthophoto."
    if sig.ml_places is not None and sig.ml_places > 0:
        return sig.ml_places, "ml", "Régression sur puce orthophoto (checkpoint entraîné)."
    return None, "none", "Aucune source suffisante."


def compare_ml_to_baseline(ml_raw: Optional[float], baseline: Optional[int]) -> str:
    """Message honnête si le ML diverge fortement de la baseline."""
    if ml_raw is None or baseline is None or baseline <= 0:
        return ""
    ml_i = max(0, int(round(ml_raw)))
    rel = abs(ml_i - baseline) / max(baseline, 1)
    if rel > 0.5:
        return (
            f"Attention : le ML ({ml_i}) diffère de plus de 50 % de la baseline non-ML ({baseline}) ; "
            "ne pas faire confiance au ML sans validation terrain ou métriques d’évaluation."
        )
    return ""
