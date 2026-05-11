"""Niveau de preuve visuelle.

Règle stricte :
- La spécialisation parking est dérivée de ``visual_model_type`` réel (yolo_parking, custom checkpoint
  avec métadonnées). Un flag CLI ``--visual-model-specialized`` ne peut **plus** transformer un
  SegFormer générique en « comptage de places fiable ».
- La capacité géométrique est la seule capacité visuelle considérée fiable (medium/strong).
- La surface SegFormer seule ne devient jamais une capacité principale : elle est exposée comme
  ``surface_only_capacity_hint`` (indice à vérifier manuellement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from parking_capacity.parking_geometry import GeometryParkingAnalysis
from parking_capacity.vision_estimate import VisionEstimate

try:
    from parking_capacity.parking_scenarios import ScenarioEstimate  # type: ignore
except ImportError:  # éviter cycle import
    ScenarioEstimate = None  # type: ignore

SEGFORMER_NOT_LINE_COUNTER = (
    "SegFormer générique (UTEL-UIUC) : segmentation « parking », pas un compteur de marquages au sol ; "
    "indice surface seulement, ne sert jamais de capacité principale."
)

_SPECIALIZED_MODEL_TYPES = ("yolo_parking", "custom_specialized")


def _is_truly_specialized(visual_model_type: str, has_loaded_weights: bool) -> bool:
    """Spécialisation = backend dédié parking ET poids effectivement chargés.

    Un SegFormer générique reste **non spécialisé** quel que soit le flag CLI.
    """
    return visual_model_type in _SPECIALIZED_MODEL_TYPES and has_loaded_weights


@dataclass
class VisualEvidenceResult:
    visual_evidence_level: str
    image_used: bool
    image_confidence: str
    parking_area_detected_m2: Optional[float]
    parking_spaces_detected_count: Optional[int]
    surface_only_capacity_hint: Optional[int]
    fallback_reason: str
    specialized: bool = False
    parking_visual_mode: str = "unknown"
    visual_capacity_min: Optional[int] = None
    visual_capacity_max: Optional[int] = None


def compute_visual_evidence(
    vision_est: Optional[VisionEstimate],
    *,
    osm_parking_polygon_in_scope: bool,
    image_fetched: bool,
    geometry: Optional[GeometryParkingAnalysis] = None,
    visual_model_type: str = "segformer_generic",
    visual_model_specialized_for_parking: bool = False,
    specialized_weights_loaded: bool = False,
    scenario_mode: str = "unknown",
    scenario_estimate: Optional[Any] = None,
) -> VisualEvidenceResult:
    """Calcule visual_evidence_level + capacité visuelle si réellement justifiée.

    ``visual_model_specialized_for_parking`` est seulement un *opt-in utilisateur* : il n'autorise la
    spécialisation que si ``visual_model_type`` est un backend dédié parking (YOLO / custom). Pour
    SegFormer générique, il est ignoré.
    """
    specialized = _is_truly_specialized(visual_model_type, specialized_weights_loaded) and \
        bool(visual_model_specialized_for_parking)

    # 0) Multi-scénarios : marked_slots / unmarked_surface / roadside / courtyard
    # On utilise scenario_estimate quand il existe et propose une vraie estimation.
    if (
        image_fetched
        and scenario_estimate is not None
        and getattr(scenario_estimate, "capacity_estimate", None) is not None
        and getattr(scenario_estimate, "confidence", "none") in ("weak", "medium", "strong")
    ):
        conf = scenario_estimate.confidence
        # marked_slots medium/strong = preuve principale ; les autres scénarios sortent en medium/weak
        mode = scenario_mode
        if mode == "marked_slots" and conf == "strong":
            level = "strong"
        elif mode == "marked_slots" and conf == "medium":
            level = "medium"
        elif mode == "unmarked_surface" and conf == "medium":
            level = "medium"
        elif mode == "roadside_parking" and conf == "medium":
            level = "medium"
        elif mode == "courtyard_parking" and conf == "medium":
            level = "weak"
        else:
            level = "weak"
        img_conf = "medium" if conf in ("medium", "strong") else "low"
        return VisualEvidenceResult(
            visual_evidence_level=level,
            image_used=True,
            image_confidence=img_conf,
            parking_area_detected_m2=None,
            parking_spaces_detected_count=int(scenario_estimate.capacity_estimate),
            surface_only_capacity_hint=None,
            fallback_reason=f"scenario_{mode}_{conf}",
            specialized=specialized,
            parking_visual_mode=mode,
            visual_capacity_min=scenario_estimate.capacity_min,
            visual_capacity_max=scenario_estimate.capacity_max,
        )

    # 1) Géométrie dominante si structure détectée (priorité absolue)
    if image_fetched and geometry and geometry.geometric_capacity_estimate is not None:
        if geometry.geometry_confidence == "strong":
            return VisualEvidenceResult(
                visual_evidence_level="strong",
                image_used=True,
                image_confidence="medium",
                parking_area_detected_m2=None,
                parking_spaces_detected_count=int(geometry.geometric_capacity_estimate),
                surface_only_capacity_hint=None,
                fallback_reason="geometrie_rangees_regulieres_strong",
                specialized=specialized,
            )
        if geometry.geometry_confidence == "medium":
            return VisualEvidenceResult(
                visual_evidence_level="medium",
                image_used=True,
                image_confidence="medium",
                parking_area_detected_m2=None,
                parking_spaces_detected_count=int(geometry.geometric_capacity_estimate),
                surface_only_capacity_hint=None,
                fallback_reason="geometrie_rangees_regulieres_medium",
                specialized=specialized,
            )
        if geometry.geometry_confidence == "weak":
            # Géométrie faible : on conserve la valeur géométrique comme hint, pas comme primary fiable.
            return VisualEvidenceResult(
                visual_evidence_level="weak",
                image_used=True,
                image_confidence="low",
                parking_area_detected_m2=None,
                parking_spaces_detected_count=int(geometry.geometric_capacity_estimate),
                surface_only_capacity_hint=None,
                fallback_reason="geometrie_rangees_faibles",
                specialized=specialized,
            )

    # 2) SegFormer / autres : surface = indice seulement
    if not image_fetched:
        return VisualEvidenceResult(
            "none", False, "low", None, None, None, "orthophoto_non_téléchargée", specialized,
        )
    if vision_est is None:
        return VisualEvidenceResult(
            "none", True, "low", None, None, None, "vision_indisponible", specialized,
        )

    frac = vision_est.parking_pixel_fraction
    area = float(vision_est.parking_area_m2)
    est = int(vision_est.estimated_spaces)

    # Spécialisation réelle (YOLO parking avec poids) : comptage autorisé
    if specialized:
        return _specialized_evidence(vision_est, osm_parking_polygon_in_scope, frac, area, est)

    # SegFormer générique : la surface est un *indice* (jamais un primary), même si l'utilisateur a coché le flag.
    if frac < 0.003:
        return VisualEvidenceResult(
            "none", True, "low", None, None, None,
            "segformer_fraction_négligeable", specialized,
        )
    if frac < 0.012:
        return VisualEvidenceResult(
            "weak", True, "low", area, None, est if est > 0 else None,
            "segformer_indice_faible_surface_seule", specialized,
        )
    # Fraction notable : on remonte l'aire détectée comme surface_only_capacity_hint,
    # **jamais** comme parking_spaces_detected_count.
    return VisualEvidenceResult(
        "weak", True, "low", area, None, est if est > 0 else None,
        "segformer_surface_only_capacity_hint_non_specialise", specialized,
    )


def _specialized_evidence(
    vision_est: VisionEstimate,
    osm_poly: bool,
    frac: float,
    area: float,
    est: int,
) -> VisualEvidenceResult:
    if frac < 0.003:
        return VisualEvidenceResult(
            "none", True, "low", None, None, None, "specialise_fraction_negligeable", True,
        )
    if frac < 0.012:
        return VisualEvidenceResult(
            "weak", True, "low", area, None, est if est > 0 else None,
            "specialise_preuve_visuelle_faible", True,
        )
    if frac < 0.04:
        return VisualEvidenceResult(
            "medium", True, "medium" if osm_poly else "low", area, est, None,
            "specialise_aire_avec_polygone_osm" if osm_poly else "specialise_aire", True,
        )
    if osm_poly:
        return VisualEvidenceResult(
            "strong", True, "medium", area, est, None, "specialise_masque_et_osm", True,
        )
    return VisualEvidenceResult(
        "medium", True, "low", area, est, None, "specialise_masque_sans_polygone_osm", True,
    )


def vision_notes_with_disclaimer(vision_est: Optional[VisionEstimate]) -> str:
    if vision_est is None:
        return SEGFORMER_NOT_LINE_COUNTER
    return f"{vision_est.notes} {SEGFORMER_NOT_LINE_COUNTER}"
