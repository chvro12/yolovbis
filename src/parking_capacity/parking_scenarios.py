"""Estimation multi-scénarios : marked_slots / unmarked_surface / roadside_parking / courtyard_parking.

Beaucoup de parkings réels n'ont pas de marquage clair (cours industrielles, bord de chaussée,
parkings gravier). Cette couche évalue **chaque mode en parallèle** et choisit le plus plausible
selon la classification de surface, sans rendre les séparateurs perpendiculaires obligatoires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from parking_capacity.gis_fusion import GisFusionResult, compute_fusion_area_metrics
from parking_capacity.imagery_wms import OrthoChip, chip_m2_per_pixel
from parking_capacity.parking_geometry import (
    GeometryParkingAnalysis,
    analyze_parking_geometry,
    merge_geometry_analyses,
)
from parking_capacity.semantic_layer import (
    SemanticContext,
    clamp_capacity_to_semantic_bounds,
    compute_semantic_context,
)
from parking_capacity.site_classification import (
    m2_per_space_courtyard_range,
    m2_per_space_unmarked_range,
)
from parking_capacity.surface_classification import SurfaceClassification, classify_surfaces
from parking_capacity.vehicle_detection import VehicleDetectionResult, detect_vehicles

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


MODES = (
    "marked_slots",
    "unmarked_surface",
    "roadside_parking",
    "courtyard_parking",
    "unknown",
)

# Facteur de surface utile (places / circulation)
SMALL_LOT_USABLE_LO = 0.55
SMALL_LOT_USABLE_HI = 0.70
COURTYARD_USABLE_LO = 0.35
COURTYARD_USABLE_HI = 0.55
M2_PER_SPACE_LO = 25.0
M2_PER_SPACE_HI = 32.0
ROADSIDE_SLOT_LENGTH_LO_M = 5.0
ROADSIDE_SLOT_LENGTH_HI_M = 5.5


@dataclass
class ScenarioEstimate:
    """Une estimation par mode de stationnement détecté."""

    mode: str
    capacity_estimate: Optional[int]
    capacity_min: Optional[int]
    capacity_max: Optional[int]
    confidence: str  # none | weak | medium | strong
    notes: str = ""
    extras: dict = field(default_factory=dict)


@dataclass
class MultiScenarioResult:
    primary_mode: str
    primary_estimate: Optional[ScenarioEstimate]
    components: Dict[str, Optional[ScenarioEstimate]]
    surface: SurfaceClassification
    geometry: Optional[GeometryParkingAnalysis]
    notes: List[str] = field(default_factory=list)
    vehicles: Optional[VehicleDetectionResult] = None
    semantic: Optional[SemanticContext] = None
    gis_fusion: Optional[GisFusionResult] = None
    fusion_excluded_building_area_m2: Optional[float] = None
    fusion_usable_parking_area_m2: Optional[float] = None
    fusion_final_candidate_area_m2: Optional[float] = None
    fusion_final_parking_candidate_mask: Optional[np.ndarray] = None


# -----------------------------
# Marked slots (via géométrie)
# -----------------------------

def _scenario_marked_slots(
    chip: OrthoChip,
    surface: SurfaceClassification,
    *,
    segformer_mask: Optional[np.ndarray],
    m_per_px: float,
) -> Tuple[Optional[ScenarioEstimate], Optional[GeometryParkingAnalysis]]:
    """Délègue à la chaîne géométrique avec roof_mask + cap longueur, puis applique un
    sanity-check par densité : si capacité / surface éligible < ~15 m²/place, on dégrade
    voire on rejette la valeur (signe que des bordures de toits ont été comptées).
    """
    geo_full = analyze_parking_geometry(
        chip,
        roof_mask=surface.roof_mask,
        max_row_length_m=45.0,
        require_separators=False,
    )
    geo_roi = None
    if segformer_mask is not None and segformer_mask.shape[:2] == (chip.height_px, chip.width_px):
        if float(segformer_mask.astype(np.float32).mean()) >= 0.02:
            geo_roi = analyze_parking_geometry(
                chip,
                segformer_roi_mask=segformer_mask,
                roof_mask=surface.roof_mask,
                max_row_length_m=45.0,
                require_separators=False,
            )

    geo = merge_geometry_analyses(geo_full, geo_roi)
    if geo is None or geo.geometric_capacity_estimate is None:
        return None, geo

    if geo.geometry_confidence == "none":
        return None, geo

    cap = int(geo.geometric_capacity_estimate)
    cap_min = int(geo.geometric_capacity_min) if geo.geometric_capacity_min else None
    cap_max = int(geo.geometric_capacity_max) if geo.geometric_capacity_max else None
    conf = geo.geometry_confidence
    sanity_notes: List[str] = []

    # Sanity-check 1 : densité (m² par place estimée) vs surface bitumée éligible.
    # Une place de parking marquée occupe **avec circulation** ~22-32 m². En dessous c'est
    # implausible (on a compté des bordures de toits / voirie comme rangées).
    eligible_m2 = float(surface.parking_eligible_mask.sum()) * (m_per_px ** 2)
    if eligible_m2 > 50.0 and cap > 0:
        m2_per_space = eligible_m2 / cap
        if m2_per_space < 10.0:
            # Totalement implausible : on rejette la valeur marked_slots.
            sanity_notes.append(f"densite_implausible_{m2_per_space:.1f}m2_par_place")
            return None, geo
        if m2_per_space < 18.0:
            # Très dense : downgrade fort.
            conf = "weak"
            sanity_notes.append(f"densite_basse_{m2_per_space:.1f}m2_par_place")
        elif m2_per_space < 25.0:
            # Limite basse de plausibilité : on dégrade d'un cran.
            if conf == "strong":
                conf = "medium"
            elif conf == "medium":
                conf = "weak"
            sanity_notes.append(f"densite_limite_{m2_per_space:.1f}m2_par_place")

    # Sanity-check 2 : trop de rangées par rapport à la surface (ratio rangées/100m²)
    if eligible_m2 > 50.0:
        rows_per_100m2 = geo.parking_rows_detected / (eligible_m2 / 100.0)
        if rows_per_100m2 > 1.5 and conf in ("medium", "strong"):
            conf = "weak"
            sanity_notes.append("trop_de_rangees_pour_surface")

    # Sanity-check 3 : rangées trop longues (> 40m) doivent être ultra majoritaires en séparateurs.
    row_lengths = list(geo.row_lengths_m)
    if row_lengths:
        long_ratio = sum(1 for L in row_lengths if L > 40.0) / len(row_lengths)
        if long_ratio > 0.5 and conf in ("medium", "strong"):
            # Majoritairement des rangées longues : probable confusion avec bordures architecturales.
            conf = "weak"
            sanity_notes.append(f"rangees_longues_majoritaires_{long_ratio:.0%}")

    notes = geo.notes
    if sanity_notes:
        notes = f"{notes} | sanity: {','.join(sanity_notes)}"

    est = ScenarioEstimate(
        mode="marked_slots",
        capacity_estimate=cap,
        capacity_min=cap_min,
        capacity_max=cap_max,
        confidence=conf,
        notes=notes,
        extras={
            "parking_rows_detected": geo.parking_rows_detected,
            "row_lengths_m": list(geo.row_lengths_m),
            "repeated_pattern_score": geo.repeated_pattern_score,
            "estimated_row_orientation_deg": geo.estimated_row_orientation_deg,
            "eligible_area_m2": round(eligible_m2, 1),
            "m2_per_estimated_space": round(eligible_m2 / max(cap, 1), 1) if eligible_m2 > 0 else None,
            "sanity_notes": sanity_notes,
        },
    )
    return est, geo


# -----------------------------
# Unmarked surface (bitume libre garable)
# -----------------------------

def _largest_connected_area_m2(mask: np.ndarray, m_per_px: float) -> Tuple[float, float, Optional[Tuple[int, int, int, int]]]:
    """Retourne (aire_largest_m2, total_eligible_m2, bbox_largest)."""
    if not _HAS_CV2 or mask.dtype != np.bool_:
        return 0.0, float(mask.sum()) * (m_per_px ** 2), None
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    total_eligible_m2 = float(mask.sum()) * (m_per_px ** 2)
    if n <= 1:
        return 0.0, total_eligible_m2, None
    # Plus grande composante (hors fond 0)
    largest = max(range(1, n), key=lambda i: stats[i, 4])
    area_m2 = float(stats[largest, 4]) * (m_per_px ** 2)
    bbox = (
        int(stats[largest, 0]),
        int(stats[largest, 1]),
        int(stats[largest, 2]),
        int(stats[largest, 3]),
    )
    return area_m2, total_eligible_m2, bbox


def _adjacency_to_road(mask: np.ndarray, road_mask: np.ndarray) -> float:
    """Mesure le contact entre le masque garable et la chaussée (0-1)."""
    if not _HAS_CV2 or mask.dtype != np.bool_ or road_mask.dtype != np.bool_:
        return 0.0
    if mask.sum() == 0 or road_mask.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(np.bool_)
    contact = (dilated & road_mask).sum()
    return float(np.clip(contact / max(mask.sum() * 0.05, 1.0), 0.0, 1.0))


def _scenario_unmarked_surface(
    chip: OrthoChip,
    surface: SurfaceClassification,
    *,
    m_per_px: float,
    site_type: str = "unknown",
) -> Optional[ScenarioEstimate]:
    """Zone bitumée compacte non marquée : capacité théorique par surface utile (hors circulation)."""
    eligible = surface.parking_eligible_mask
    largest_m2, total_m2, _ = _largest_connected_area_m2(eligible, m_per_px)
    if largest_m2 < 60.0:
        return None
    # Adjacence à la chaussée → accessibilité
    adj = _adjacency_to_road(eligible, surface.road_mask)

    # Allées / circulation : on ne compte pas 100 % de la plus grande composante comme places.
    circulation_factor = 0.86
    largest_net_m2 = largest_m2 * circulation_factor

    # Facteur d'utilisation : plus la zone est compacte, plus on garde de surface.
    if largest_m2 < 400.0:
        u_lo, u_hi = SMALL_LOT_USABLE_LO, SMALL_LOT_USABLE_HI
        confidence = "medium" if adj > 0.15 else "weak"
    elif largest_m2 < 2500.0:
        u_lo, u_hi = 0.45, 0.65
        confidence = "medium" if adj > 0.15 else "weak"
    else:
        # Très grande zone : risque d'inclure des éléments non-parking
        u_lo, u_hi = 0.35, 0.55
        confidence = "weak"

    m_lo, m_hi = m2_per_space_unmarked_range(site_type)
    usable_lo_m2 = largest_net_m2 * u_lo
    usable_hi_m2 = largest_net_m2 * u_hi
    cap_min = max(1, int(round(usable_lo_m2 / m_hi)))
    cap_max = int(round(usable_hi_m2 / m_lo)) + 1
    cap_est = int(round((cap_min + cap_max) / 2.0))

    return ScenarioEstimate(
        mode="unmarked_surface",
        capacity_estimate=cap_est,
        capacity_min=cap_min,
        capacity_max=cap_max,
        confidence=confidence,
        notes="zone_bitumee_compacte_sans_marquage",
        extras={
            "unmarked_area_m2": round(largest_m2, 1),
            "total_eligible_area_m2": round(total_m2, 1),
            "usable_area_factor_lo": u_lo,
            "usable_area_factor_hi": u_hi,
            "adjacency_to_road": round(adj, 3),
            "circulation_factor": circulation_factor,
            "site_type": site_type,
            "m2_per_space_range": [m_lo, m_hi],
        },
    )


# -----------------------------
# Roadside parking
# -----------------------------

def _road_length_m(road_mask: np.ndarray, m_per_px: float) -> float:
    """Approximation longueur cumulée bandes de chaussée (sqrt aire / largeur approx)."""
    if not _HAS_CV2 or road_mask.dtype != np.bool_:
        return 0.0
    m = road_mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    total_len_m = 0.0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        long_side_m = max(ww, hh) * m_per_px
        short_side_m = max(min(ww, hh) * m_per_px, 1e-6)
        if long_side_m < 8.0 or short_side_m > 12.0:
            continue
        total_len_m += long_side_m
    return total_len_m


def _scenario_roadside_parking(
    chip: OrthoChip,
    surface: SurfaceClassification,
    *,
    m_per_px: float,
) -> Optional[ScenarioEstimate]:
    """Stationnement linéaire : longueur utile / 5–5,5 m (entrées / carrefours non modélisés → fraction prudente)."""
    road_len = _road_length_m(surface.road_mask, m_per_px)
    if road_len < 10.0:
        return None
    # Chaussée large détectée → moins de linéaire exploitable (virages, traversées piétons non séparés).
    road_frac = float(surface.road_mask.mean()) if surface.road_mask.size else 0.0
    usable_fraction = 0.6 if road_len < 40.0 else 0.45
    if road_frac > 0.35:
        usable_fraction *= 0.88
    usable_len = road_len * usable_fraction
    cap_min = max(0, int(round(usable_len / ROADSIDE_SLOT_LENGTH_HI_M)))
    cap_max = int(round(usable_len / ROADSIDE_SLOT_LENGTH_LO_M)) + 1
    cap_est = int(round((cap_min + cap_max) / 2.0))
    if cap_max <= 0:
        return None
    confidence = "weak"
    if road_len >= 30.0 and usable_fraction >= 0.55:
        confidence = "medium"
    return ScenarioEstimate(
        mode="roadside_parking",
        capacity_estimate=cap_est,
        capacity_min=cap_min,
        capacity_max=cap_max,
        confidence=confidence,
        notes="stationnement_bord_chaussee_estime",
        extras={
            "roadside_length_m": round(road_len, 1),
            "roadside_usable_fraction": usable_fraction,
            "entrances_crossings_model_note": "fraction_prudente_sans_carte_entrées_piétons",
        },
    )


# -----------------------------
# Courtyard parking
# -----------------------------

def _scenario_courtyard_parking(
    chip: OrthoChip,
    surface: SurfaceClassification,
    *,
    m_per_px: float,
    site_type: str = "unknown",
) -> Optional[ScenarioEstimate]:
    """Cour / garage / zone privée : surface bitumée adjacente à un bâtiment (ratios selon ``site_type``)."""
    if surface.roof_mask.sum() < 50:
        return None
    eligible = surface.parking_eligible_mask
    if not _HAS_CV2 or eligible.sum() < 50:
        return None
    # Dilate les toits et garde l'intersection avec eligible : zone bitumée touchant un bâtiment.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    near_roof = cv2.dilate(surface.roof_mask.astype(np.uint8), k, iterations=2).astype(np.bool_)
    courtyard_mask = eligible & near_roof
    if courtyard_mask.sum() < 50:
        return None
    area_m2 = float(courtyard_mask.sum()) * (m_per_px ** 2)
    if area_m2 < 80.0:
        return None
    u_lo, u_hi = COURTYARD_USABLE_LO, COURTYARD_USABLE_HI
    m_lo, m_hi = m2_per_space_courtyard_range(site_type)
    usable_lo_m2 = area_m2 * u_lo
    usable_hi_m2 = area_m2 * u_hi
    cap_min = max(1, int(round(usable_lo_m2 / m_hi)))
    cap_max = int(round(usable_hi_m2 / m_lo)) + 1
    cap_est = int(round((cap_min + cap_max) / 2.0))
    # Confiance : medium si zone significative ET accessibilité, sinon weak
    adj = _adjacency_to_road(courtyard_mask, surface.road_mask)
    confidence = "medium" if area_m2 > 250.0 and adj > 0.10 else "weak"
    return ScenarioEstimate(
        mode="courtyard_parking",
        capacity_estimate=cap_est,
        capacity_min=cap_min,
        capacity_max=cap_max,
        confidence=confidence,
        notes="cour_industrielle_ou_arriere_batiment",
        extras={
            "courtyard_area_m2": round(area_m2, 1),
            "usable_area_factor_lo": u_lo,
            "usable_area_factor_hi": u_hi,
            "adjacency_to_road": round(adj, 3),
            "site_type": site_type,
            "m2_per_space_range": [m_lo, m_hi],
        },
    )


# -----------------------------
# Choix du primary
# -----------------------------

_CONF_RANK = {"strong": 4, "high": 4, "medium": 3, "low": 2, "weak": 1, "none": 0}


def _pick_primary(components: Dict[str, Optional[ScenarioEstimate]]) -> Tuple[str, Optional[ScenarioEstimate]]:
    """Hiérarchie scénario (à OSM-fiable près, géré ailleurs) :
    marked_slots > unmarked_surface > roadside_parking > courtyard_parking.
    En cas d'égalité de confiance, on garde l'ordre ci-dessus.
    """
    order = ("marked_slots", "unmarked_surface", "roadside_parking", "courtyard_parking")

    best_mode = "unknown"
    best: Optional[ScenarioEstimate] = None
    best_rank = -1
    for mode in order:
        s = components.get(mode)
        if s is None or s.capacity_estimate is None or s.confidence == "none":
            continue
        r = _CONF_RANK.get(s.confidence, 0)
        if r > best_rank or (r == best_rank and best is None):
            best_rank = r
            best = s
            best_mode = mode
    return best_mode, best


def analyze_parking_scenarios(
    chip: OrthoChip,
    *,
    segformer_parking_mask: Optional[np.ndarray] = None,
    yolo_vehicle_weights=None,
    has_osm_capacity: bool = False,
    bdtopo_buildings_mask: Optional[np.ndarray] = None,
    gis_augmentation: Optional[GisFusionResult] = None,
    max_plausible_slots_cap: int = 39,
    site_type: str = "unknown",
) -> MultiScenarioResult:
    """``parking_capacity_estimation`` : surface → scénarios théoriques → preuves véhicules (secondaires) → sémantique.

    Les véhicules détectés ne fixent **pas** la capacité principale ; ils alimentent seulement la confiance
    et les métadonnées ``supporting_evidence`` (voir pipeline / ``result.json``).
    Le plafond physique limite les estimations ; **aucun plancher** n’est imposé à partir du seul comptage
    de véhicules (évite de confondre occupation instantanée et capacité théorique).
    """
    rgb = np.asarray(chip.image.convert("RGB"), dtype=np.uint8)
    m_per_px = math.sqrt(max(chip_m2_per_pixel(chip), 1e-9))
    surface = classify_surfaces(rgb, m_per_px=m_per_px, segformer_parking_mask=segformer_parking_mask)

    fusion = gis_augmentation
    fusion_excl: Optional[float] = None
    fusion_usable: Optional[float] = None
    fusion_final_m2: Optional[float] = None
    fusion_final_mask: Optional[np.ndarray] = None

    if fusion is not None:
        if (
            fusion.road_mask_gis_hw is not None
            and fusion.road_mask_gis_hw.shape == surface.road_mask.shape
        ):
            surface.road_mask = surface.road_mask | fusion.road_mask_gis_hw

    bd_for_semantic = bdtopo_buildings_mask
    b_src = "heuristic"
    if fusion is not None:
        b_src = fusion.building_mask_source
        if fusion.building_mask_hw is not None:
            gbm = fusion.building_mask_hw
            if gbm.shape == surface.roof_mask.shape:
                if bd_for_semantic is None:
                    bd_for_semantic = gbm
                else:
                    bd_for_semantic = np.logical_or(bd_for_semantic, gbm)
    elif bdtopo_buildings_mask is not None:
        b_src = "bdtopo"

    if fusion is not None:
        fusion_excl, fusion_usable, fusion_final_m2, fusion_final_mask = compute_fusion_area_metrics(
            chip, surface.parking_eligible_mask, surface.road_mask, fusion,
        )

    marked, geo = _scenario_marked_slots(
        chip, surface, segformer_mask=segformer_parking_mask, m_per_px=m_per_px,
    )
    unmarked = _scenario_unmarked_surface(chip, surface, m_per_px=m_per_px, site_type=site_type)
    roadside = _scenario_roadside_parking(chip, surface, m_per_px=m_per_px)
    courtyard = _scenario_courtyard_parking(chip, surface, m_per_px=m_per_px, site_type=site_type)

    components: Dict[str, Optional[ScenarioEstimate]] = {
        "marked_slots": marked,
        "unmarked_surface": unmarked,
        "roadside_parking": roadside,
        "courtyard_parking": courtyard,
    }

    # --- Couche sémantique ---
    # Dilate l'asphalt mask pour réintégrer les pixels véhicules eux-mêmes (souvent
    # exclus par les filtres "shadow" / "non-asphalt" alors qu'ils sont sur asphalte).
    veh_mask = surface.asphalt_mask.copy()
    if _HAS_CV2 and veh_mask.any():
        kk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        veh_mask = cv2.dilate(veh_mask.astype(np.uint8), kk, iterations=2).astype(bool)
    vehicles = detect_vehicles(
        chip.image,
        asphalt_mask=veh_mask,
        m_per_px=m_per_px,
        yolo_weights=yolo_vehicle_weights,
    )
    geo_rep = geo.repeated_pattern_score if geo else 0.0
    sep_density = 0.0
    if marked and isinstance(marked.extras, dict):
        rls = marked.extras.get("row_lengths_m") or []
        if rls and marked.capacity_estimate:
            # densité séparateurs estimée = nb places / longueur totale rangées
            sep_density = float(marked.capacity_estimate) / max(sum(rls), 1.0) * 1.0
    rs_road = fusion.road_source if fusion is not None else "heuristic"
    if rs_road == "none" and bool(surface.road_mask.any()):
        rs_road = "image_heuristic"
    semantic = compute_semantic_context(
        rgb,
        surface,
        m_per_px=m_per_px,
        vehicles=vehicles,
        geometry_repeated_pattern_score=geo_rep,
        geometry_separator_density=sep_density,
        has_osm_capacity=has_osm_capacity,
        bdtopo_buildings_mask=bd_for_semantic,
        gis_road_connection=bool(fusion.road_connection_gis) if fusion is not None else False,
        access_distance_m_gis=fusion.access_distance_m if fusion is not None else None,
        road_network_score_gis=float(fusion.road_network_score) if fusion is not None else 0.0,
        road_source=rs_road,
        building_mask_source=b_src,
        max_plausible_slots_cap=max_plausible_slots_cap,
    )

    # Applique plancher/plafond + dégradation par usability_score
    components = _apply_semantic_clamping(components, semantic, has_osm_capacity=has_osm_capacity)

    primary_mode, primary_est = _pick_primary(components)

    notes: List[str] = []
    if surface.roof_likelihood > 0.10:
        notes.append(
            f"toits détectés couvrant ~{surface.roof_likelihood*100:.0f}% de la puce"
        )
    if surface.road_likelihood > 0.05:
        notes.append(f"chaussée détectée ~{surface.road_likelihood*100:.0f}% de la puce")
    if surface.vegetation_likelihood > 0.15:
        notes.append(f"végétation ~{surface.vegetation_likelihood*100:.0f}%")
    if vehicles.vehicle_count > 0:
        notes.append(
            f"Présence de véhicules (indice secondaire, non utilisée comme capacité théorique) : "
            f"n={vehicles.vehicle_count}, méthode={vehicles.method}, alignement={vehicles.vehicle_alignment_score:.2f}"
        )
    if semantic.plausible_capacity_ceiling is not None:
        notes.append(
            f"plafond physique : ~{semantic.plausible_capacity_ceiling} places "
            f"(surface utile {semantic.parking_outside_buildings_ratio*100:.0f}% hors bâtiments)"
        )

    return MultiScenarioResult(
        primary_mode=primary_mode if primary_est else "unknown",
        primary_estimate=primary_est,
        components=components,
        surface=surface,
        geometry=geo,
        notes=notes,
        vehicles=vehicles,
        semantic=semantic,
        gis_fusion=fusion,
        fusion_excluded_building_area_m2=fusion_excl,
        fusion_usable_parking_area_m2=fusion_usable,
        fusion_final_candidate_area_m2=fusion_final_m2,
        fusion_final_parking_candidate_mask=fusion_final_mask,
    )


def _apply_semantic_clamping(
    components: Dict[str, Optional[ScenarioEstimate]],
    semantic: SemanticContext,
    *,
    has_osm_capacity: bool,
) -> Dict[str, Optional[ScenarioEstimate]]:
    """Plafonne chaque scénario par le plafond physique ; **aucun plancher** issu du comptage véhicules."""
    ceiling = semantic.plausible_capacity_ceiling
    floor = 0

    out: Dict[str, Optional[ScenarioEstimate]] = {}
    for mode, est in components.items():
        if est is None:
            out[mode] = None
            continue
        cap_new, notes = clamp_capacity_to_semantic_bounds(
            est.capacity_estimate, ceiling=ceiling, floor=floor,
        )
        cap_min_new, _ = clamp_capacity_to_semantic_bounds(
            est.capacity_min, ceiling=ceiling, floor=0,
        )
        cap_max_new, _ = clamp_capacity_to_semantic_bounds(
            est.capacity_max, ceiling=ceiling, floor=0,
        )
        conf = est.confidence

        # Dégradation par usability_score si pas d'OSM
        if not has_osm_capacity:
            if semantic.parking_usability_score < 25.0 and conf in ("medium", "strong"):
                conf = "weak"
            elif semantic.parking_usability_score < 45.0 and conf == "strong":
                conf = "medium"

        extras = dict(est.extras)
        if notes:
            extras["semantic_clamp_notes"] = notes
        extras["parking_usability_score"] = semantic.parking_usability_score
        extras["semantic_confidence"] = semantic.semantic_confidence
        extras["supporting_vehicle_presence_count"] = semantic.observed_vehicle_floor
        extras["plausible_capacity_ceiling"] = semantic.plausible_capacity_ceiling

        out[mode] = ScenarioEstimate(
            mode=mode,
            capacity_estimate=cap_new,
            capacity_min=cap_min_new,
            capacity_max=cap_max_new,
            confidence=conf,
            notes=est.notes,
            extras=extras,
        )
    return out
