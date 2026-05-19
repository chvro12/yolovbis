"""Couche sémantique pour ``parking_capacity_estimation`` : bâtiments, accès, qualité de zone.

L’objectif est l’estimation **théorique** de places (marquées, surface, linéaire, cour), pas le
comptage de véhicules présents. Les véhicules ne font qu’alimenter une **preuve d’usage** et la
confiance ; ils ne doivent pas piloter la capacité principale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from parking_capacity.surface_classification import SurfaceClassification
from parking_capacity.vehicle_detection import VehicleDetectionResult

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# Bornes physiques produit
M2_PER_SPACE_PHYSICAL_MIN = 12.0     # impossible en deçà sans empilement vertical
M2_PER_SPACE_PHYSICAL_TYP = 25.0     # référence parking standard
M2_PER_SPACE_PHYSICAL_MAX = 40.0     # parking très lâche

# Plafond produit : capacité plausible stricte < 40 places (défaut 39)
DEFAULT_MAX_PLAUSIBLE_CAPACITY_SLOTS = 39


@dataclass
class SemanticEvidence:
    """Scores 0-1 par dimension sémantique."""

    vehicle_evidence: float = 0.0           # présence + densité véhicules
    vehicle_alignment_evidence: float = 0.0  # alignement de véhicules
    building_exclusion_score: float = 0.0    # à quel point on a écarté les toits
    road_access_score: float = 0.0           # connexion route → zone garable
    compactness_score: float = 0.0           # la zone garable est-elle d'un seul tenant ?
    separators_score: float = 0.0            # densité marquages au sol
    geometry_score: float = 0.0              # qualité géométrie (rep. score)
    osm_score: float = 0.0                   # tag OSM dispo ?


@dataclass
class SemanticContext:
    """Contexte sémantique complet d'une puce."""

    building_mask: np.ndarray
    building_area_m2: float
    parking_outside_buildings_ratio: float
    vehicle_access_score: float
    road_connection_detected: bool
    observed_vehicle_floor: int  # présence observée (preuve secondaire ; ne impose pas la capacité)
    plausible_capacity_ceiling: Optional[int]  # plafond physique (surface/12 m²)
    parking_usability_score: float         # 0-100
    semantic_confidence: str               # none | weak | medium | strong
    evidence: SemanticEvidence = field(default_factory=SemanticEvidence)
    notes: List[str] = field(default_factory=list)
    access_distance_m: Optional[float] = None
    road_network_score: float = 0.0
    road_source: str = "heuristic"
    building_mask_source: str = "heuristic"


# -----------------------------
# Bâtiments : heuristique locale
# -----------------------------

def _building_mask_heuristic(
    rgb: np.ndarray,
    surface: SurfaceClassification,
) -> np.ndarray:
    """Bâtiments = non-asphalt, non-végétation, non-ombre, couleur "construite" + ombre adjacente.

    Fallback quand BD TOPO n'est pas dispo. Tend à manquer des bâtiments aux toits gris très
    similaires à l'asphalte ; mais inclut tuiles, métal coloré, etc.
    """
    if not _HAS_CV2:
        return surface.roof_mask.copy()

    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.int16)

    # Couleur "toit" : variations courantes
    red_tile = (r > 120) & (r > b + 15) & (r > g + 8)
    metallic = (gray > 95) & (gray < 195) & (np.abs(r - g) < 18) & (np.abs(g - b) < 18) & (np.abs(r - b) < 18)
    dark_roof = (gray < 90) & (np.abs(r - g) < 22) & (np.abs(g - b) < 22)

    # Exclure asphalte et chaussée et végétation et ombre
    not_road = ~surface.road_mask
    not_veg = ~surface.vegetation_mask
    not_shadow = ~surface.shadow_mask
    raw = (red_tile | metallic | dark_roof) & not_road & not_veg & not_shadow

    # Fermeture morphologique pour rassembler une toiture fragmentée
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2)

    # Composantes connexes : on garde celles > 60 m² (sera filtré plus tard via m_per_px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for i in range(1, n):
        area = stats[i, 4]
        if area < 60:  # seuil très large en pixels ; le seuil m² est appliqué par l'appelant
            continue
        out[labels == i] = 1

    # Inclure aussi les toits déjà détectés par surface_classification (red tile + closed contours)
    out = (out.astype(np.bool_) | surface.roof_mask)
    return out


def build_building_mask(
    rgb: np.ndarray,
    surface: SurfaceClassification,
    *,
    m_per_px: float,
    bdtopo_buildings_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Retourne (building_mask H×W bool, building_area_m2). Priorité au masque BD TOPO si fourni."""
    if bdtopo_buildings_mask is not None and bdtopo_buildings_mask.shape == surface.roof_mask.shape:
        m = bdtopo_buildings_mask.astype(np.bool_)
    else:
        m = _building_mask_heuristic(rgb, surface)

    # Filtrage final : composantes < 25 m² supprimées
    if _HAS_CV2 and m.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        clean = np.zeros_like(m, dtype=np.bool_)
        px2_to_m2 = m_per_px * m_per_px
        for i in range(1, n):
            area_m2 = float(stats[i, 4]) * px2_to_m2
            if area_m2 >= 25.0:
                clean[labels == i] = True
        m = clean

    area_m2 = float(m.sum()) * (m_per_px ** 2)
    return m, area_m2


# -----------------------------
# Accès véhicule (connexion route → zone)
# -----------------------------

def _access_to_road(
    parking_eligible_mask: np.ndarray,
    road_mask: np.ndarray,
) -> Tuple[float, bool]:
    """Mesure le contact + la connectivité entre la zone garable et la chaussée.

    Retourne (score 0-1, road_connection_detected).
    """
    if not _HAS_CV2 or parking_eligible_mask.dtype != np.bool_:
        return 0.0, False
    if parking_eligible_mask.sum() == 0:
        return 0.0, False

    # Si pas de route détectée, on retombe sur l'hypothèse : la zone garable doit toucher un bord du chip
    # (sortie probable vers la voirie).
    if road_mask is None or road_mask.sum() == 0:
        h, w = parking_eligible_mask.shape
        border = np.zeros_like(parking_eligible_mask, dtype=np.bool_)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(parking_eligible_mask.astype(np.uint8), k, iterations=1).astype(np.bool_)
        contact = (dilated & border).sum()
        return float(np.clip(contact / max(parking_eligible_mask.sum() * 0.03, 1.0), 0.0, 1.0)), bool(contact > 5)

    # Avec route : dilatation + intersection
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(parking_eligible_mask.astype(np.uint8), k, iterations=2).astype(np.bool_)
    contact = (dilated & road_mask).sum()
    score = float(np.clip(contact / max(parking_eligible_mask.sum() * 0.04, 1.0), 0.0, 1.0))
    return score, bool(contact > 5)


def _compactness_score(mask: np.ndarray) -> float:
    """Compacité : aire(plus grande composante) / aire totale."""
    if not _HAS_CV2 or mask.sum() == 0:
        return 0.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0.0
    largest = max(range(1, n), key=lambda i: stats[i, 4])
    total = float(mask.sum())
    return float(np.clip(stats[largest, 4] / max(total, 1.0), 0.0, 1.0))


# -----------------------------
# Score final
# -----------------------------

def _semantic_confidence(score: float) -> str:
    if score >= 65:
        return "strong"
    if score >= 45:
        return "medium"
    if score >= 25:
        return "weak"
    return "none"


def compute_semantic_context(
    rgb: np.ndarray,
    surface: SurfaceClassification,
    *,
    m_per_px: float,
    vehicles: VehicleDetectionResult,
    geometry_repeated_pattern_score: float = 0.0,
    geometry_separator_density: float = 0.0,
    has_osm_capacity: bool = False,
    bdtopo_buildings_mask: Optional[np.ndarray] = None,
    gis_road_connection: bool = False,
    access_distance_m_gis: Optional[float] = None,
    road_network_score_gis: float = 0.0,
    road_source: str = "heuristic",
    building_mask_source: str = "heuristic",
    max_plausible_slots_cap: int = DEFAULT_MAX_PLAUSIBLE_CAPACITY_SLOTS,
) -> SemanticContext:
    """Fusionne masques, accès, géométrie → ``parking_usability_score`` 0-100.

    ``observed_vehicle_floor`` recopie le nombre de véhicules visibles **à titre indicatif**
    (preuve secondaire). Il n'est **pas** utilisé comme plancher de capacité dans le pipeline.
    """
    h, w = rgb.shape[:2]
    notes: List[str] = []

    building_mask, building_area_m2 = build_building_mask(
        rgb, surface, m_per_px=m_per_px, bdtopo_buildings_mask=bdtopo_buildings_mask,
    )

    # Surface garable = parking_eligible - bâtiments
    parking_outside = surface.parking_eligible_mask & ~building_mask
    parking_outside_area_m2 = float(parking_outside.sum()) * (m_per_px ** 2)
    eligible_total_m2 = float(surface.parking_eligible_mask.sum()) * (m_per_px ** 2)
    outside_ratio = (
        parking_outside_area_m2 / max(eligible_total_m2, 1.0)
        if eligible_total_m2 > 0 else 0.0
    )

    access_score, road_connected = _access_to_road(parking_outside, surface.road_mask)
    if gis_road_connection:
        road_connected = True
        access_score = max(access_score, max(0.35, road_network_score_gis * 0.92))
    else:
        access_score = max(access_score, road_network_score_gis * 0.55)

    access_distance_out: Optional[float] = None
    if access_distance_m_gis is not None and access_distance_m_gis < 1e8:
        access_distance_out = round(float(access_distance_m_gis), 2)

    compactness = _compactness_score(parking_outside)

    # Capacité plancher : véhicules visibles
    observed_floor = vehicles.vehicle_count

    # Plafond physique : surface utile / 12 m² (très lâche). En dessous c'est implausible.
    plausible_ceiling = (
        int(parking_outside_area_m2 / M2_PER_SPACE_PHYSICAL_MIN)
        if parking_outside_area_m2 > 20.0 else None
    )
    # Plafond produit configurable : sert de cap maximum pour petits commerces.
    # Le cap dur (par défaut 39) protège les petites surfaces où l'estimation surface/12 m²
    # explose à cause de bruit de segmentation. Mais sur une grande parcelle réelle, le cap
    # dur écrasait les estimations légitimes (ex: clinique vétérinaire à 80 places réduite à 39).
    # On élève donc le cap à la valeur cohérente avec la surface utile (1 place / 25 m²),
    # qui est elle-même bornée par le ceiling géométrique 1/12 m². Au final :
    #     - petite surface (< ~975 m²) : cap = max_plausible_slots_cap (39)
    #     - grande surface : cap suit la surface mesurée
    # Si l'observation véhicules dépasse, elle prime (preuve directe).
    if plausible_ceiling is not None and max_plausible_slots_cap >= 0:
        surface_implied_cap = int(parking_outside_area_m2 / M2_PER_SPACE_PHYSICAL_TYP)
        effective_cap = max(int(max_plausible_slots_cap), surface_implied_cap)
        capped = min(plausible_ceiling, effective_cap)
        plausible_ceiling = max(capped, int(observed_floor))
    elif plausible_ceiling is not None:
        plausible_ceiling = max(plausible_ceiling, int(observed_floor))

    # Score sémantique
    evidence = SemanticEvidence(
        vehicle_evidence=min(0.22, vehicles.vehicle_count / 30.0),
        vehicle_alignment_evidence=vehicles.vehicle_alignment_score,
        building_exclusion_score=float(np.clip(outside_ratio, 0.0, 1.0)),
        road_access_score=float(access_score),
        compactness_score=float(compactness),
        separators_score=float(np.clip(geometry_separator_density / 0.20, 0.0, 1.0)),
        geometry_score=float(np.clip(geometry_repeated_pattern_score, 0.0, 1.0)),
        osm_score=1.0 if has_osm_capacity else 0.0,
    )

    # Pondération produit
    score = (
        8.0 * evidence.vehicle_evidence
        + 8.0 * evidence.vehicle_alignment_evidence
        + 12.0 * evidence.road_access_score
        + 10.0 * evidence.compactness_score
        + 10.0 * evidence.building_exclusion_score
        + 12.0 * evidence.geometry_score
        + 8.0 * evidence.separators_score
        + 18.0 * evidence.osm_score
    )
    # Garde-fous
    if vehicles.vehicle_count == 0 and not has_osm_capacity:
        score *= 0.75
        notes.append("aucun_vehicule_detecte_legere_baisse_confiance")
    if outside_ratio < 0.25 and not has_osm_capacity:
        score *= 0.6
        notes.append("majorite_eligible_dans_batiment_baisse_score")
    if not road_connected and not has_osm_capacity:
        score *= 0.7
        notes.append("aucun_acces_route_baisse_score")

    score = float(np.clip(score, 0.0, 100.0))

    rn_score = float(np.clip(max(road_network_score_gis, min(1.0, access_score)), 0.0, 1.0))

    return SemanticContext(
        building_mask=building_mask,
        building_area_m2=round(building_area_m2, 1),
        parking_outside_buildings_ratio=round(outside_ratio, 3),
        vehicle_access_score=round(access_score, 3),
        road_connection_detected=bool(road_connected),
        observed_vehicle_floor=int(observed_floor),
        plausible_capacity_ceiling=plausible_ceiling,
        parking_usability_score=round(score, 1),
        semantic_confidence=_semantic_confidence(score),
        evidence=evidence,
        notes=notes,
        access_distance_m=access_distance_out,
        road_network_score=round(rn_score, 4),
        road_source=road_source,
        building_mask_source=building_mask_source,
    )


def clamp_capacity_to_semantic_bounds(
    capacity: Optional[int],
    *,
    ceiling: Optional[int],
    floor: int = 0,
) -> Tuple[Optional[int], List[str]]:
    """Applique un plafond physique (et optionnellement un plancher explicite, rarement les véhicules)."""
    notes: List[str] = []
    if capacity is None:
        return None, notes
    cap = int(capacity)
    if ceiling is not None and cap > ceiling:
        notes.append(f"capacite_plafonne_au_plafond_physique_{ceiling}")
        cap = ceiling
    if floor > 0 and cap < floor:
        notes.append(f"capacite_remonte_au_plancher_vehicules_{floor}")
        cap = floor
    return cap, notes
