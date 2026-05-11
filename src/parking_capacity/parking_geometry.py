"""Analyse géométrique parking : prétraitement, détection lignes multi-échelle, rangées, capacité estimée.

Cette chaîne remplace l'ancien « surface ÷ m²_par_place » : la capacité géométrique n'est produite que si
des rangées régulières sont effectivement détectées dans l'orthophoto. Toutes les valeurs intermédiaires
(échelle, lignes, candidats rangées, raisons de rejet) sont exposées pour le diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from parking_capacity.imagery_wms import OrthoChip, chip_m2_per_pixel

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


SLOT_WIDTH_MIN_M = 2.3
SLOT_WIDTH_MAX_M = 2.7
SLOT_WIDTH_TYP_M = 2.5
SLOT_LENGTH_MIN_M = 4.5
SLOT_LENGTH_MAX_M = 5.5
SLOT_LENGTH_TYP_M = 5.0
AISLE_WIDTH_M = 6.0  # allée centrale typique entre 2 rangées dos à dos


@dataclass
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    length_px: float


@dataclass
class RowCandidate:
    """Une rangée détectée : ligne de centre + longueur utile + nombre de places estimé."""

    orientation_deg: float
    center_proj: float  # projection sur axe perpendiculaire (px)
    length_m: float
    capacity: int
    line_count: int
    accepted: bool = False
    reject_reason: Optional[str] = None
    separator_count: int = 0
    separator_density_per_m: float = 0.0
    inside_roof: bool = False
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass
class GeometryDebug:
    """Champs détaillés pour diagnostic ; sérialisés dans result.json."""

    meters_per_pixel: float = 0.0
    raw_line_count: int = 0
    filtered_line_count: int = 0
    usable_line_count: int = 0
    dominant_orientations_deg: List[float] = field(default_factory=list)
    row_candidates: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    capacity_formula_used: Optional[str] = None
    row_lengths_m: List[float] = field(default_factory=list)
    chain_failure: Optional[str] = None  # explication unique si la chaîne s'arrête tôt


@dataclass
class GeometryParkingAnalysis:
    """Sortie structurée géométrie parking.

    ``geometric_capacity_estimate`` est ``None`` tant qu'aucune rangée exploitable n'est validée :
    on ne convertit jamais une simple surface en capacité ici.
    """

    parking_rows_detected: int
    estimated_row_orientation_deg: float
    estimated_slot_width_m: float
    estimated_slot_length_m: float
    repeated_pattern_score: float
    geometric_capacity_estimate: Optional[int]
    geometric_capacity_min: Optional[int]
    geometric_capacity_max: Optional[int]
    geometry_confidence: str
    asphalt_fraction_estimate: float
    parking_structure_detected: bool
    notes: str
    slot_angle_deg: float = 0.0
    line_detection_count: int = 0
    usable_line_count: int = 0
    row_lengths_m: List[float] = field(default_factory=list)
    debug: GeometryDebug = field(default_factory=GeometryDebug)


# -----------------------------
# Helpers de prétraitement
# -----------------------------

def _rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _gray_clahe(rgb: np.ndarray) -> np.ndarray:
    """Niveaux de gris + CLAHE pour relever les marquages clairs sur asphalte."""
    if _HAS_CV2:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)
    return np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)


def _vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """Heuristique chlorophylle : vert dominant => probable arbre / herbe."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    veg = (g > r + 8) & (g > b + 6) & (g > 60)
    return veg.astype(np.bool_)


def _white_marking_mask(rgb: np.ndarray, gray_clahe: np.ndarray) -> np.ndarray:
    """Lignes blanches / jaunes : seuil haut sur luminance + faible saturation rouge-bleu."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    bright = gray_clahe > int(np.percentile(gray_clahe, 88))
    near_white = (np.abs(r - g) < 28) & (np.abs(g - b) < 32) & (r > 130)
    yellowish = (r > 140) & (g > 110) & (b < 130) & (r > b + 25)
    return (bright & (near_white | yellowish)).astype(np.bool_)


def _asphalt_fraction(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    if _HAS_CV2:
        blur = cv2.GaussianBlur(g, (9, 9), 0)
    else:
        blur = g
    low = blur < (np.percentile(blur, 42) + 8)
    flat = np.abs(g - blur) < 20
    mask = low & flat
    return float(np.clip(np.mean(mask), 0.0, 1.0))


def _angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _circular_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _meters_per_pixel(m2_per_px: float) -> float:
    return math.sqrt(max(m2_per_px, 1e-9))


# -----------------------------
# Détection lignes multi-échelle
# -----------------------------

def _detect_edges(
    gray_clahe: np.ndarray,
    *,
    marking_mask: np.ndarray,
    veg_mask: np.ndarray,
    roi_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Canny adaptatif : seuils calés sur la médiane, restreint au domaine utile."""
    g = gray_clahe
    if _HAS_CV2:
        blur = cv2.GaussianBlur(g, (5, 5), 0)
    else:
        blur = g
    med = float(np.median(blur))
    sigma = 0.33
    lo = max(20, int((1.0 - sigma) * med))
    hi = max(lo + 10, int((1.0 + sigma) * med))

    if _HAS_CV2:
        edges = cv2.Canny(blur, lo, hi)
    else:
        gx = np.gradient(blur.astype(np.float32), axis=1)
        gy = np.gradient(blur.astype(np.float32), axis=0)
        mag = np.sqrt(gx * gx + gy * gy)
        edges = (mag > np.percentile(mag, 90)).astype(np.uint8) * 255

    # Marquages blancs renforcés : OR pour ne pas perdre les lignes peu contrastées.
    if marking_mask.shape == edges.shape:
        edges = np.maximum(edges, (marking_mask.astype(np.uint8) * 255))

    # Suppression végétation
    if veg_mask.shape == edges.shape:
        edges[veg_mask] = 0

    if roi_mask is not None and roi_mask.shape == edges.shape and _HAS_CV2:
        m255 = (roi_mask.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m255 = cv2.dilate(m255, k, iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=m255)
    elif roi_mask is not None and roi_mask.shape == edges.shape:
        edges = edges * roi_mask.astype(np.uint8)

    return edges


def _hough_multi_scale(edges: np.ndarray, m_per_px: float) -> List[LineSegment]:
    """HoughLinesP à 2 paramétrages : lignes courtes (places) + lignes longues (bordures/rangées)."""
    if not _HAS_CV2:
        return []
    h, w = edges.shape
    diag = math.hypot(w, h)
    # 1 m réel ≈ 1/m_per_px pixels
    px_per_m = 1.0 / max(m_per_px, 1e-6)

    short_min_len = max(10, int(1.8 * px_per_m))   # ≥ ~1.8 m
    long_min_len = max(20, int(4.5 * px_per_m))    # ≥ ~4.5 m (longueur place)
    short_gap = max(4, int(0.8 * px_per_m))
    long_gap = max(6, int(1.5 * px_per_m))

    L1 = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(35, int(diag * 0.04)),
        minLineLength=short_min_len,
        maxLineGap=short_gap,
    )
    L2 = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(45, int(diag * 0.07)),
        minLineLength=long_min_len,
        maxLineGap=long_gap,
    )

    segs: List[LineSegment] = []
    for L in (L1, L2):
        if L is None:
            continue
        for ln in L[:, 0]:
            x1, y1, x2, y2 = (float(ln[i]) for i in range(4))
            le = math.hypot(x2 - x1, y2 - y1)
            if le < short_min_len:
                continue
            ang = _angle_deg(x1, y1, x2, y2)
            segs.append(LineSegment(x1, y1, x2, y2, ang, le))
    return segs


def _dominant_orientations(segs: List[LineSegment]) -> List[float]:
    """Histogramme angulaire 6° + retour des 1-2 orientations majoritaires."""
    if not segs:
        return []
    angs = np.array([s.angle_deg for s in segs], dtype=np.float64)
    weights = np.array([s.length_px for s in segs], dtype=np.float64)
    bins = np.arange(0, 186, 6)
    hist, _ = np.histogram(angs, bins=bins, weights=weights)
    if hist.sum() <= 0:
        return []
    order = np.argsort(hist)[::-1]
    out: List[float] = []
    for idx in order:
        if hist[idx] <= 0:
            break
        center = float(bins[idx] + 3.0)
        if any(_circular_distance_deg(center, c) < 25.0 for c in out):
            continue
        out.append(center)
        if len(out) >= 2:
            break
    return out


# -----------------------------
# Estimation des rangées
# -----------------------------

def _project_lines_perpendicular(
    segs: List[LineSegment],
    orientation_deg: float,
) -> List[Tuple[float, LineSegment]]:
    """Pour chaque segment, projeter son milieu sur l'axe perpendiculaire à l'orientation."""
    rad = math.radians(orientation_deg + 90.0)
    nx, ny = math.cos(rad), math.sin(rad)
    out: List[Tuple[float, LineSegment]] = []
    for s in segs:
        mx = 0.5 * (s.x1 + s.x2)
        my = 0.5 * (s.y1 + s.y2)
        proj = mx * nx + my * ny
        out.append((proj, s))
    out.sort(key=lambda t: t[0])
    return out


def _cluster_rows(
    projs: List[Tuple[float, LineSegment]],
    *,
    px_per_m: float,
) -> List[Tuple[float, List[LineSegment]]]:
    """Cluster 1D le long de l'axe perpendiculaire : une rangée = un amas de lignes proches."""
    if not projs:
        return []
    cluster_gap = max(1.2 * px_per_m, 6.0)  # ~1.2 m entre lignes d'une même rangée
    clusters: List[List[Tuple[float, LineSegment]]] = []
    current: List[Tuple[float, LineSegment]] = [projs[0]]
    for prev, nxt in zip(projs, projs[1:]):
        if (nxt[0] - prev[0]) <= cluster_gap:
            current.append(nxt)
        else:
            clusters.append(current)
            current = [nxt]
    clusters.append(current)
    out: List[Tuple[float, List[LineSegment]]] = []
    for c in clusters:
        center = float(np.mean([p for p, _ in c]))
        out.append((center, [s for _, s in c]))
    return out


def _row_length_m(segs: List[LineSegment], orientation_deg: float, m_per_px: float) -> float:
    """Étalement des milieux le long de l'axe orientation."""
    if not segs:
        return 0.0
    rad = math.radians(orientation_deg)
    tx, ty = math.cos(rad), math.sin(rad)
    extents: List[float] = []
    for s in segs:
        for x, y in ((s.x1, s.y1), (s.x2, s.y2)):
            extents.append(x * tx + y * ty)
    if not extents:
        return 0.0
    span_px = float(max(extents) - min(extents))
    return span_px * m_per_px


def _build_row_candidates(
    segs: List[LineSegment],
    orientation_deg: float,
    *,
    m_per_px: float,
    roof_mask: Optional[np.ndarray] = None,
) -> List[RowCandidate]:
    """Cluster les segments orientés ~⊥ par rangée et estime longueur + places.

    Les rangées s'appuient principalement sur des lignes ~⊥ (séparateurs de places quand ils existent).
    Si trop peu de lignes ⊥, on retombe sur des lignes parallèles à l'orientation (bordures) — utile
    pour les zones bitumées non marquées : la « rangée » devient alors un *axe* exploitable, pas
    un compteur de places.
    """
    if not segs:
        return []
    px_per_m = 1.0 / max(m_per_px, 1e-6)
    perp_target = (orientation_deg + 90.0) % 180.0
    perp_segs = [s for s in segs if _circular_distance_deg(s.angle_deg, perp_target) < 22.0]
    parallel_segs = [s for s in segs if _circular_distance_deg(s.angle_deg, orientation_deg) < 18.0]

    # Si on a assez de perpendiculaires : cluster classique places.
    if len(perp_segs) >= 4:
        projs = _project_lines_perpendicular(perp_segs, orientation_deg)
        clusters = _cluster_rows(projs, px_per_m=px_per_m)
    else:
        # Sinon on cluster les parallèles (bordures de rangée) — séparateur density restera 0.
        projs = _project_lines_perpendicular(parallel_segs, orientation_deg)
        clusters = _cluster_rows(projs, px_per_m=px_per_m)

    cands: List[RowCandidate] = []
    for center, members in clusters:
        if len(members) < 3:
            continue
        row_len_m = _row_length_m(members, orientation_deg, m_per_px)
        if row_len_m < 4.5:
            continue
        cap = max(1, int(round(row_len_m / SLOT_WIDTH_TYP_M)))

        # Centre géométrique du cluster (en pixels image)
        cx = float(np.mean([0.5 * (s.x1 + s.x2) for s in members]))
        cy = float(np.mean([0.5 * (s.y1 + s.y2) for s in members]))

        # Densité de séparateurs : lignes ~⊥ qui croisent cette rangée
        if perp_segs:
            sep_centers = _project_lines_perpendicular(perp_segs, orientation_deg + 90.0)
            # On compte les séparateurs dont la projection sur l'axe orientation tombe dans
            # l'étendue de la rangée. Approximation : on prend ceux dont le milieu est à
            # < row_len/2 du centre de la rangée le long de l'orientation.
            rad = math.radians(orientation_deg)
            tx, ty = math.cos(rad), math.sin(rad)
            row_center_along = cx * tx + cy * ty
            half = max(row_len_m / max(m_per_px, 1e-6) * 0.5, 1.0)
            seps = 0
            for s in perp_segs:
                mx = 0.5 * (s.x1 + s.x2)
                my = 0.5 * (s.y1 + s.y2)
                # distance au centre rangée le long de l'orientation
                along = mx * tx + my * ty
                if abs(along - row_center_along) > half:
                    continue
                # distance ⊥ au centre rangée
                rad_perp = math.radians(orientation_deg + 90.0)
                nx, ny = math.cos(rad_perp), math.sin(rad_perp)
                perp = mx * nx + my * ny
                if abs(perp - center) > 4.0 * px_per_m:
                    continue
                seps += 1
            sep_density = seps / max(row_len_m, 1.0)
        else:
            seps = 0
            sep_density = 0.0

        inside_roof = False
        if roof_mask is not None and 0 <= int(cy) < roof_mask.shape[0] and 0 <= int(cx) < roof_mask.shape[1]:
            inside_roof = bool(roof_mask[int(cy), int(cx)])

        cands.append(
            RowCandidate(
                orientation_deg=orientation_deg,
                center_proj=center,
                length_m=row_len_m,
                capacity=cap,
                line_count=len(members),
                separator_count=seps,
                separator_density_per_m=sep_density,
                inside_roof=inside_roof,
                center_x=cx,
                center_y=cy,
            )
        )
    return cands


def _repeated_pattern_score(rows: List[RowCandidate]) -> Tuple[float, List[float]]:
    """Score 0-1 sur la régularité inter-rangées (espacements ~6-8 m typiques)."""
    if len(rows) < 2:
        return 0.0, []
    centers = sorted(r.center_proj for r in rows)
    gaps = np.diff(np.array(centers, dtype=np.float64))
    gaps = gaps[gaps > 1e-3]
    if gaps.size < 1:
        return 0.0, []
    med = float(np.median(gaps))
    cv = float(np.std(gaps) / (med + 1e-6))
    score = float(np.clip(1.0 - min(cv, 1.2) / 1.2, 0.0, 1.0))
    return score, gaps.tolist()


# -----------------------------
# Chaîne complète
# -----------------------------

def _empty_result(asp: float, debug: GeometryDebug, *, note: str, conf: str = "none") -> GeometryParkingAnalysis:
    return GeometryParkingAnalysis(
        parking_rows_detected=0,
        estimated_row_orientation_deg=0.0,
        estimated_slot_width_m=SLOT_WIDTH_TYP_M,
        estimated_slot_length_m=SLOT_LENGTH_TYP_M,
        repeated_pattern_score=0.0,
        geometric_capacity_estimate=None,
        geometric_capacity_min=None,
        geometric_capacity_max=None,
        geometry_confidence=conf,
        asphalt_fraction_estimate=round(asp, 3),
        parking_structure_detected=False,
        notes=note,
        slot_angle_deg=0.0,
        line_detection_count=debug.raw_line_count,
        usable_line_count=debug.usable_line_count,
        row_lengths_m=list(debug.row_lengths_m),
        debug=debug,
    )


def _confidence_label(
    *,
    accepted_rows: int,
    rep_score: float,
    usable_lines: int,
    asphalt: float,
) -> str:
    if accepted_rows >= 3 and rep_score >= 0.45 and usable_lines >= 24:
        return "strong"
    if accepted_rows >= 2 and rep_score >= 0.30 and usable_lines >= 14:
        return "medium"
    if accepted_rows >= 1 and (rep_score >= 0.18 or usable_lines >= 10) and asphalt >= 0.10:
        return "weak"
    return "none"


def _analyze(
    chip: OrthoChip,
    *,
    roi_mask: Optional[np.ndarray] = None,
    roof_mask: Optional[np.ndarray] = None,
    max_row_length_m: float = 60.0,
    require_separators: bool = False,
    debug_collect: bool = False,
) -> GeometryParkingAnalysis:
    rgb = _rgb_array(chip.image)
    gray = _gray_clahe(rgb)
    veg = _vegetation_mask(rgb)
    marking = _white_marking_mask(rgb, gray)
    asp = _asphalt_fraction(gray)

    m2_per_px = chip_m2_per_pixel(chip)
    m_per_px = _meters_per_pixel(m2_per_px)

    debug = GeometryDebug(meters_per_pixel=round(m_per_px, 4))

    if not _HAS_CV2:
        debug.chain_failure = "opencv_indisponible"
        return _empty_result(asp, debug, note="opencv_indisponible")

    edges = _detect_edges(gray, marking_mask=marking, veg_mask=veg, roi_mask=roi_mask)
    edge_density = float((edges > 0).mean()) if edges.size else 0.0
    if edge_density < 0.001:
        debug.chain_failure = "aucune_arête_détectée"
        return _empty_result(asp, debug, note="aucune_arête_détectée")

    segs = _hough_multi_scale(edges, m_per_px)
    debug.raw_line_count = len(segs)

    # Filtrage : longueur >= 1.8m, hors végétation
    min_len_px = max(10.0, 1.8 / max(m_per_px, 1e-6))
    filtered = [s for s in segs if s.length_px >= min_len_px]
    debug.filtered_line_count = len(filtered)

    if len(filtered) < 6:
        debug.chain_failure = "lignes_filtrees_insuffisantes"
        return _empty_result(asp, debug, note="lignes_filtrees_insuffisantes")

    orients = _dominant_orientations(filtered)
    debug.dominant_orientations_deg = [round(a, 1) for a in orients]
    if not orients:
        debug.chain_failure = "aucune_orientation_dominante"
        return _empty_result(asp, debug, note="aucune_orientation_dominante")

    # Lignes "utiles" : alignées avec une orientation dominante (±14°)
    usable = [s for s in filtered if any(_circular_distance_deg(s.angle_deg, o) < 14.0 for o in orients)]
    debug.usable_line_count = len(usable)
    if len(usable) < 6:
        debug.chain_failure = "alignement_avec_orientations_faible"
        return _empty_result(asp, debug, note="alignement_avec_orientations_faible")

    # Pour chaque orientation, construire les rangées candidates
    all_cands: List[RowCandidate] = []
    for o in orients:
        all_cands.extend(_build_row_candidates(filtered, o, m_per_px=m_per_px, roof_mask=roof_mask))
    debug.row_candidates = len(all_cands)
    if not all_cands:
        debug.chain_failure = "aucun_groupe_de_lignes_perpendiculaires"
        return _empty_result(asp, debug, note="aucun_groupe_de_lignes_perpendiculaires")

    # Acceptation contextuelle : longueur, places, séparateurs (qualité, pas obligatoire),
    # rejet si la rangée tombe dans un toit détecté.
    accepted: List[RowCandidate] = []
    rejected_reasons: List[str] = []
    for c in all_cands:
        if c.inside_roof:
            c.reject_reason = "centre_dans_toit_detecte"
        elif c.length_m < 4.5:
            c.reject_reason = "rangee_trop_courte"
        elif c.length_m > max_row_length_m:
            # Rangée invraisemblablement longue → suspect (bordure de toit / voirie).
            # On l'accepte seulement si elle a une vraie densité de séparateurs.
            if c.separator_density_per_m < 0.15:
                c.reject_reason = "rangee_trop_longue_sans_separateurs"
        elif c.capacity < 3:
            c.reject_reason = "places_estimees_insuffisantes"
        elif c.line_count < 3:
            c.reject_reason = "lignes_alignees_insuffisantes"
        elif require_separators and c.separator_density_per_m < 0.12:
            c.reject_reason = "densite_separateurs_insuffisante"

        if c.reject_reason is None:
            c.accepted = True
            # Cap la capacité de la rangée à 1.5x la valeur cohérente si séparateurs faibles.
            if c.separator_density_per_m < 0.10 and c.length_m > 35.0:
                c.capacity = max(3, int(round(c.length_m / max(SLOT_WIDTH_TYP_M * 1.4, 3.0))))
            accepted.append(c)
        if not c.accepted and c.reject_reason:
            rejected_reasons.append(c.reject_reason)

    debug.accepted_rows = len(accepted)
    debug.rejected_rows = len(all_cands) - len(accepted)
    debug.rejection_reasons = sorted(set(rejected_reasons))
    debug.row_lengths_m = [round(c.length_m, 2) for c in accepted]

    if not accepted:
        # Garde l'orientation détectée et le compte brut (utile pour l'UI), mais pas de capacité.
        out = _empty_result(asp, debug, note="rangees_candidates_rejetees")
        # On garde une trace pour l'UI sans prétendre à une structure parking.
        out.estimated_row_orientation_deg = orients[0]
        out.parking_rows_detected = 0
        return out

    rep_score, gaps = _repeated_pattern_score(accepted)
    confidence = _confidence_label(
        accepted_rows=len(accepted),
        rep_score=rep_score,
        usable_lines=len(usable),
        asphalt=asp,
    )

    # Estimation slot width depuis l'espacement médian de lignes consécutives ⊥
    slot_w_m = SLOT_WIDTH_TYP_M
    perp_target = (orients[0] + 90.0) % 180.0
    perp_lines = [s for s in usable if _circular_distance_deg(s.angle_deg, perp_target) < 22.0]
    if len(perp_lines) >= 4:
        proj_perp = _project_lines_perpendicular(perp_lines, orients[0] + 90.0)
        coords = [p for p, _ in proj_perp]
        if len(coords) >= 4:
            diffs = np.diff(np.array(coords))
            diffs = diffs[(diffs > 0.6 / max(m_per_px, 1e-6)) & (diffs < 3.5 / max(m_per_px, 1e-6))]
            if diffs.size >= 2:
                slot_w_m = float(np.clip(np.median(diffs) * m_per_px, SLOT_WIDTH_MIN_M, SLOT_WIDTH_MAX_M))

    # Détection allée centrale -> doubler les rangées dos à dos
    n_rows_logical = len(accepted)
    double_row_bonus = 0
    if gaps:
        gaps_m = [g * m_per_px for g in gaps]
        # Une allée correspond à un écart ~ AISLE_WIDTH_M ± 2m
        aisle_gaps = [g for g in gaps_m if abs(g - AISLE_WIDTH_M) < 2.5]
        if aisle_gaps:
            double_row_bonus = min(len(aisle_gaps), n_rows_logical // 2 + 1)

    sum_capacity = sum(c.capacity for c in accepted)
    capacity = sum_capacity + double_row_bonus * max(1, int(np.median([c.capacity for c in accepted])))
    capacity = int(np.clip(capacity, 1, 1500))

    cap_min = max(1, int(round(capacity * 0.78)))
    cap_max = int(round(capacity * 1.22)) + 2

    debug.capacity_formula_used = (
        "sum(longueur_rangée_m / slot_width_m) + bonus_allée_centrale"
    )

    return GeometryParkingAnalysis(
        parking_rows_detected=n_rows_logical,
        estimated_row_orientation_deg=round(orients[0], 2),
        estimated_slot_width_m=round(slot_w_m, 2),
        estimated_slot_length_m=SLOT_LENGTH_TYP_M,
        repeated_pattern_score=round(rep_score, 3),
        geometric_capacity_estimate=capacity,
        geometric_capacity_min=cap_min,
        geometric_capacity_max=cap_max,
        geometry_confidence=confidence,
        asphalt_fraction_estimate=round(asp, 3),
        parking_structure_detected=confidence in ("medium", "strong"),
        notes="opencv_canny_hough_rangees" + ("_roi_segformer" if roi_mask is not None else ""),
        slot_angle_deg=round(orients[0], 2),
        line_detection_count=debug.raw_line_count,
        usable_line_count=debug.usable_line_count,
        row_lengths_m=list(debug.row_lengths_m),
        debug=debug,
    )


def merge_geometry_analyses(
    full: Optional[GeometryParkingAnalysis],
    roi: Optional[GeometryParkingAnalysis],
) -> Optional[GeometryParkingAnalysis]:
    """Fusionne analyse plein-cadre vs analyse ROI (SegFormer) : prend la plus structurée."""
    if full is None:
        return roi
    if roi is None:
        return full

    def rank(g: GeometryParkingAnalysis) -> tuple:
        conf_map = {"strong": 4, "high": 4, "medium": 3, "low": 2, "weak": 2, "none": 0}
        conf = conf_map.get(g.geometry_confidence, 0)
        cap = g.geometric_capacity_estimate
        cap_n = int(cap) if cap is not None else -1
        struct = 1 if g.parking_structure_detected else 0
        return (conf, struct, g.repeated_pattern_score, cap_n)

    rf, rr = rank(full), rank(roi)
    if rf > rr:
        return full
    if rr > rf:
        return roi
    if roi.parking_structure_detected and not full.parking_structure_detected:
        return roi
    return full


def analyze_parking_geometry(
    chip: OrthoChip,
    *,
    segformer_roi_mask: Optional[np.ndarray] = None,
    roof_mask: Optional[np.ndarray] = None,
    max_row_length_m: float = 60.0,
    require_separators: bool = False,
) -> GeometryParkingAnalysis:
    """Chaîne géométrique complète : prétraitement → Canny adaptatif → Hough multi-échelle →
    orientations → rangées candidates → acceptation → capacité géométrique structurée.

    ``roof_mask`` : rangées dont le centre tombe dans un toit détecté sont rejetées.
    ``max_row_length_m`` : au-delà, rangée acceptée seulement si densité de séparateurs ≥ 0.15/m.
    ``require_separators`` : si vrai, rejette les rangées sans séparateurs perpendiculaires
    (utile uniquement quand on cherche des places clairement marquées).

    Aucun résultat « surface / m² » n'est produit ici : si la chaîne échoue, ``geometric_capacity_estimate``
    reste ``None`` et ``debug.chain_failure`` indique précisément où elle s'est arrêtée.
    """
    return _analyze(
        chip,
        roi_mask=segformer_roi_mask,
        roof_mask=roof_mask,
        max_row_length_m=max_row_length_m,
        require_separators=require_separators,
    )


def render_geometry_debug_overlays(
    chip: OrthoChip,
    analysis: GeometryParkingAnalysis,
    *,
    segformer_roi_mask: Optional[np.ndarray] = None,
) -> dict[str, Image.Image]:
    """Génère les images PNG de diagnostic (edges/hough/rows/overlay).

    Retourne un dict de PIL.Image (clés sans extension) ; appelants persistent eux-mêmes.
    """
    out: dict[str, Image.Image] = {}
    if not _HAS_CV2:
        return out
    rgb = _rgb_array(chip.image)
    gray = _gray_clahe(rgb)
    veg = _vegetation_mask(rgb)
    marking = _white_marking_mask(rgb, gray)
    edges = _detect_edges(gray, marking_mask=marking, veg_mask=veg, roi_mask=segformer_roi_mask)

    out["debug_edges"] = Image.fromarray(edges)

    h, w = edges.shape
    m_per_px = _meters_per_pixel(chip_m2_per_pixel(chip))
    segs = _hough_multi_scale(edges, m_per_px)
    hough_img = rgb.copy()
    for s in segs:
        cv2.line(hough_img, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)), (255, 80, 80), 1)
    out["debug_hough_lines"] = Image.fromarray(hough_img)

    rows_img = rgb.copy()
    if analysis.estimated_row_orientation_deg or analysis.parking_rows_detected:
        # rejoue le cluster pour visualiser les rangées acceptées
        orients = _dominant_orientations(segs)
        for o in orients[:2]:
            cands = _build_row_candidates(segs, o, m_per_px=m_per_px)
            for c in cands:
                # tracer une ligne médiane de la rangée
                rad = math.radians(o)
                tx, ty = math.cos(rad), math.sin(rad)
                rad_perp = math.radians(o + 90.0)
                nx, ny = math.cos(rad_perp), math.sin(rad_perp)
                cx = nx * c.center_proj
                cy = ny * c.center_proj
                # tracer une ligne de longueur estimée à travers le centre
                half_len = (c.length_m / max(m_per_px, 1e-6)) * 0.5
                x1 = int(cx - tx * half_len)
                y1 = int(cy - ty * half_len)
                x2 = int(cx + tx * half_len)
                y2 = int(cy + ty * half_len)
                color = (40, 200, 60) if c.capacity >= 3 and c.length_m >= 4.5 else (200, 200, 40)
                cv2.line(rows_img, (x1, y1), (x2, y2), color, 2)
    out["debug_parking_rows"] = Image.fromarray(rows_img)

    overlay = rgb.copy()
    # halo edges en rouge
    overlay[edges > 0] = (0.5 * overlay[edges > 0] + 0.5 * np.array([255, 30, 30], dtype=np.uint8)).astype(np.uint8)
    # tracer rangées acceptées
    if analysis.parking_rows_detected:
        orients = _dominant_orientations(segs)
        for o in orients[:1]:
            cands = _build_row_candidates(segs, o, m_per_px=m_per_px)
            for c in cands:
                rad = math.radians(o)
                tx, ty = math.cos(rad), math.sin(rad)
                rad_perp = math.radians(o + 90.0)
                nx, ny = math.cos(rad_perp), math.sin(rad_perp)
                cx = nx * c.center_proj
                cy = ny * c.center_proj
                half_len = (c.length_m / max(m_per_px, 1e-6)) * 0.5
                x1 = int(cx - tx * half_len)
                y1 = int(cy - ty * half_len)
                x2 = int(cx + tx * half_len)
                y2 = int(cy + ty * half_len)
                if c.capacity >= 3 and c.length_m >= 4.5:
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 30), 3)
    out["debug_geometry_overlay"] = Image.fromarray(overlay)
    return out
