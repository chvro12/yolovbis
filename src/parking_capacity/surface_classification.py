"""Classification heuristique des surfaces sur orthophoto BD ORTHO.

Distingue bitume garable / toit / chaussée / végétation / ombre / bord-de-bâtiment.

L'objectif **n'est pas** de remplacer un modèle sémantique entraîné, mais de fournir des indices
robustes pour discriminer :
  - une vraie zone garable (asphalt continu, accessible),
  - un toit (rectangulaire fermé, lignes longues régulières, souvent ombre adjacente),
  - une chaussée (linéaire, fines bandes, parfois marquage axial),
  - un terrain végétal,
  - une ombre portée.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


@dataclass
class SurfaceClassification:
    """Masques bool H×W + statistiques globales 0-1 + masque parking_eligible final."""

    asphalt_mask: np.ndarray
    roof_mask: np.ndarray
    road_mask: np.ndarray
    vegetation_mask: np.ndarray
    shadow_mask: np.ndarray
    building_edge_mask: np.ndarray
    parking_eligible_mask: np.ndarray

    asphalt_likelihood: float = 0.0
    roof_likelihood: float = 0.0
    road_likelihood: float = 0.0
    vegetation_likelihood: float = 0.0
    shadow_likelihood: float = 0.0
    building_edge_likelihood: float = 0.0

    notes: List[str] = field(default_factory=list)


def _vegetation(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (g > r + 8) & (g > b + 6) & (g > 60)


def _shadow(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Très sombre et bord saillant à proximité = ombre portée."""
    dark = gray < max(35, int(np.percentile(gray, 8)))
    r = rgb[..., 0].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    blueish = (b - r) > 4  # ombre bleutée typique
    return dark & blueish


def _asphalt_raw(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Gris uniforme, peu saturé, ni vert ni très sombre ni très clair."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    low_sat = (np.abs(r - g) < 28) & (np.abs(g - b) < 28) & (np.abs(r - b) < 28)
    mid_lum = (gray > 55) & (gray < 175)
    return low_sat & mid_lum


def _building_contours(gray: np.ndarray, m_per_px: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Trouve contours fermés quasi-rectangulaires de taille bâtiment → masque toits.

    Retourne (roof_mask H×W bool, building_edge_mask H×W bool, edge_likelihood 0-1).
    """
    h, w = gray.shape
    if not _HAS_CV2:
        return (
            np.zeros((h, w), dtype=np.bool_),
            np.zeros((h, w), dtype=np.bool_),
            0.0,
        )

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    sigma = 0.33
    lo = max(20, int((1.0 - sigma) * med))
    hi = max(lo + 10, int((1.0 + sigma) * med))
    edges = cv2.Canny(blur, lo, hi)
    edge_density = float((edges > 0).mean())

    # Dilate pour fermer les contours de toits
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    # Contours externes
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roof_mask = np.zeros_like(gray, dtype=np.uint8)
    min_area_m2 = 25.0  # plus petit qu'un bâtiment correct
    max_area_m2 = max(2500.0, (h * w) * (m_per_px ** 2) * 0.45)
    px2_to_m2 = m_per_px * m_per_px

    for cnt in contours:
        area_px = float(cv2.contourArea(cnt))
        area_m2 = area_px * px2_to_m2
        if area_m2 < min_area_m2 or area_m2 > max_area_m2:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri < 1e-3:
            continue
        approx = cv2.approxPolyDP(cnt, 0.022 * peri, True)
        n_corners = len(approx)
        # Toits = 4-10 angles dominants, compacité raisonnable
        if n_corners < 4 or n_corners > 12:
            continue
        # Compacité : 4πA / P² ; un rectangle ~ 0.78 — trop bas = forme allongée bizarre
        comp = (4.0 * math.pi * area_px) / max(peri * peri, 1.0)
        if comp < 0.20:
            continue
        # Vérifier qu'au moins 2 paires de côtés ~ parallèles (≥ aspect rectangulaire)
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        if min(rw, rh) <= 1.0:
            continue
        ratio = max(rw, rh) / max(min(rw, rh), 1.0)
        if ratio > 12.0:  # trop allongé = c'est une route, pas un toit
            continue
        cv2.drawContours(roof_mask, [cnt], -1, color=1, thickness=cv2.FILLED)

    edge_likelihood = float(np.clip(edge_density / 0.15, 0.0, 1.0))
    return roof_mask.astype(np.bool_), edges.astype(np.bool_), edge_likelihood


def _road_mask(
    asphalt: np.ndarray,
    roof: np.ndarray,
    m_per_px: float,
) -> np.ndarray:
    """Bandes asphaltées fines et allongées = chaussée.

    Heuristique : composantes connexes d'asphalte (hors toit) au moins 12 m de long et
    rapport long/large ≥ 6 ; ou touchant le bord du chip et traversant.
    """
    h, w = asphalt.shape
    if not _HAS_CV2 or asphalt.dtype != np.bool_:
        return np.zeros((h, w), dtype=np.bool_)
    candidate = (asphalt & ~roof).astype(np.uint8)
    if candidate.sum() == 0:
        return candidate.astype(np.bool_)

    # Composantes connexes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    road = np.zeros_like(candidate)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 50:
            continue
        long_side_m = max(ww, hh) * m_per_px
        short_side_m = max(min(ww, hh) * m_per_px, 1e-6)
        if long_side_m < 12.0:
            continue
        # bande étroite ET longue OU touche les 2 bords opposés
        aspect = long_side_m / short_side_m
        touches_border = (x == 0) or (y == 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if aspect >= 6.0 and short_side_m < 9.0:
            road[labels == i] = 1
        elif touches_border and aspect >= 3.5 and short_side_m < 10.0:
            road[labels == i] = 1
    return road.astype(np.bool_)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Suppression bruit : ouverture / fermeture morphologiques."""
    if not _HAS_CV2:
        return mask
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(np.bool_)


def classify_surfaces(
    rgb: np.ndarray,
    *,
    m_per_px: float,
    segformer_parking_mask: Optional[np.ndarray] = None,
) -> SurfaceClassification:
    """Calcule masques + indices pour distinguer surfaces garables / non-garables.

    ``segformer_parking_mask`` (H×W bool) : si fourni, sert d'amorce — les pixels SegFormer
    « parking » qui passent les filtres restent en parking_eligible.
    """
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w = rgb.shape[:2]

    if _HAS_CV2:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)

    veg = _vegetation(rgb)
    shadow = _shadow(rgb, gray)
    asph_raw = _asphalt_raw(rgb, gray) & ~veg & ~shadow
    asph = _clean_mask(asph_raw)

    roof, building_edges, edge_lh = _building_contours(gray, m_per_px)
    road = _road_mask(asph, roof, m_per_px)

    # Parking-eligible = bitume - toits - chaussée - végétation - ombre
    parking_eligible = asph & ~roof & ~road & ~veg & ~shadow
    parking_eligible = _clean_mask(parking_eligible)

    # Si SegFormer a un masque parking utilisable, on l'utilise pour renforcer la classification.
    if segformer_parking_mask is not None and segformer_parking_mask.shape == parking_eligible.shape:
        seg = segformer_parking_mask.astype(np.bool_) & ~roof & ~veg & ~shadow
        # Intersection prudente : SegFormer ne classifie pas les chaussées comme parking en général,
        # on garde donc l'OR (asph_eligible ∪ seg_parking_filtré).
        parking_eligible = parking_eligible | seg
        parking_eligible = _clean_mask(parking_eligible)

    total = max(h * w, 1)
    return SurfaceClassification(
        asphalt_mask=asph,
        roof_mask=roof,
        road_mask=road,
        vegetation_mask=veg,
        shadow_mask=shadow,
        building_edge_mask=building_edges,
        parking_eligible_mask=parking_eligible,
        asphalt_likelihood=float(np.clip(asph.sum() / total, 0.0, 1.0)),
        roof_likelihood=float(np.clip(roof.sum() / total, 0.0, 1.0)),
        road_likelihood=float(np.clip(road.sum() / total, 0.0, 1.0)),
        vegetation_likelihood=float(np.clip(veg.sum() / total, 0.0, 1.0)),
        shadow_likelihood=float(np.clip(shadow.sum() / total, 0.0, 1.0)),
        building_edge_likelihood=edge_lh,
    )
