"""Détection de véhicules sur orthophoto.

Deux pistes :
1. **YOLO** si poids fournis (ultralytics, classe ``car``/``vehicle``/``truck``) — détecteur sémantique.
2. **Fallback OpenCV** : détection de blobs rectangulaires sur asphalte, filtrés par taille
   (2–3 m × 4–5 m → ~8–15 m² au sol) et aspect ratio (1.4–3.0).

Les véhicules détectés servent de **preuve d'usage parking** : ils ne donnent pas directement la
capacité, mais valident qu'une zone bitumée est effectivement utilisée pour stationner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


VEHICLE_AREA_MIN_M2 = 4.0    # camionnette serrée
VEHICLE_AREA_MAX_M2 = 20.0   # gros utilitaire
VEHICLE_LENGTH_MIN_M = 2.5
VEHICLE_LENGTH_MAX_M = 6.5
VEHICLE_WIDTH_MIN_M = 1.2
VEHICLE_WIDTH_MAX_M = 3.5
VEHICLE_ASPECT_MIN = 1.3
VEHICLE_ASPECT_MAX = 3.0


@dataclass
class Vehicle:
    """Une voiture/camion détectée : centre + bbox + orientation."""

    cx: float
    cy: float
    width_px: float
    height_px: float
    angle_deg: float  # orientation longueur (0-180)
    area_px: float
    confidence: float  # 0-1 (heuristique pour fallback)


@dataclass
class VehicleCluster:
    """Un alignement de véhicules le long d'un axe."""

    members: List[Vehicle]
    orientation_deg: float
    length_m: float
    extent_perp_m: float


@dataclass
class VehicleDetectionResult:
    vehicles: List[Vehicle] = field(default_factory=list)
    clusters: List[VehicleCluster] = field(default_factory=list)
    vehicle_count: int = 0
    vehicle_density_score: float = 0.0  # 0-1 : véhicules détectés / aire asphalt
    vehicle_alignment_score: float = 0.0  # 0-1 : fraction véhicules en alignement
    method: str = "none"  # "yolo" | "opencv_fallback" | "none"


# -----------------------------
# Fallback OpenCV
# -----------------------------

def _detect_vehicles_opencv(
    rgb: np.ndarray,
    asphalt_mask: np.ndarray,
    *,
    m_per_px: float,
) -> List[Vehicle]:
    """Détection naïve : blobs contrastés sur asphalte de taille véhicule.

    Combine deux signaux :
    - blobs **sombres** (voitures foncées sur asphalte clair) via seuil bas.
    - blobs **clairs** (voitures claires sur asphalte sombre) via seuil haut.

    Filtré par dimensions plausibles. Pas un vrai détecteur d'objets — un compteur d'évidence.
    """
    if not _HAS_CV2 or asphalt_mask.size == 0:
        return []

    h, w = asphalt_mask.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Stat sur la zone asphalt seulement
    asph_vals = gray[asphalt_mask]
    if asph_vals.size < 100:
        return []
    asph_med = float(np.median(asph_vals))
    asph_std = max(float(np.std(asph_vals)), 8.0)

    # Seuils : 1.2 sigma de part et d'autre
    dark_th = max(20.0, asph_med - 1.2 * asph_std)
    bright_th = min(235.0, asph_med + 1.2 * asph_std)

    dark = ((gray < dark_th) & asphalt_mask).astype(np.uint8)
    bright = ((gray > bright_th) & asphalt_mask).astype(np.uint8)

    # Ouverture légère pour le grain, fermeture pour combler vitrage/pare-brise.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=1)

    px2_to_m2 = m_per_px * m_per_px

    vehicles: List[Vehicle] = []
    for mask, kind in ((dark, "dark"), (bright, "bright")):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            area_m2 = area_px * px2_to_m2
            if area_m2 < VEHICLE_AREA_MIN_M2 or area_m2 > VEHICLE_AREA_MAX_M2:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), ang = rect
            if rw <= 1 or rh <= 1:
                continue
            longe = max(rw, rh)
            shorte = min(rw, rh)
            longe_m = longe * m_per_px
            shorte_m = shorte * m_per_px
            if longe_m < VEHICLE_LENGTH_MIN_M or longe_m > VEHICLE_LENGTH_MAX_M:
                continue
            if shorte_m < VEHICLE_WIDTH_MIN_M or shorte_m > VEHICLE_WIDTH_MAX_M:
                continue
            aspect = longe / shorte
            if aspect < VEHICLE_ASPECT_MIN or aspect > VEHICLE_ASPECT_MAX:
                continue
            # Confidence : centré sur la fourchette canonique
            mid_area = (VEHICLE_AREA_MIN_M2 + VEHICLE_AREA_MAX_M2) / 2.0
            score_area = 1.0 - min(abs(area_m2 - mid_area) / mid_area, 1.0)
            mid_aspect = 2.0
            score_aspect = 1.0 - min(abs(aspect - mid_aspect) / mid_aspect, 1.0)
            conf = float(np.clip(0.5 * (score_area + score_aspect), 0.0, 1.0))
            # rect.angle est l'angle du côté width ; on normalise sur l'orientation du côté long.
            angle_deg = float(ang)
            if rw < rh:
                angle_deg = (angle_deg + 90.0) % 180.0
            angle_deg = angle_deg % 180.0
            vehicles.append(
                Vehicle(
                    cx=float(cx),
                    cy=float(cy),
                    width_px=float(rw),
                    height_px=float(rh),
                    angle_deg=angle_deg,
                    area_px=area_px,
                    confidence=conf,
                )
            )

    # Déduplication grossière : si deux véhicules sont à <2m l'un de l'autre, garde le plus
    # confiant.
    px_per_m = 1.0 / max(m_per_px, 1e-6)
    min_dist_px = 2.0 * px_per_m
    vehicles.sort(key=lambda v: -v.confidence)
    kept: List[Vehicle] = []
    for v in vehicles:
        if any(math.hypot(v.cx - k.cx, v.cy - k.cy) < min_dist_px for k in kept):
            continue
        kept.append(v)
    return kept


# -----------------------------
# YOLO (optionnel)
# -----------------------------

def _detect_vehicles_yolo(image: Image.Image, weights: Path) -> List[Vehicle]:
    """YOLO ultralytics ; les classes COCO ``car``/``truck``/``bus``/``motorcycle`` deviennent véhicule."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return []
    if not weights.is_file():
        return []
    try:
        model = YOLO(str(weights))
        arr = np.asarray(image.convert("RGB"))
        results = model.predict(arr, conf=0.20, verbose=False)
    except Exception:  # noqa: BLE001
        return []

    vehicle_classes = {"car", "truck", "bus", "motorcycle", "vehicle"}
    out: List[Vehicle] = []
    for r in results:
        names = r.names if hasattr(r, "names") else {}
        if r.boxes is None:
            continue
        for b in r.boxes:
            try:
                cls_id = int(b.cls.cpu().numpy().flatten()[0])
                name = str(names.get(cls_id, "")).lower()
                if name and name not in vehicle_classes:
                    continue
                xyxy = b.xyxy.cpu().numpy().flatten().tolist()
                conf = float(b.conf.cpu().numpy().flatten()[0])
            except Exception:  # noqa: BLE001
                continue
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = xyxy[:4]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            w = max(abs(x2 - x1), 1.0)
            h = max(abs(y2 - y1), 1.0)
            longe = max(w, h)
            shorte = min(w, h)
            angle = 0.0 if w >= h else 90.0
            area = w * h
            out.append(Vehicle(cx, cy, w, h, angle, area, conf))
    return out


# -----------------------------
# Alignement / clusters
# -----------------------------

def _cluster_aligned(vehicles: List[Vehicle], m_per_px: float) -> List[VehicleCluster]:
    """Groupe véhicules par orientation puis par projection sur l'axe perpendiculaire."""
    if len(vehicles) < 2:
        return []
    out: List[VehicleCluster] = []

    # Histogramme angulaire 10°
    bins = np.arange(0, 191, 10)
    angs = np.array([v.angle_deg for v in vehicles])
    hist, _ = np.histogram(angs, bins=bins)
    if hist.size == 0:
        return []
    # Orientations dominantes (>= 2 véhicules)
    for idx in np.argsort(hist)[::-1]:
        if hist[idx] < 2:
            break
        target = float(bins[idx] + 5.0)
        members = [v for v in vehicles if abs((v.angle_deg - target + 90) % 180 - 90) < 18.0]
        if len(members) < 2:
            continue
        # Projection perpendiculaire (axe normal à l'orientation)
        rad = math.radians(target + 90.0)
        nx, ny = math.cos(rad), math.sin(rad)
        projs = [(m.cx * nx + m.cy * ny, m) for m in members]
        projs.sort()
        # Cluster 1D : sous-groupes de véhicules dont la projection diffère de < 4 m
        cluster_gap = 4.0 / max(m_per_px, 1e-6)
        current = [projs[0]]
        groups: List[List[Tuple[float, Vehicle]]] = []
        for prev, nxt in zip(projs, projs[1:]):
            if (nxt[0] - prev[0]) <= cluster_gap:
                current.append(nxt)
            else:
                groups.append(current)
                current = [nxt]
        groups.append(current)
        for grp in groups:
            if len(grp) < 2:
                continue
            grp_members = [v for _, v in grp]
            rad_along = math.radians(target)
            tx, ty = math.cos(rad_along), math.sin(rad_along)
            extents = [v.cx * tx + v.cy * ty for v in grp_members]
            perp = [v.cx * nx + v.cy * ny for v in grp_members]
            length_m = float(max(extents) - min(extents)) * m_per_px
            extent_perp_m = float(max(perp) - min(perp)) * m_per_px
            out.append(
                VehicleCluster(
                    members=grp_members,
                    orientation_deg=target,
                    length_m=length_m,
                    extent_perp_m=extent_perp_m,
                )
            )
    return out


# -----------------------------
# API publique
# -----------------------------

def detect_vehicles(
    image: Image.Image,
    *,
    asphalt_mask: Optional[np.ndarray],
    m_per_px: float,
    yolo_weights: Optional[Path] = None,
) -> VehicleDetectionResult:
    """Pipeline complet : YOLO si poids dispo, sinon OpenCV ; clusters + scores."""
    vehicles: List[Vehicle] = []
    method = "none"

    if yolo_weights is not None:
        vehicles = _detect_vehicles_yolo(image, yolo_weights)
        if vehicles:
            method = "yolo"

    if not vehicles and asphalt_mask is not None and asphalt_mask.sum() > 0:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        vehicles = _detect_vehicles_opencv(rgb, asphalt_mask.astype(np.bool_), m_per_px=m_per_px)
        if vehicles:
            method = "opencv_fallback"

    clusters = _cluster_aligned(vehicles, m_per_px) if vehicles else []

    # Scores
    if asphalt_mask is not None and asphalt_mask.sum() > 0:
        asph_m2 = float(asphalt_mask.sum()) * (m_per_px ** 2)
        density = len(vehicles) / max(asph_m2 / 25.0, 1.0)  # référence : 1 véhicule / 25 m²
        density_score = float(np.clip(density, 0.0, 1.0))
    else:
        density_score = 0.0

    if vehicles:
        n_clustered = sum(len(c.members) for c in clusters)
        alignment_score = float(np.clip(n_clustered / max(len(vehicles), 1), 0.0, 1.0))
    else:
        alignment_score = 0.0

    return VehicleDetectionResult(
        vehicles=vehicles,
        clusters=clusters,
        vehicle_count=len(vehicles),
        vehicle_density_score=round(density_score, 3),
        vehicle_alignment_score=round(alignment_score, 3),
        method=method,
    )
