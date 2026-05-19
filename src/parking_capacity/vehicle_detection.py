"""Détection de véhicules sur orthophoto.

Trois pistes (priorité décroissante) :

1. **YOLO + SAHI** (Slicing Aided Hyper Inference) — détecteur Ultralytics découpé en patchs pour
   ne pas rater les petits véhicules sur grande orthophoto. Active si ``yolo_weights`` est fourni
   *ou* si ``auto_download`` est activé (télécharge ``yolov8s.pt`` Ultralytics qui détecte
   ``car/truck/bus/motorcycle`` en COCO).
2. **YOLO simple** (sans SAHI) si SAHI indisponible.
3. **Fallback OpenCV** : blobs rectangulaires contrastés sur asphalte (heuristique, peu fiable).

Pour des résultats vraiment robustes sur orthophoto BD ORTHO (0.1-0.3 m/px), on recommande de
fine-tuner sur **CARPK** ou **DOTA-v2** — voir ``docs/aerial_vehicle_detection.md``.

Les véhicules détectés alimentent la couche sémantique (preuve d'usage parking, score d'alignement)
mais ne pilotent pas la capacité principale.
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
# YOLO + SAHI (optionnel)
# -----------------------------

VEHICLE_COCO_CLASSES = {"car", "truck", "bus", "motorcycle", "vehicle"}
# Classes VisDrone (drone aérien) : modèle mshamrai/yolov8s-visdrone
# 0:pedestrian 1:people 2:bicycle 3:car 4:van 5:truck 6:tricycle 7:awning-tricycle 8:bus 9:motor
VEHICLE_VISDRONE_CLASSES = {"car", "van", "truck", "bus", "motor"}
ALL_VEHICLE_CLASSES = VEHICLE_COCO_CLASSES | VEHICLE_VISDRONE_CLASSES

DEFAULT_YOLO_MODEL = "yolov8s.pt"  # Ultralytics COCO (faible sur aérien)
DEFAULT_AERIAL_HF_REPO = "mshamrai/yolov8s-visdrone"  # VisDrone (drone aérien, vraies voitures)
DEFAULT_AERIAL_HF_FILENAME = "best.pt"

# Chemin local d'un éventuel modèle fine-tuné sur chips BD ORTHO françaises
# (self-pseudo-labeling, voir scripts/finetune_aerial_yolo.py).
FINETUNED_FRENCH_WEIGHTS = Path(
    "/Users/mac/Yolo/data/aerial_weights/finetuned_v1/run1/weights/best.pt"
)

# Fine-tuning sur DOTAv1 vehicles (51 700 véhicules réels annotés humainement, Wuhan University).
# On essaie run2 (paramètres optimisés MPS), puis run1 (legacy).
DOTA_FINETUNED_WEIGHTS_CANDIDATES = [
    Path("/Users/mac/Yolo/data/aerial_weights/dota_finetune_v1/run2/weights/best.pt"),
    Path("/Users/mac/Yolo/data/aerial_weights/dota_finetune_v1/run1/weights/best.pt"),
]
DOTA_FINETUNED_WEIGHTS = DOTA_FINETUNED_WEIGHTS_CANDIDATES[0]  # alias historique


def _resolve_finetuned_french_weights() -> Optional[str]:
    """Retourne le chemin du modèle fine-tuné français si présent, sinon None."""
    if FINETUNED_FRENCH_WEIGHTS.is_file():
        return str(FINETUNED_FRENCH_WEIGHTS)
    return None


def _resolve_dota_finetuned_weights() -> Optional[str]:
    """Retourne le chemin du modèle fine-tuné DOTA si présent (run2 puis run1)."""
    for candidate in DOTA_FINETUNED_WEIGHTS_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_yolo_weights(weights: Optional[Path], auto_download: bool) -> Optional[str]:
    """Retourne le chemin (ou nom de modèle Ultralytics) ou None si rien d'utilisable."""
    if weights is not None and Path(weights).is_file():
        return str(weights)
    if auto_download:
        # Ultralytics télécharge le poids automatiquement à la première utilisation.
        return DEFAULT_YOLO_MODEL
    return None


def _find_local_default_yolo_weights() -> Optional[str]:
    """Cherche ``yolov8s.pt`` dans le CWD ou à la racine du dépôt.

    Permet d'utiliser YOLO sans drapeau explicite quand le poids est posé à côté du projet
    (le cas par défaut après ``ultralytics`` qui télécharge dans le CWD).
    """
    candidates = [Path.cwd() / DEFAULT_YOLO_MODEL]
    try:
        candidates.append(Path(__file__).resolve().parents[2] / DEFAULT_YOLO_MODEL)
    except IndexError:
        pass
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _resolve_aerial_yolo_weights(cache_dir: Optional[Path] = None) -> Optional[str]:
    """Télécharge (et cache) le YOLO aérien VisDrone depuis HuggingFace Hub.

    Retourne le chemin local du fichier ``best.pt`` ou ``None`` si impossible.
    Le modèle VisDrone donne des détections beaucoup plus fiables sur orthophoto BD ORTHO
    que ``yolov8s.pt`` COCO (entraîné sur photos sol).
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None
    cache = str(cache_dir) if cache_dir else None
    try:
        return hf_hub_download(
            repo_id=DEFAULT_AERIAL_HF_REPO,
            filename=DEFAULT_AERIAL_HF_FILENAME,
            cache_dir=cache,
        )
    except Exception:  # noqa: BLE001
        return None


def _detect_vehicles_yolo_sahi(
    image: Image.Image,
    model_path: str,
    *,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_ratio: float = 0.20,
    conf_th: float = 0.25,
) -> Tuple[List[Vehicle], str]:
    """SAHI sliced inference avec un modèle Ultralytics. Retourne (vehicles, method)."""
    try:
        from sahi import AutoDetectionModel  # type: ignore
        from sahi.predict import get_sliced_prediction  # type: ignore
    except ImportError:
        return [], "yolo_no_sahi"

    try:
        det_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf_th,
            device="cpu",
        )
    except Exception:  # noqa: BLE001
        try:
            det_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=conf_th,
                device="cpu",
            )
        except Exception:  # noqa: BLE001
            return [], "yolo_load_failed"

    try:
        arr = np.asarray(image.convert("RGB"))
        result = get_sliced_prediction(
            arr,
            det_model,
            slice_height=int(slice_height),
            slice_width=int(slice_width),
            overlap_height_ratio=float(overlap_ratio),
            overlap_width_ratio=float(overlap_ratio),
            verbose=0,
        )
    except Exception:  # noqa: BLE001
        return [], "yolo_inference_failed"

    out: List[Vehicle] = []
    for op in result.object_prediction_list:
        try:
            name = (op.category.name or "").lower()
        except AttributeError:
            name = ""
        if name and name not in ALL_VEHICLE_CLASSES:
            continue
        try:
            box = op.bbox.to_xyxy()
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            conf = float(op.score.value)
        except Exception:  # noqa: BLE001
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(abs(x2 - x1), 1.0)
        h = max(abs(y2 - y1), 1.0)
        angle = 0.0 if w >= h else 90.0
        out.append(Vehicle(cx=cx, cy=cy, width_px=w, height_px=h, angle_deg=angle, area_px=w * h, confidence=conf))
    return out, "yolo_sahi"


def _detect_vehicles_yolo_direct(image: Image.Image, model_path: str, conf_th: float = 0.20) -> List[Vehicle]:
    """Inférence Ultralytics directe sans SAHI (fallback)."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return []
    try:
        model = YOLO(model_path)
        arr = np.asarray(image.convert("RGB"))
        results = model.predict(arr, conf=conf_th, verbose=False)
    except Exception:  # noqa: BLE001
        return []

    out: List[Vehicle] = []
    for r in results:
        names = r.names if hasattr(r, "names") else {}
        if r.boxes is None:
            continue
        for b in r.boxes:
            try:
                cls_id = int(b.cls.cpu().numpy().flatten()[0])
                name = str(names.get(cls_id, "")).lower()
                if name and name not in ALL_VEHICLE_CLASSES:
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
            angle = 0.0 if w >= h else 90.0
            out.append(Vehicle(cx, cy, w, h, angle, w * h, conf))
    return out


def _filter_by_size_m(vehicles: List[Vehicle], m_per_px: float, *, allow_large: bool = True) -> List[Vehicle]:
    """Garde uniquement les détections dont la bbox correspond à un véhicule plausible (m²).

    ``allow_large`` : si True, accepte aussi camions/utilitaires (jusqu'à 50 m²) — utile pour
    modèle VisDrone qui détecte trucks/vans/bus.
    """
    out: List[Vehicle] = []
    max_longe = VEHICLE_LENGTH_MAX_M * (2.5 if allow_large else 1.5)
    max_shorte = VEHICLE_WIDTH_MAX_M * (2.5 if allow_large else 1.5)
    max_area = 50.0 if allow_large else 35.0
    for v in vehicles:
        longe = max(v.width_px, v.height_px) * m_per_px
        shorte = min(v.width_px, v.height_px) * m_per_px
        area_m2 = v.area_px * (m_per_px ** 2)
        if longe < VEHICLE_LENGTH_MIN_M * 0.6 or longe > max_longe:
            continue
        if shorte < VEHICLE_WIDTH_MIN_M * 0.6 or shorte > max_shorte:
            continue
        if area_m2 < 2.0 or area_m2 > max_area:
            continue
        out.append(v)
    return out


def _clip_to_polygon_pixels(vehicles: List[Vehicle], polygon_px: np.ndarray) -> List[Vehicle]:
    """Filtre les véhicules dont le centre est à l'intérieur du polygone (pixels)."""
    if polygon_px is None or polygon_px.shape[0] < 3 or not _HAS_CV2:
        return vehicles
    poly = polygon_px.astype(np.int32)
    out: List[Vehicle] = []
    for v in vehicles:
        if cv2.pointPolygonTest(poly, (float(v.cx), float(v.cy)), False) >= 0:
            out.append(v)
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
    auto_download_yolo: bool = False,
    auto_download_aerial_yolo: bool = False,
    use_finetuned_french: bool = False,
    use_dota_finetuned: bool = False,
    aerial_weights_cache_dir: Optional[Path] = None,
    sahi_slice_px: int = 512,
    sahi_overlap: float = 0.20,
    conf_threshold: Optional[float] = None,
    parcelle_polygon_px: Optional[np.ndarray] = None,
) -> VehicleDetectionResult:
    """Pipeline complet : YOLO+SAHI → YOLO direct → OpenCV ; filtre par taille m² + polygone parcelle.

    ``parcelle_polygon_px`` : polygone N×2 (pixels relatifs à l'image), pour ne compter que les
    véhicules à l'intérieur de la parcelle cadastrale (recommandé fortement).
    """
    vehicles: List[Vehicle] = []
    method = "none"
    is_aerial = False

    # Priorité : poids fournis > DOTA fine-tuned > french fine-tuned > VisDrone HF Hub > COCO
    model_path: Optional[str] = None
    used_finetuned = False
    used_dota = False
    if yolo_weights is not None and Path(yolo_weights).is_file():
        model_path = str(yolo_weights)
        is_aerial = True
    elif use_dota_finetuned:
        dt = _resolve_dota_finetuned_weights()
        if dt is not None:
            model_path = dt
            is_aerial = True
            used_dota = True
        else:
            # Fallback VisDrone si DOTA pas dispo
            model_path = _resolve_aerial_yolo_weights(cache_dir=aerial_weights_cache_dir)
            is_aerial = model_path is not None
    elif use_finetuned_french:
        ft = _resolve_finetuned_french_weights()
        if ft is not None:
            model_path = ft
            is_aerial = True
            used_finetuned = True
        else:
            model_path = _resolve_aerial_yolo_weights(cache_dir=aerial_weights_cache_dir)
            is_aerial = model_path is not None
    elif auto_download_aerial_yolo:
        model_path = _resolve_aerial_yolo_weights(cache_dir=aerial_weights_cache_dir)
        is_aerial = model_path is not None
    elif auto_download_yolo:
        model_path = DEFAULT_YOLO_MODEL
        is_aerial = False

    # Dernier recours : poids local yolov8s.pt présent dans le CWD ou la racine du dépôt.
    # Évite que la détection retombe en silence sur ``opencv_fallback`` quand le poids
    # est disponible mais qu'aucun drapeau d'activation n'a été passé.
    if model_path is None:
        local = _find_local_default_yolo_weights()
        if local is not None:
            model_path = local
            is_aerial = False

    # Seuil par défaut : 0.10 aérien (modèle spécialisé), 0.07 COCO (récupère les faibles).
    # Le modèle COCO yolov8s.pt résolu via auto-download OU via découverte locale partage
    # le même seuil bas — un seuil 0.20 trop strict masquait les véhicules en orthophoto.
    if conf_threshold is None:
        if is_aerial:
            conf_threshold = 0.10
        elif model_path is not None and not is_aerial:
            conf_threshold = 0.07
        else:
            conf_threshold = 0.20

    if model_path is not None:
        sahi_vehs, sahi_method = _detect_vehicles_yolo_sahi(
            image,
            model_path,
            slice_height=sahi_slice_px,
            slice_width=sahi_slice_px,
            overlap_ratio=sahi_overlap,
            conf_th=conf_threshold,
        )
        if sahi_vehs:
            vehicles = sahi_vehs
            if used_dota:
                method = "yolo_dota_finetuned_sahi"
            elif used_finetuned:
                method = "yolo_finetuned_french_sahi"
            else:
                method = "yolo_visdrone_sahi" if is_aerial else sahi_method
        elif sahi_method in ("yolo_no_sahi", "yolo_load_failed"):
            direct = _detect_vehicles_yolo_direct(image, model_path, conf_th=conf_threshold)
            if direct:
                vehicles = direct
                if used_dota:
                    method = "yolo_dota_finetuned_direct"
                elif used_finetuned:
                    method = "yolo_finetuned_french_direct"
                else:
                    method = "yolo_visdrone_direct" if is_aerial else "yolo_direct"

    if not vehicles and asphalt_mask is not None and asphalt_mask.sum() > 0:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        vehicles = _detect_vehicles_opencv(rgb, asphalt_mask.astype(np.bool_), m_per_px=m_per_px)
        if vehicles:
            method = "opencv_fallback"

    # Filtre par taille au sol plausible (élimine les faux positifs YOLO)
    if vehicles:
        vehicles = _filter_by_size_m(vehicles, m_per_px)

    # Filtre par polygone parcelle (gros gain : exclut voirie + parkings voisins)
    if vehicles and parcelle_polygon_px is not None:
        before = len(vehicles)
        vehicles = _clip_to_polygon_pixels(vehicles, parcelle_polygon_px)
        if before > 0 and not vehicles:
            method = f"{method}_no_in_polygon"

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
