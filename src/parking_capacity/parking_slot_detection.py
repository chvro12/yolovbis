"""Détection de places de parking marquées (pleines + vides) sur orthophoto.

Trois pistes :

1. **YOLO fine-tuné** (poids fournis ou Roboflow Universe) :
   - Détecte places **pleines** et **vides** avec bbox orientée.
   - Classes attendues : ``parking_space_empty`` / ``parking_space_filled`` / ``parking_slot`` / ``parking``.
2. **Heuristique vehicles ⊕ géométrie** :
   - Combine les rangées géométriques détectées (longueur, orientation, slot_width) avec les
     véhicules détectés (`vehicle_detection`).
   - Places remplies ≈ véhicules visibles dans une rangée.
   - Places vides ≈ (longueur rangée / slot_width) − places_remplies.
   - Sortie : **un comptage prudent** clampé par le plafond physique de la couche sémantique.
3. **Aucune source dispo** : retourne un résultat vide.

L'avantage du mode 2 (heuristique) : il **fonctionne dès maintenant** sans poids, en combinant les
signaux déjà calculés par le pipeline.
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


SLOT_WIDTH_TYP_M = 2.5
SLOT_WIDTH_MIN_M = 2.3
SLOT_WIDTH_MAX_M = 2.7

YOLO_SLOT_CLASSES = {
    "parking_space_empty",
    "parking_space_filled",
    "parking_space",
    "parking_slot",
    "empty_slot",
    "filled_slot",
    "occupied",
    "vacant",
    "parking",
}
YOLO_EMPTY_KEYWORDS = ("empty", "vacant", "free", "unoccupied")
YOLO_FILLED_KEYWORDS = ("filled", "occupied", "taken")


@dataclass
class Slot:
    """Une place de parking détectée (bbox xyxy en pixels image)."""

    cx: float
    cy: float
    width_px: float
    height_px: float
    angle_deg: float
    status: str  # "empty" | "filled" | "unknown"
    confidence: float
    source: str  # "yolo" | "heuristic" | "roboflow"


@dataclass
class SlotDetectionResult:
    slots: List[Slot] = field(default_factory=list)
    slots_filled_count: int = 0
    slots_empty_count: int = 0
    slots_total_count: int = 0
    method: str = "none"  # "yolo" | "yolo_sahi" | "roboflow" | "heuristic_vehicles_rows" | "none"
    notes: List[str] = field(default_factory=list)


# -----------------------------
# Heuristique : véhicules + rangées
# -----------------------------

def _row_axes_pixels(row_lengths_m: List[float], orientation_deg: float, m_per_px: float, chip_w: int, chip_h: int) -> List[dict]:
    """Reconstruit la position approximative des rangées centrales dans l'image.

    Sans coordonnées d'origine, on suppose les rangées centrées dans la chip et empilées.
    C'est une approximation : pour une vraie association vehicle→row il faudrait le
    `RowCandidate.center_proj` mais celui-ci n'est pas exposé dans le pipeline actuel.
    """
    if not row_lengths_m:
        return []
    n = len(row_lengths_m)
    spacing_m = 7.0  # ~6-8 m entre rangées dos à dos
    cx_img = chip_w / 2.0
    cy_img = chip_h / 2.0
    rad = math.radians(orientation_deg)
    tx, ty = math.cos(rad), math.sin(rad)
    rad_perp = math.radians(orientation_deg + 90.0)
    nx, ny = math.cos(rad_perp), math.sin(rad_perp)
    spacing_px = spacing_m / max(m_per_px, 1e-6)

    rows = []
    for i, L in enumerate(row_lengths_m):
        offset = (i - (n - 1) / 2.0) * spacing_px
        cx = cx_img + nx * offset
        cy = cy_img + ny * offset
        half_len_px = (L / max(m_per_px, 1e-6)) * 0.5
        rows.append({
            "cx": cx, "cy": cy, "tx": tx, "ty": ty,
            "nx": nx, "ny": ny,
            "length_m": L, "half_len_px": half_len_px,
            "slot_count_geom": max(1, int(round(L / SLOT_WIDTH_TYP_M))),
        })
    return rows


def _assign_vehicles_to_rows(vehicles, rows_meta: List[dict], m_per_px: float) -> List[int]:
    """Pour chaque rangée, compte le nombre de véhicules dont le centre est proche."""
    counts = [0] * len(rows_meta)
    if not vehicles or not rows_meta:
        return counts
    tol_perp_px = 5.0 / max(m_per_px, 1e-6)  # 5 m tolérance perpendiculaire
    for v in vehicles:
        best_idx = -1
        best_d = float("inf")
        for i, r in enumerate(rows_meta):
            dx = v.cx - r["cx"]
            dy = v.cy - r["cy"]
            along = dx * r["tx"] + dy * r["ty"]
            perp = dx * r["nx"] + dy * r["ny"]
            if abs(along) > r["half_len_px"]:
                continue
            if abs(perp) > tol_perp_px:
                continue
            if abs(perp) < best_d:
                best_d = abs(perp)
                best_idx = i
        if best_idx >= 0:
            counts[best_idx] += 1
    return counts


def detect_slots_heuristic(
    chip_image: Image.Image,
    *,
    row_lengths_m: List[float],
    row_orientation_deg: float,
    vehicles,
    m_per_px: float,
    plausible_ceiling: Optional[int] = None,
    parcelle_polygon_px: Optional[np.ndarray] = None,
) -> SlotDetectionResult:
    """Combine rangées géométriques + véhicules → comptage places pleines + vides.

    Très prudent : applique le plafond ``plausible_ceiling`` (couche sémantique) si fourni.
    """
    if not row_lengths_m:
        return SlotDetectionResult(method="none", notes=["pas_de_rangees_geom"])

    w, h = chip_image.size
    rows_meta = _row_axes_pixels(row_lengths_m, row_orientation_deg, m_per_px, w, h)
    veh_per_row = _assign_vehicles_to_rows(vehicles, rows_meta, m_per_px)

    slots_total = 0
    slots_filled = 0
    slots: List[Slot] = []

    rad = math.radians(row_orientation_deg)
    tx, ty = math.cos(rad), math.sin(rad)
    slot_px = SLOT_WIDTH_TYP_M / max(m_per_px, 1e-6)

    for i, r in enumerate(rows_meta):
        n_geom = r["slot_count_geom"]
        n_filled = min(veh_per_row[i], n_geom)
        n_empty = max(0, n_geom - n_filled)
        slots_total += n_geom
        slots_filled += n_filled

        # Génère des slots synthétiques le long de la rangée (pour visualisation)
        cx0 = r["cx"] - r["tx"] * r["half_len_px"]
        cy0 = r["cy"] - r["ty"] * r["half_len_px"]
        for k in range(n_geom):
            sc_x = cx0 + r["tx"] * slot_px * (k + 0.5)
            sc_y = cy0 + r["ty"] * slot_px * (k + 0.5)
            status = "filled" if k < n_filled else "empty"
            slots.append(Slot(
                cx=float(sc_x), cy=float(sc_y),
                width_px=float(slot_px), height_px=float(slot_px * 2.0),
                angle_deg=float(row_orientation_deg),
                status=status, confidence=0.45, source="heuristic",
            ))

    # Filtre parcelle
    if parcelle_polygon_px is not None and _HAS_CV2 and parcelle_polygon_px.shape[0] >= 3:
        poly = parcelle_polygon_px.astype(np.int32)
        filtered: List[Slot] = []
        for s in slots:
            if cv2.pointPolygonTest(poly, (float(s.cx), float(s.cy)), False) >= 0:
                filtered.append(s)
        slots = filtered
        slots_filled = sum(1 for s in slots if s.status == "filled")
        slots_total = len(slots)

    slots_empty = slots_total - slots_filled

    # Plafond physique
    if plausible_ceiling is not None and slots_total > plausible_ceiling:
        # Tronque proportionnellement
        ratio = plausible_ceiling / max(slots_total, 1)
        slots_total = int(plausible_ceiling)
        slots_filled = int(round(slots_filled * ratio))
        slots_empty = slots_total - slots_filled
        slots = slots[:slots_total]

    return SlotDetectionResult(
        slots=slots,
        slots_filled_count=slots_filled,
        slots_empty_count=slots_empty,
        slots_total_count=slots_total,
        method="heuristic_vehicles_rows",
        notes=[],
    )


# -----------------------------
# YOLO (optionnel) — slot detection
# -----------------------------

def _classify_slot_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in YOLO_FILLED_KEYWORDS):
        return "filled"
    if any(k in n for k in YOLO_EMPTY_KEYWORDS):
        return "empty"
    return "unknown"


def _detect_slots_yolo(
    image: Image.Image,
    weights: Path,
    *,
    conf_th: float = 0.20,
    use_sahi: bool = True,
    sahi_slice_px: int = 512,
) -> Tuple[List[Slot], str]:
    """Inférence Ultralytics (avec SAHI optionnel) pour slot detection."""
    if use_sahi:
        try:
            from sahi import AutoDetectionModel  # type: ignore
            from sahi.predict import get_sliced_prediction  # type: ignore
        except ImportError:
            use_sahi = False

    if use_sahi:
        try:
            det = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(weights),
                confidence_threshold=conf_th,
                device="cpu",
            )
            arr = np.asarray(image.convert("RGB"))
            result = get_sliced_prediction(
                arr, det,
                slice_height=sahi_slice_px, slice_width=sahi_slice_px,
                overlap_height_ratio=0.20, overlap_width_ratio=0.20, verbose=0,
            )
        except Exception:  # noqa: BLE001
            return [], "yolo_load_failed"

        out: List[Slot] = []
        for op in result.object_prediction_list:
            try:
                name = (op.category.name or "").lower()
            except AttributeError:
                name = ""
            if not any(k in name for k in YOLO_SLOT_CLASSES):
                continue
            box = op.bbox.to_xyxy()
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            w = max(abs(x2 - x1), 1.0)
            h = max(abs(y2 - y1), 1.0)
            angle = 0.0 if w >= h else 90.0
            out.append(Slot(
                cx=cx, cy=cy, width_px=w, height_px=h, angle_deg=angle,
                status=_classify_slot_name(name),
                confidence=float(op.score.value), source="yolo",
            ))
        return out, "yolo_sahi"

    # Sans SAHI : YOLO direct
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return [], "yolo_no_lib"
    try:
        model = YOLO(str(weights))
        arr = np.asarray(image.convert("RGB"))
        results = model.predict(arr, conf=conf_th, verbose=False)
    except Exception:  # noqa: BLE001
        return [], "yolo_inference_failed"

    out = []
    for r in results:
        names = r.names if hasattr(r, "names") else {}
        if r.boxes is None:
            continue
        for b in r.boxes:
            try:
                cls_id = int(b.cls.cpu().numpy().flatten()[0])
                cname = str(names.get(cls_id, "")).lower()
                if not any(k in cname for k in YOLO_SLOT_CLASSES):
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
            out.append(Slot(
                cx=cx, cy=cy, width_px=w, height_px=h, angle_deg=0.0,
                status=_classify_slot_name(cname),
                confidence=conf, source="yolo",
            ))
    return out, "yolo_direct"


# -----------------------------
# Roboflow Universe (optionnel)
# -----------------------------

def _detect_slots_roboflow(
    image: Image.Image,
    *,
    api_key: str,
    model_id: str,  # "workspace/project/version" ou nom direct
    conf_th: float = 0.30,
) -> Tuple[List[Slot], str]:
    """Roboflow Universe inference via leur API HTTP.

    Format model_id : "workspace/project/version" (ex. "smart-parking/aerial-parking-slots/3").
    Requires ``inference_sdk`` ou simplement requête HTTP REST.
    """
    try:
        from inference_sdk import InferenceHTTPClient  # type: ignore
    except ImportError:
        return [], "roboflow_no_sdk"
    try:
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key,
        )
        arr = np.asarray(image.convert("RGB"))
        result = client.infer(arr, model_id=model_id)
    except Exception:  # noqa: BLE001
        return [], "roboflow_inference_failed"

    out: List[Slot] = []
    preds = result.get("predictions", []) if isinstance(result, dict) else []
    for p in preds:
        try:
            cname = str(p.get("class", "")).lower()
            conf = float(p.get("confidence", 0.0))
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            w = float(p.get("width", 1.0))
            h = float(p.get("height", 1.0))
        except (TypeError, ValueError):
            continue
        if conf < conf_th:
            continue
        out.append(Slot(
            cx=x, cy=y, width_px=w, height_px=h, angle_deg=0.0,
            status=_classify_slot_name(cname),
            confidence=conf, source="roboflow",
        ))
    return out, "roboflow"


def _clip_slots_to_polygon(slots: List[Slot], polygon_px: np.ndarray) -> List[Slot]:
    if polygon_px is None or polygon_px.shape[0] < 3 or not _HAS_CV2:
        return slots
    poly = polygon_px.astype(np.int32)
    return [s for s in slots if cv2.pointPolygonTest(poly, (float(s.cx), float(s.cy)), False) >= 0]


# -----------------------------
# API publique
# -----------------------------

def detect_slots(
    image: Image.Image,
    *,
    m_per_px: float,
    row_lengths_m: List[float],
    row_orientation_deg: float,
    vehicles=None,
    plausible_ceiling: Optional[int] = None,
    parcelle_polygon_px: Optional[np.ndarray] = None,
    yolo_weights: Optional[Path] = None,
    roboflow_api_key: Optional[str] = None,
    roboflow_model_id: Optional[str] = None,
) -> SlotDetectionResult:
    """Pipeline complet : YOLO → Roboflow → heuristique véhicules+rangées.

    Le mode **heuristique** fonctionne dès maintenant sans poids — il combine les rangées
    géométriques déjà détectées avec les véhicules de Phase 2.
    """
    notes: List[str] = []

    # 1) YOLO si poids dispo
    if yolo_weights is not None and Path(yolo_weights).is_file():
        slots, method = _detect_slots_yolo(image, Path(yolo_weights))
        if slots:
            slots = _clip_slots_to_polygon(slots, parcelle_polygon_px) if parcelle_polygon_px is not None else slots
            filled = sum(1 for s in slots if s.status == "filled")
            empty = sum(1 for s in slots if s.status == "empty")
            unknown = sum(1 for s in slots if s.status == "unknown")
            total = len(slots)
            return SlotDetectionResult(
                slots=slots,
                slots_filled_count=filled,
                slots_empty_count=empty + unknown,  # places "unknown" comptées comme vides
                slots_total_count=total,
                method=method,
                notes=notes,
            )
        notes.append(f"yolo_indispo:{method}")

    # 2) Roboflow Universe si clé fournie
    if roboflow_api_key and roboflow_model_id:
        slots, method = _detect_slots_roboflow(
            image, api_key=roboflow_api_key, model_id=roboflow_model_id,
        )
        if slots:
            slots = _clip_slots_to_polygon(slots, parcelle_polygon_px) if parcelle_polygon_px is not None else slots
            filled = sum(1 for s in slots if s.status == "filled")
            empty = sum(1 for s in slots if s.status == "empty")
            unknown = sum(1 for s in slots if s.status == "unknown")
            return SlotDetectionResult(
                slots=slots,
                slots_filled_count=filled,
                slots_empty_count=empty + unknown,
                slots_total_count=len(slots),
                method=method,
                notes=notes,
            )
        notes.append(f"roboflow_indispo:{method}")

    # 3) Heuristique : véhicules + rangées
    result = detect_slots_heuristic(
        image,
        row_lengths_m=row_lengths_m,
        row_orientation_deg=row_orientation_deg,
        vehicles=vehicles or [],
        m_per_px=m_per_px,
        plausible_ceiling=plausible_ceiling,
        parcelle_polygon_px=parcelle_polygon_px,
    )
    if notes:
        result.notes = list(result.notes) + notes
    return result
