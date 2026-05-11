"""Estimation de surface parking via SegFormer (Hugging Face) + masque pour géométrie ROI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

from parking_capacity.imagery_wms import OrthoChip

logger = logging.getLogger(__name__)

# Le dépôt HF ne contient que ``best_model.ckpt`` (Lightning) : pas de preprocessor_config.json.
# On charge le préprocesseur + la config depuis le backbone MIT-B5 (512×512), puis les poids UTEL.
_UTEL_REPO_ID = "UTEL-UIUC/SegFormer-large-parking"
_UTEL_CKPT_FILENAME = "best_model.ckpt"
_SEGFORMER_BACKBONE_ID = "nvidia/mit-b5"

_processor: Any = None
_model: Any = None


def _strip_lightning_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
    return out


def _lazy_load(device: str | None = None) -> tuple[Any, SegformerForSemanticSegmentation]:
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    from huggingface_hub import hf_hub_download

    proc = AutoImageProcessor.from_pretrained(_SEGFORMER_BACKBONE_ID)

    cfg = SegformerConfig.from_pretrained(_SEGFORMER_BACKBONE_ID)
    cfg.num_labels = 2
    cfg.id2label = {0: "background", 1: "parking"}
    cfg.label2id = {"background": 0, "parking": 1}

    mdl = SegformerForSemanticSegmentation(cfg)
    ckpt_path = hf_hub_download(_UTEL_REPO_ID, _UTEL_CKPT_FILENAME)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = ckpt.get("state_dict") or ckpt
    if not isinstance(raw_sd, dict):
        raise RuntimeError("Checkpoint UTEL : state_dict introuvable.")
    sd = _strip_lightning_prefix(raw_sd) if any(k.startswith("model.") for k in raw_sd) else raw_sd
    mdl.load_state_dict(sd, strict=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.train(False)

    _processor, _model = proc, mdl
    logger.info(
        "SegFormer parking UTEL chargé (poids %s, preprocessor %s).",
        _UTEL_REPO_ID,
        _SEGFORMER_BACKBONE_ID,
    )
    return proc, mdl


def _parking_class_index(model: SegformerForSemanticSegmentation) -> int:
    id2label = model.config.id2label or {}
    for idx, name in id2label.items():
        low = str(name).lower()
        if "parking" in low or "lot" in low:
            return int(idx)
    if len(id2label) >= 2:
        return 1
    return 0


@dataclass
class VisionEstimate:
    """Indicateurs dérivés du masque (aire, fraction) — pas une vérité « nombre de places » seul."""

    parking_pixel_fraction: float
    parking_area_m2: float
    estimated_spaces: int
    confidence: str
    device: str
    notes: str


@dataclass
class SegformerChipOutput:
    """Segmentation + masque booléen (H×W) aligné sur la puce : sert de ROI pour la géométrie lignes."""

    estimate: VisionEstimate
    parking_mask_hw: np.ndarray


def segment_parking_on_chip(
    chip: OrthoChip,
    *,
    m2_per_space: float = 26.0,
    use_vision: bool = True,
    device: str | None = None,
) -> Optional[SegformerChipOutput]:
    """
    Segmentation binaire « parking » sur la puce.

    Le masque sert à **restreindre** l’analyse géométrique (Hough) à la zone lot ; on n’utilise plus
    ``aire / m²_par_place`` comme capacité principale lorsque le modèle n’est pas spécialisé places.
    """
    if not use_vision:
        return None

    proc, mdl = _lazy_load(device=device)
    dev = next(mdl.parameters()).device

    img = chip.image
    inputs = proc(images=img, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = mdl(**inputs)
        logits = out.logits
        logits = F.interpolate(
            logits,
            size=(chip.height_px, chip.width_px),
            mode="bilinear",
            align_corners=False,
        )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    parking_idx = _parking_class_index(mdl)
    mask = (pred == parking_idx).astype(np.bool_)
    frac = float(mask.mean()) if mask.size else 0.0

    m2px = chip_m2_per_pixel_from_chip(chip)
    area_m2 = frac * (chip.width_px * chip.height_px) * m2px
    est = int(max(0, area_m2 / m2_per_space)) if m2_per_space > 0 else 0

    if frac < 0.01:
        conf = "low"
        notes = "Très petite fraction classée parking."
    elif frac < 0.03:
        conf = "medium"
        notes = "Fraction modérée ; vérifier manuellement."
    else:
        conf = "medium"
        notes = "Grande zone parking détectée ; estimation indicative."

    est_obj = VisionEstimate(
        parking_pixel_fraction=frac,
        parking_area_m2=float(area_m2),
        estimated_spaces=est,
        confidence=conf,
        device=str(dev),
        notes=notes,
    )
    return SegformerChipOutput(estimate=est_obj, parking_mask_hw=mask)


def estimate_from_chip(
    chip: OrthoChip,
    *,
    m2_per_space: float = 26.0,
    use_vision: bool = True,
    device: str | None = None,
) -> VisionEstimate | None:
    """Compatibilité : retourne seulement l’estimation surface (pour baseline / métriques)."""
    out = segment_parking_on_chip(
        chip, m2_per_space=m2_per_space, use_vision=use_vision, device=device
    )
    return out.estimate if out else None


def chip_m2_per_pixel_from_chip(chip: OrthoChip) -> float:
    dx = (chip.maxx - chip.minx) / chip.width_px
    dy = (chip.maxy - chip.miny) / chip.height_px
    return float(dx * dy)
