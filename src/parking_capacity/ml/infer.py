"""Inférence régression capacité depuis une image (puce orthophoto)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

from parking_capacity.ml.models import ARCHITECTURES, build_model

DEFAULT_HALF_SIDE_M = 55.0
DEFAULT_CHIP_PIXELS = 512


def read_inference_config(checkpoint: Path) -> Dict[str, Any]:
    """
    Lit la géométrie WMS et l'architecture sans charger les poids (via model_meta.json si présent).
    Sinon charge le fichier .pt (plus lourd).
    """
    checkpoint = Path(checkpoint)
    sidecar = checkpoint.with_name("model_meta.json")
    if sidecar.exists():
        d = json.loads(sidecar.read_text(encoding="utf-8"))
        d.setdefault("target_transform", "none")
        return d
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint invalide : dict attendu")
    return {
        "architecture": str(payload.get("architecture", "tiny")),
        "img_size": int(payload.get("img_size", 128)),
        "half_side_m": float(payload.get("half_side_m", DEFAULT_HALF_SIDE_M)),
        "chip_pixels": int(payload.get("chip_pixels", DEFAULT_CHIP_PIXELS)),
        "target_transform": str(payload.get("target_transform", "none")),
    }


def _pil_to_model_input(image: Image.Image, img_size: int) -> torch.Tensor:
    img = image.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def load_regressor(
    checkpoint: Path,
    *,
    device_str: str | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], torch.device]:
    checkpoint = Path(checkpoint)
    cfg = read_inference_config(checkpoint)
    arch = str(cfg.get("architecture", "tiny"))
    if arch not in ARCHITECTURES:
        arch = "tiny"
    img_size = int(cfg.get("img_size", 128))

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Checkpoint invalide : clé state_dict manquante")
    model = build_model(arch, pretrained=False)  # type: ignore[arg-type]
    model.load_state_dict(payload["state_dict"])
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train(False)
    meta = {**cfg, "architecture": arch, "img_size": img_size}
    return model, meta, device


def predict_capacity_from_pil(
    image: Image.Image,
    checkpoint: Path,
    *,
    device_str: str | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Retourne (prédiction float, infos : device, img_size, architecture).
    """
    model, meta, device = load_regressor(checkpoint, device_str=device_str)
    img_size = int(meta["img_size"])
    x = _pil_to_model_input(image, img_size).to(device)
    with torch.no_grad():
        y = model(x).squeeze(-1)
    raw_out = float(y.item())
    tt = str(meta.get("target_transform", "none"))
    if tt == "log1p":
        pred = max(0.0, math.expm1(raw_out))
    else:
        pred = raw_out
    info = {
        "architecture": meta["architecture"],
        "img_size": img_size,
        "device": str(device),
        "target_transform": tt,
        "raw_model_output": raw_out,
    }
    return pred, info
