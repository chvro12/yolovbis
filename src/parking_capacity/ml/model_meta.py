"""Métadonnées d’entraînement / inférence et garde-fous ML."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def load_model_meta(checkpoint: Path) -> Optional[Dict[str, Any]]:
    """Lit ``model_meta.json`` à côté du checkpoint, si présent."""
    checkpoint = Path(checkpoint)
    sidecar = checkpoint.with_name("model_meta.json")
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def model_meta_blocks_primary_ml(meta: Dict[str, Any]) -> bool:
    """
    True si le modèle ne doit pas être utilisé comme source principale
    (sauf ``--force-ml``).
    """
    mode = str(meta.get("dataset_mode", "unknown")).lower()
    if mode in ("synthetic", "mock"):
        return True
    n_train = meta.get("n_train_samples", meta.get("n_samples"))
    try:
        n_int = int(n_train) if n_train is not None else 0
    except (TypeError, ValueError):
        n_int = 0
    if n_int < 100:
        return True
    r2 = meta.get("val_r2")
    try:
        r2f = float(r2)
    except (TypeError, ValueError):
        r2f = float("nan")
    if not math.isnan(r2f) and r2f < 0:
        return True
    return False


def should_skip_ml_inference(
    checkpoint: Optional[Path],
    *,
    force_ml: bool,
) -> tuple[bool, Optional[Dict[str, Any]], str]:
    if checkpoint is None:
        return False, None, ""
    ck = Path(checkpoint)
    meta = load_model_meta(ck)
    if force_ml:
        return False, meta, ""
    if meta is None:
        return False, None, ""
    if model_meta_blocks_primary_ml(meta):
        return (
            True,
            meta,
            (
                "Inférence ML ignorée : `model_meta.json` indique un jeu synthétique/mock, "
                "moins de 100 exemples d’entraînement, ou un R² de validation négatif. "
                "Utilisez `--force-ml` pour forcer l’inférence."
            ),
        )
    return False, meta, ""
