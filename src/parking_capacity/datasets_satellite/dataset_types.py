"""Types de jeu (`dataset_type`) et règles d’entraînement segmentation satellite / orthophoto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from parking_capacity.datasets_satellite.download_utils import project_data_datasets_dir
from parking_capacity.datasets_satellite.registry import load_registry

# Valeurs documentées (registre + inspect)
DATASET_TYPES = ("camera", "drone", "aerial", "satellite", "mixed")

# Orthophoto / segmentation satellite : autorisé sauf mode purement « caméra ground ».
SATELLITE_SEGMENTATION_ALLOWED_TYPES: FrozenSet[str] = frozenset(
    {"aerial", "satellite", "mixed", "drone"}
)


DEFAULT_REGISTRY_DATASET_TYPES: Dict[str, str] = {
    "apklot": "mixed",
    "dota": "satellite",
    "xview": "satellite",
    "spacenet": "satellite",
}


def get_dataset_type(entry: Optional[Dict[str, Any]]) -> str:
    """Retourne ``dataset_type`` avec défaut conservateur."""
    if not entry:
        return "mixed"
    dt = entry.get("dataset_type")
    if isinstance(dt, str) and dt in DATASET_TYPES:
        return dt
    return "mixed"


def registry_allows_satellite_segmentation_training(entry: Optional[Dict[str, Any]]) -> bool:
    """True si le registre autorise l’entraînement « vue plongeante / satellite »."""
    return get_dataset_type(entry) in SATELLITE_SEGMENTATION_ALLOWED_TYPES


def apklot_prepare_allows_satellite_training(meta: Optional[Dict[str, Any]]) -> bool:
    """
    Pour APKLOT : le registre peut être ``mixed`` mais le préparé peut être 100 % caméra.
    Utilise ``dataset_prepare_meta.json`` écrit par ``prepare_apklot_dataset``.
    """
    if not meta:
        return True
    if meta.get("dataset") != "apklot":
        return True
    return bool(meta.get("satellite_segmentation_suitable", True))


def training_gate_message(dataset: str, entry: Dict[str, Any], prepare_meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """Message d’erreur si entraînement refusé ; ``None`` si OK."""
    if not registry_allows_satellite_segmentation_training(entry):
        return (
            f"Le jeu « {dataset} » est de type « {get_dataset_type(entry)} », incompatible avec "
            "l’entraînement segmentation satellite / orthophoto par défaut. "
            "Utilisez un jeu aerial/satellite/mixed ou ``--force-incompatible-dataset``."
        )
    if dataset == "apklot" and not apklot_prepare_allows_satellite_training(prepare_meta):
        return (
            "APKLOT préparé ne contient aucune image sous un chemin « Satellite » "
            "(README upstream : « 1. Satellite », Google Maps API). "
            "Exécutez ``git lfs pull`` dans raw/apklot, ou ``datasets-prepare --apklot-view all``, "
            "ou choisissez DOTA/xView/SpaceNet pour l’aerial ; ``--force-incompatible-dataset`` pour ignorer."
        )
    return None


def resolve_prepared_dir(dataset: str, project_root: Optional[Path] = None) -> Path:
    reg = load_registry(project_root)
    info = reg.get("datasets", {}).get(dataset)
    if not info:
        raise ValueError(f"Dataset inconnu : {dataset}")
    p = Path(info["prepared_path"])
    if p.is_absolute():
        return p.resolve()
    root = project_data_datasets_dir(project_root).parent.parent
    return (root / p).resolve()


def load_prepare_meta(dataset: str, project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    prep = resolve_prepared_dir(dataset, project_root)
    meta = prep / "dataset_prepare_meta.json"
    if meta.is_file():
        return json.loads(meta.read_text(encoding="utf-8"))
    return None


def assert_satellite_segmentation_training_allowed(
    dataset: str,
    *,
    force: bool = False,
    project_root: Optional[Path] = None,
) -> None:
    if force:
        return
    reg = load_registry(project_root)
    entry = reg.get("datasets", {}).get(dataset, {})
    meta = load_prepare_meta(dataset, project_root) if dataset == "apklot" else None
    msg = training_gate_message(dataset, entry, meta)
    if msg:
        raise ValueError(msg)
