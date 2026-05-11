"""Registre local ``dataset_registry.json`` (chemins, statut, licences)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from parking_capacity.datasets_satellite.download_utils import project_data_datasets_dir


def registry_path(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "dataset_registry.json"


def default_registry(project_root: Optional[Path] = None) -> Dict[str, Any]:
    base = str(project_data_datasets_dir(project_root))
    return {
        "version": 2,
        "updated_at": None,
        "datasets": {
            "apklot": {
                "name": "APKLOT",
                "dataset_type": "mixed",
                "source_url": "https://github.com/langheran/APKLOT",
                "license_notes": "MIT (code) ; images Google Maps — respecter https://www.google.com/permissions/geoguidelines/ (fair use, pas de revente).",
                "classes": ["parking_block"],
                "raw_path": f"{base}/raw/apklot",
                "prepared_path": f"{base}/prepared/apklot",
                "status": "missing",
                "size_bytes_estimate": None,
                "notes": (
                    "Jeux parking vue plongeante ET caméra : dossiers « 1. Satellite » (Google Maps) et "
                    "« 2. Camera » (LabelMe). Git LFS requis pour la partie satellite ; sans LFS, souvent "
                    "seule la caméra est présente — utiliser prepare --apklot-view satellite|all selon le besoin."
                ),
            },
            "dota": {
                "name": "DOTA",
                "dataset_type": "satellite",
                "source_url": "https://captain-whu.github.io/DOTA/",
                "license_notes": "Recherche académique uniquement ; pas d’usage commercial. Images Google Earth / GF-2 / JL-1 / CycloMedia — conditions fournisseurs.",
                "classes": [
                    "plane",
                    "ship",
                    "storage-tank",
                    "baseball-diamond",
                    "tennis-court",
                    "basketball-court",
                    "ground-track-field",
                    "harbor",
                    "bridge",
                    "large-vehicle",
                    "small-vehicle",
                    "helicopter",
                    "roundabout",
                    "soccer-ball-field",
                    "swimming-pool",
                ],
                "raw_path": f"{base}/raw/dota",
                "prepared_path": f"{base}/prepared/dota",
                "status": "missing",
                "size_bytes_estimate": None,
                "notes": "ZIPs souvent sur Baidu Pan / Google Drive ; placer les archives sous raw/dota/ puis datasets-prepare.",
            },
            "xview": {
                "name": "xView",
                "dataset_type": "satellite",
                "source_url": "https://xviewdataset.org/",
                "license_notes": "xView Challenge Dataset Agreement (Defense Digital Service) — usage selon conditions d’inscription ; pas de redistribution.",
                "classes": ["multi_class_satellite_objects"],
                "raw_path": f"{base}/raw/xview",
                "prepared_path": f"{base}/prepared/xview",
                "status": "missing",
                "size_bytes_estimate": None,
                "notes": "Inscription sur xviewdataset.org ; miroirs possibles (Kaggle, AWS). Téléchargement manuel fréquent.",
            },
            "spacenet": {
                "name": "SpaceNet",
                "dataset_type": "satellite",
                "source_url": "https://spacenet.ai/",
                "license_notes": "SpaceNet dataset license (voir pages AWS Open Data par challenge) ; attribution requise.",
                "classes": ["building", "road", "off_nadir_objects_challenge_dependent"],
                "raw_path": f"{base}/raw/spacenet",
                "prepared_path": f"{base}/prepared/spacenet",
                "status": "missing",
                "size_bytes_estimate": None,
                "notes": "Téléchargement typique : aws s3 sync depuis buckets spacenet-dataset (voir docs).",
            },
        },
    }


def migrate_registry(data: Dict[str, Any]) -> bool:
    """Ajoute ``dataset_type`` et passe ``version`` à 2 si nécessaire."""
    changed = False
    ver = data.get("version", 1)
    defaults_type = {
        "apklot": "mixed",
        "dota": "satellite",
        "xview": "satellite",
        "spacenet": "satellite",
    }
    if ver < 2:
        for name, ds in list(data.get("datasets", {}).items()):
            if "dataset_type" not in ds:
                ds["dataset_type"] = defaults_type.get(name, "mixed")
                changed = True
        data["version"] = 2
        changed = True
    return changed


def load_registry(project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = registry_path(project_root)
    if not path.is_file():
        reg = default_registry(project_root)
        save_registry(reg, project_root)
        return reg
    data = json.loads(path.read_text(encoding="utf-8"))
    if migrate_registry(data):
        save_registry(data, project_root)
    return data


def save_registry(data: Dict[str, Any], project_root: Optional[Path] = None) -> None:
    path = registry_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_dataset_status(
    name: str,
    *,
    status: str,
    size_bytes: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    reg = load_registry(project_root)
    if name not in reg.get("datasets", {}):
        reg.setdefault("datasets", {})[name] = default_registry(project_root)["datasets"].get(
            name,
            {"name": name, "status": status},
        )
    ds = reg["datasets"][name]
    ds["status"] = status
    if size_bytes is not None:
        ds["size_bytes_estimate"] = size_bytes
    if extra:
        ds.update(extra)
    save_registry(reg, project_root)
    return reg
