"""SpaceNet — bâtiments, routes, segmentation satellite (AWS Open Data)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from parking_capacity.datasets_satellite.converters import dir_size_bytes, write_metadata_jsonl
from parking_capacity.datasets_satellite.download_utils import command_available, project_data_datasets_dir
from parking_capacity.datasets_satellite.registry import update_dataset_status

SPACENET_SYNC_CMD_ENV = "SPACENET_SYNC_CMD"
# Exemple documenté : aws s3 cp s3://spacenet-dataset/spacenet/SN5_roads/ …

SPACENET_MANUAL_ROOT_ENV = "SPACENET_MANUAL_ROOT"


def _default_raw(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "raw" / "spacenet"


def _default_prepared(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "prepared" / "spacenet"


def download_spacenet(
    dest: Optional[Path] = None,
    *,
    sync_cmd: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Télécharge via ``aws s3`` si ``SPACENET_SYNC_CMD`` ou aws disponible.

    Exemple de commande (à adapter au challenge) ::
        aws s3 sync s3://spacenet-dataset/spacenet/SN2_BUILDINGS/ …/raw/spacenet/sn2
    """
    dest = dest or _default_raw(project_root)
    dest.mkdir(parents=True, exist_ok=True)
    manual = os.environ.get(SPACENET_MANUAL_ROOT_ENV)
    if manual and Path(manual).is_dir():
        return {"ok": True, "path": manual, "notes": "SPACENET_MANUAL_ROOT défini."}

    cmd = sync_cmd or os.environ.get(SPACENET_SYNC_CMD_ENV)
    if cmd:
        p = subprocess.run(cmd, shell=True, cwd=str(dest.parent), capture_output=True, text=True)
        if p.returncode != 0:
            return {"ok": False, "error": (p.stderr or p.stdout)[:4000], "path": str(dest)}
        sz = dir_size_bytes(dest)
        update_dataset_status("spacenet", status="downloaded", size_bytes=sz, project_root=project_root)
        return {"ok": True, "path": str(dest), "size_bytes": sz}

    notes = []
    if command_available("aws"):
        notes.append(
            "AWS CLI détecté : définissez SPACENET_SYNC_CMD avec un URI "
            "s3://spacenet-dataset/... (voir registry Open Data SpaceNet)."
        )
    else:
        notes.append("Installez AWS CLI v2 et configurez les credentials si besoin.")

    doc = (
        "SpaceNet est sur AWS Registry Open Data — buckets publics read-only.\n"
        "Lister : aws s3 ls s3://spacenet-dataset/\n"
        "Puis : aws s3 sync s3://spacenet-dataset/<challenge>/ " + str(dest) + "\n"
    )
    (dest / "DOWNLOAD_INSTRUCTIONS.txt").write_text(doc, encoding="utf-8")
    update_dataset_status(
        "spacenet",
        status="missing",
        extra={"manual_required": True},
        project_root=project_root,
    )
    return {"ok": False, "path": str(dest), "instructions": doc, "notes": notes}


def prepare_spacenet_dataset(
    raw_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Indexe GeoJSON (bâtiments / routes) et TIFF présents."""
    if raw_root is None and os.environ.get(SPACENET_MANUAL_ROOT_ENV):
        raw_root = Path(os.environ[SPACENET_MANUAL_ROOT_ENV])
    raw_root = raw_root or _default_raw(project_root)
    raw_root = Path(raw_root)
    out_dir = out_dir or _default_prepared(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.is_dir():
        return {"ok": False, "error": f"Répertoire introuvable : {raw_root}"}

    rows: List[Dict[str, object]] = []
    for pat in ("*.geojson", "*.json", "*.tif", "*.tiff", "*.png"):
        for p in raw_root.rglob(pat):
            if p.name.startswith("."):
                continue
            rows.append(
                {
                    "path": str(p.resolve()),
                    "suffix": p.suffix.lower(),
                    "size_bytes": p.stat().st_size if p.is_file() else 0,
                }
            )

    write_metadata_jsonl(rows, out_dir / "metadata.jsonl")
    manifest = {"dataset": "spacenet", "n_files": len(rows), "raw_root": str(raw_root.resolve())}
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not rows:
        update_dataset_status(
            "spacenet",
            status="missing",
            extra={"last_prepare_error": "Aucun fichier indexé."},
            project_root=project_root,
        )
        return {"ok": False, "manifest": manifest, "prepared": str(out_dir)}

    sz = dir_size_bytes(out_dir)
    update_dataset_status("spacenet", status="prepared", size_bytes=sz, project_root=project_root)
    return {"ok": True, "manifest": manifest, "prepared": str(out_dir)}
