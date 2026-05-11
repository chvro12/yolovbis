"""xView — détection d’objets à très haute résolution (souvent accès restreint)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from parking_capacity.datasets_satellite.converters import dir_size_bytes, write_metadata_jsonl
from parking_capacity.datasets_satellite.download_utils import project_data_datasets_dir
from parking_capacity.datasets_satellite.registry import update_dataset_status

XVIEW_SYNC_CMD_ENV = "XVIEW_SYNC_CMD"
XVIEW_MANUAL_ROOT_ENV = "XVIEW_MANUAL_ROOT"


def _default_raw(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "raw" / "xview"


def _default_prepared(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "prepared" / "xview"


def download_xview(
    dest: Optional[Path] = None,
    *,
    sync_cmd: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Téléchargement automatisé si ``XVIEW_SYNC_CMD`` est défini (ex. aws s3 sync …).

    Sinon : instructions pour inscription https://xviewdataset.org/ et dépôt manuel.
    """
    dest = dest or _default_raw(project_root)
    dest.mkdir(parents=True, exist_ok=True)
    manual = os.environ.get(XVIEW_MANUAL_ROOT_ENV)
    if manual and Path(manual).is_dir():
        return {
            "ok": True,
            "path": manual,
            "notes": "XVIEW_MANUAL_ROOT pointe vers des données déjà présentes.",
        }

    cmd = sync_cmd or os.environ.get(XVIEW_SYNC_CMD_ENV)
    if cmd:
        p = subprocess.run(cmd, shell=True, cwd=str(dest), capture_output=True, text=True)
        if p.returncode != 0:
            return {"ok": False, "error": p.stderr or p.stdout, "path": str(dest)}
        sz = dir_size_bytes(dest)
        update_dataset_status("xview", status="downloaded", size_bytes=sz, project_root=project_root)
        return {"ok": True, "path": str(dest), "size_bytes": sz, "via": "XVIEW_SYNC_CMD"}

    instructions = (
        "1) Créer un compte sur https://xviewdataset.org/ et accepter les conditions.\n"
        "2) Télécharger les GeoTIFF / GeoJSON selon la documentation officielle.\n"
        "3) Placer les fichiers sous "
        f"{dest}\n"
        "   ou définir XVIEW_MANUAL_ROOT=/chemin/vers/données_xview\n"
        "4) Optionnel : exporter XVIEW_SYNC_CMD='aws s3 sync s3://… "
        f"{dest}' si vous disposez d’un miroir autorisé."
    )
    (dest / "DOWNLOAD_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
    update_dataset_status(
        "xview",
        status="missing",
        extra={"manual_required": True},
        project_root=project_root,
    )
    return {"ok": False, "path": str(dest), "instructions": instructions}


def prepare_xview_dataset(
    raw_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Indexe les fichiers GeoJSON / TIFF / PNG présents et écrit ``metadata.jsonl``.
    """
    env_root = os.environ.get(XVIEW_MANUAL_ROOT_ENV)
    if raw_root is None and env_root:
        raw_root = Path(env_root)
    raw_root = raw_root or _default_raw(project_root)
    raw_root = Path(raw_root)
    out_dir = out_dir or _default_prepared(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.is_dir():
        return {"ok": False, "error": f"Répertoire introuvable : {raw_root}"}

    rows: List[Dict[str, object]] = []
    for pat in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.geojson", "*.json"):
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
    manifest = {"dataset": "xview", "n_files": len(rows), "raw_root": str(raw_root.resolve())}
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not rows:
        update_dataset_status(
            "xview",
            status="missing",
            extra={"last_prepare_error": "Aucun fichier indexé."},
            project_root=project_root,
        )
        return {"ok": False, "manifest": manifest, "prepared": str(out_dir)}

    sz = dir_size_bytes(out_dir)
    update_dataset_status("xview", status="prepared", size_bytes=sz, project_root=project_root)
    return {"ok": True, "manifest": manifest, "prepared": str(out_dir)}
