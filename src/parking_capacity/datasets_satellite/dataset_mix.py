"""Fusion de jeux préparés (manifests + chemins d’images)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from parking_capacity.datasets_satellite.converters import UNIFIED_SUBDIR, write_metadata_jsonl
from parking_capacity.datasets_satellite.download_utils import project_data_datasets_dir


def project_root_default(project_root: Optional[Path] = None) -> Path:
    return project_root or Path(__file__).resolve().parents[3]


def mix_prepared_datasets(
    sources: Sequence[Union[str, Dict[str, Any]]],
    out_dir: Path,
    *,
    copy_files: bool = False,
    prefix_ids: bool = True,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Combine plusieurs répertoires ``prepared/*/parking_capacity_dataset``.

    Chaque entrée : nom court (``\"apklot\"``) ou ``{\"name\": \"apklot\", \"path\": Path}``.
    """
    root = project_root_default(project_root)
    base_data = project_data_datasets_dir(root)
    resolved: List[Tuple[str, Path]] = []
    for s in sources:
        if isinstance(s, dict):
            name = str(s["name"])
            p = Path(s.get("path") or base_data / "prepared" / name / "parking_capacity_dataset")
        else:
            name = str(s)
            p = base_data / "prepared" / name / "parking_capacity_dataset"
        if not p.is_dir():
            raise FileNotFoundError(f"Jeu préparé introuvable : {p}")
        resolved.append((name, p.resolve()))

    out_dir.mkdir(parents=True, exist_ok=True)
    mixed_img = out_dir / "images"
    mixed_lbl = out_dir / "labels"
    mixed_msk = out_dir / "masks"
    for d in (mixed_img, mixed_lbl, mixed_msk):
        d.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for ds_name, pc_root in resolved:
        meta_path = pc_root / "metadata.jsonl"
        if not meta_path.is_file():
            continue
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("image_id", "unk")
            new_id = f"{ds_name}_{uid}" if prefix_ids else str(uid)
            row["mix_source"] = ds_name
            row["mix_image_id"] = new_id
            src_im = pc_root / row.get("image", "")
            src_msk = pc_root / row.get("mask", "")
            src_lbl = pc_root / row.get("label", "")
            ext = Path(row.get("image", "x.jpg")).suffix or ".png"
            dst_im = mixed_img / f"{new_id}{ext}"
            if src_im.is_file():
                if copy_files:
                    shutil.copy2(src_im, dst_im)
                else:
                    try:
                        if dst_im.exists() or dst_im.is_symlink():
                            dst_im.unlink()
                        dst_im.symlink_to(src_im)
                    except OSError:
                        shutil.copy2(src_im, dst_im)
            if src_msk.is_file():
                dst_m = mixed_msk / f"{new_id}.png"
                if copy_files:
                    shutil.copy2(src_msk, dst_m)
                else:
                    try:
                        if dst_m.exists() or dst_m.is_symlink():
                            dst_m.unlink()
                        dst_m.symlink_to(src_msk)
                    except OSError:
                        shutil.copy2(src_msk, dst_m)
            if src_lbl.is_file():
                dst_l = mixed_lbl / f"{new_id}.txt"
                if copy_files:
                    shutil.copy2(src_lbl, dst_l)
                else:
                    try:
                        if dst_l.exists() or dst_l.is_symlink():
                            dst_l.unlink()
                        dst_l.symlink_to(src_lbl)
                    except OSError:
                        shutil.copy2(src_lbl, dst_l)
            row["image"] = f"images/{dst_im.name}"
            row["mask"] = f"masks/{new_id}.png"
            row["label"] = f"labels/{new_id}.txt"
            rows.append(row)

    write_metadata_jsonl(rows, out_dir / "metadata.jsonl")
    manifest = {
        "sources": [{"name": n, "path": str(p)} for n, p in resolved],
        "n_rows": len(rows),
        "layout": UNIFIED_SUBDIR,
    }
    (out_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
