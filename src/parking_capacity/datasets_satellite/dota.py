"""DOTA — objets orientés (aerial / satellite)."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from parking_capacity.datasets_satellite.converters import (
    dir_size_bytes,
    export_yolo_obb_dir,
    write_metadata_jsonl,
)
from parking_capacity.datasets_satellite.download_utils import (
    download_url_streaming,
    extract_zip,
    project_data_datasets_dir,
)
from parking_capacity.datasets_satellite.registry import update_dataset_status

# URLs officielles changent ; variables d’environnement prioritaires (voir docs/datasets.md).
DOTA_TRAIN_ZIP_URL_ENV = "DOTA_TRAIN_ZIP_URL"
DOTA_VAL_ZIP_URL_ENV = "DOTA_VAL_ZIP_URL"

DOTA_CLASS_NAMES_V1 = [
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
]

NAME_TO_ID = {n: i for i, n in enumerate(DOTA_CLASS_NAMES_V1)}


def _raw(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "raw" / "dota"


def _prepared(project_root: Optional[Path] = None) -> Path:
    return project_data_datasets_dir(project_root) / "prepared" / "dota"


def download_dota(
    dest: Optional[Path] = None,
    *,
    train_url: Optional[str] = None,
    val_url: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Télécharge DOTA si des URLs sont fournies (env ou arguments).

    Sans URL : retourne instructions (Baidu / Google Drive — pas de lien direct stable).
    """
    import os

    dest = dest or _raw(project_root)
    dest.mkdir(parents=True, exist_ok=True)
    train_url = train_url or os.environ.get(DOTA_TRAIN_ZIP_URL_ENV)
    val_url = val_url or os.environ.get(DOTA_VAL_ZIP_URL_ENV)
    results: List[str] = []
    if train_url:
        tpath = dest / "DOTA_train.zip"
        download_url_streaming(train_url, tpath)
        results.append(str(tpath))
    if val_url:
        vpath = dest / "DOTA_val.zip"
        download_url_streaming(val_url, vpath)
        results.append(str(vpath))

    if not results:
        update_dataset_status(
            "dota",
            status="missing",
            extra={
                "manual_required": True,
                "page": "https://captain-whu.github.io/DOTA/dataset.html",
            },
            project_root=project_root,
        )
        return {
            "ok": False,
            "path": str(dest),
            "downloaded": [],
            "instructions": (
                "Téléchargez les archives DOTA-v1.0 (train / val) depuis la page officielle "
                "(Google Drive / Baidu). Placez les .zip sous "
                f"{dest} puis relancez avec DOTA_TRAIN_ZIP_URL / DOTA_VAL_ZIP_URL "
                "ou extrayez manuellement."
            ),
        }

    sz = dir_size_bytes(dest)
    update_dataset_status("dota", status="downloaded", size_bytes=sz, project_root=project_root)
    return {"ok": True, "path": str(dest), "downloaded": results, "size_bytes": sz}


def _extract_zips(raw: Path) -> None:
    for z in raw.glob("*.zip"):
        try:
            extract_zip(z, raw / z.stem)
        except zipfile.BadZipFile:
            continue


def parse_dota_label_txt(path: Path) -> List[Tuple[List[Tuple[float, float]], str, int]]:
    """Parse une annotation DOTA : quadrilatère + catégorie + difficult."""
    rows: List[Tuple[List[Tuple[float, float]], str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        xs = [float(parts[i]) for i in range(8)]
        quad = [(xs[i], xs[i + 1]) for i in range(0, 8, 2)]
        cat = parts[8]
        diff = int(parts[9])
        rows.append((quad, cat, diff))
    return rows


def discover_dota_roots(raw: Path) -> List[Tuple[Path, Path]]:
    """Paires (images_dir, labelTxt_dir)."""
    pairs: List[Tuple[Path, Path]] = []
    for lt in raw.glob("**/labelTxt"):
        if not lt.is_dir():
            continue
        parent = lt.parent
        for img_name in ("images", "JPEGImages"):
            im = parent / img_name
            if im.is_dir():
                pairs.append((im, lt))
                break
    return pairs


def prepare_dota_dataset(
    raw_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    categories_subset: Optional[Sequence[str]] = None,
    train_frac: float = 0.85,
    val_frac: float = 0.10,
    seed: int = 42,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Parse les ``labelTxt``, exporte YOLO-OBB et métadonnées.

    ``categories_subset`` : ex. ``(\"small-vehicle\", \"large-vehicle\")`` pour véhicules seuls.
    """
    raw_root = raw_root or _raw(project_root)
    out_dir = out_dir or _prepared(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    _extract_zips(raw_root)

    subset = set(categories_subset) if categories_subset else None
    obb_records: List[Tuple[str, int, Sequence[Tuple[float, float]], int, int]] = []
    meta_rows: List[Dict[str, object]] = []

    pairs = discover_dota_roots(raw_root)
    if not pairs:
        update_dataset_status(
            "dota",
            status="missing",
            extra={"last_prepare_error": "Aucun labelTxt/images trouvé sous raw/dota."},
            project_root=project_root,
        )
        return {
            "ok": False,
            "error": "Répertoire DOTA non trouvé. Consultez docs/datasets.md.",
            "prepared": str(out_dir),
        }

    all_ids: List[str] = []
    for img_dir, lt_dir in pairs:
        for txt in lt_dir.glob("*.txt"):
            stem = txt.stem
            img_path = None
            for ext in (".png", ".jpg", ".bmp"):
                cand = img_dir / f"{stem}{ext}"
                if cand.is_file():
                    img_path = cand
                    break
            if img_path is None:
                continue
            from PIL import Image

            im = Image.open(img_path)
            w, h = im.size
            for quad, cat, diff in parse_dota_label_txt(txt):
                if subset is not None and cat not in subset:
                    continue
                cid = NAME_TO_ID.get(cat, -1)
                if cid < 0:
                    continue
                obb_records.append((stem, cid, quad, w, h))
            all_ids.append(stem)
            meta_rows.append(
                {
                    "image_id": stem,
                    "width": w,
                    "height": h,
                    "image_path": str(img_path.resolve()),
                    "label_path": str(txt.resolve()),
                }
            )

    # split par image
    ids_unique = sorted(set(all_ids))
    rng = random.Random(seed)
    rng.shuffle(ids_unique)
    n = len(ids_unique)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    split_map: Dict[str, str] = {}
    for i, sid in enumerate(ids_unique):
        if i < n_tr:
            split_map[sid] = "train"
        elif i < n_tr + n_va:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"

    for row in meta_rows:
        row["split"] = split_map.get(row["image_id"], "train")

    write_metadata_jsonl(meta_rows, out_dir / "metadata.jsonl")

    # YOLO OBB par split (fichiers regroupés par image_id)
    for sp in ("train", "val", "test"):
        ids_sp = {row["image_id"] for row in meta_rows if row["split"] == sp}
        rec_sp = [r for r in obb_records if r[0] in ids_sp]
        if rec_sp:
            export_yolo_obb_dir(rec_sp, out_dir / "yolo_obb" / sp)

    manifest = {
        "dataset": "dota",
        "n_label_files": len(meta_rows),
        "n_obb_instances": len(obb_records),
        "classes": DOTA_CLASS_NAMES_V1,
        "subset": list(subset) if subset else None,
        "raw_root": str(raw_root.resolve()),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sz = dir_size_bytes(out_dir)
    update_dataset_status("dota", status="prepared", size_bytes=sz, project_root=project_root)
    return {"ok": True, "manifest": manifest, "prepared": str(out_dir)}
