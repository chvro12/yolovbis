"""Mise en page YOLOv8 segmentation à partir de ``parking_capacity_dataset``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_yolo_seg_layout(
    unified_root: Path,
    out_root: Path,
    *,
    copy_files: bool = True,
) -> Dict[str, Any]:
    """
    Crée ``images/{train,val,test}`` et ``labels/{train,val,test}`` + ``dataset.yaml``.

    Lit ``metadata.jsonl`` (champ ``split``). Les images sont copiées avec extension conservée.
    """
    meta_path = unified_root / "metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.jsonl introuvable : {unified_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row: Dict[str, Any] = json.loads(line)
        split = str(row.get("split") or "train")
        if split not in counts:
            split = "train"
        uid = row.get("image_id") or Path(row.get("image", "")).stem
        img_rel = row.get("image")
        lbl_rel = row.get("label")
        if not img_rel or not lbl_rel:
            continue
        src_img = unified_root / img_rel
        src_lbl = unified_root / lbl_rel
        if not src_img.is_file() or not src_lbl.is_file():
            continue
        ext = src_img.suffix.lower() or ".jpg"
        dst_img = out_root / "images" / split / f"{uid}{ext}"
        dst_lbl = out_root / "labels" / split / f"{uid}.txt"
        if copy_files:
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        else:
            try:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(src_img.resolve())
                if dst_lbl.exists():
                    dst_lbl.unlink()
                dst_lbl.symlink_to(src_lbl.resolve())
            except OSError:
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
        counts[split] = counts[split] + 1

    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names: ['parking']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Ultralytics exige au moins train + val non vides ; sinon dupliquer un échantillon train -> val
    if counts.get("val", 0) == 0 and counts.get("train", 0) > 1:
        tr_im = list((out_root / "images" / "train").glob("*"))
        if tr_im:
            sample = tr_im[0]
            stem = sample.stem
            for ext_lab in (".txt",):
                src_l = out_root / "labels" / "train" / f"{stem}{ext_lab}"
                if src_l.is_file():
                    shutil.copy2(sample, out_root / "images" / "val" / sample.name)
                    shutil.copy2(src_l, out_root / "labels" / "val" / f"{stem}.txt")
                    counts["val"] = counts.get("val", 0) + 1
                    break

    return {"dataset_yaml": str(yaml_path.resolve()), "counts": counts, "out_root": str(out_root.resolve())}


def ensure_yolo_seg_dataset(unified_root: Path, out_root: Optional[Path] = None) -> Path:
    """Construit ``yolo_seg_dataset`` à côté du jeu unifié si nécessaire."""
    out = out_root or (unified_root.parent / "yolo_seg_dataset")
    if (out / "images" / "train").is_dir() and (out / "labels" / "train").is_dir():
        return out
    build_yolo_seg_layout(unified_root, out, copy_files=True)
    return out
