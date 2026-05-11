"""Export jeu vision (COCO minimal) depuis un dossier benchmark."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


def export_vision_dataset(benchmark_results_dir: Path, out_dir: Path) -> Path:
    """
    Lit ``benchmark_results_dir`` (sortie ``benchmark-addresses``) : chaque sous-dossier avec ``chip.png``.

    Produit ``images/``, ``overlays/``, ``metadata.jsonl``, ``coco_minimal.json``.
    """
    benchmark_results_dir = Path(benchmark_results_dir)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    meta_lines: List[str] = []
    img_id = 1
    ann_id = 1

    for sub in sorted(benchmark_results_dir.iterdir()):
        if not sub.is_dir():
            continue
        chip = sub / "chip.png"
        if not chip.is_file():
            continue
        slug = sub.name
        dest_img = out_dir / "images" / f"{slug}.png"
        shutil.copyfile(chip, dest_img)
        ov_src = sub / "debug_overlay.png"
        if ov_src.is_file():
            shutil.copyfile(ov_src, out_dir / "overlays" / f"{slug}.png")

        meta_path = sub / "result.json"
        row: Dict[str, Any] = {"id": slug, "folder": str(sub.name)}
        label_source = "human"
        if meta_path.is_file():
            try:
                row.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        ls = row.get("label_source")
        if isinstance(ls, str) and ls.strip():
            label_source = ls.strip()
        else:
            prov = str(row.get("capacity_provenance", ""))
            if "osm" in prov.lower() and row.get("estimated_capacity") is None:
                label_source = "osm_pseudo"
            elif row.get("ml_estimated_capacity"):
                label_source = "model"
            elif row.get("geometric_capacity_estimate") and (
                row.get("capacity_osm_parcelle") or row.get("capacity_osm_buffer")
            ):
                label_source = "hybrid"
        row["label_source"] = label_source
        meta_lines.append(json.dumps(row, ensure_ascii=False, default=str))

        w, h = Image.open(dest_img).size
        images.append({"id": img_id, "file_name": f"images/{slug}.png", "width": w, "height": h})
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [0, 0, w, h],
                "segmentation": [],
                "iscrowd": 0,
                "label_source": label_source,
            }
        )
        img_id += 1
        ann_id += 1

    (out_dir / "metadata.jsonl").write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""), encoding="utf-8")
    coco = {
        "info": {"description": "parking-capacity vision export (minimal)", "version": "1.0"},
        "categories": [{"id": 1, "name": "parking_surface", "supercategory": "parking"}],
        "images": images,
        "annotations": annotations,
    }
    (out_dir / "coco_minimal.json").write_text(json.dumps(coco, indent=2, default=str), encoding="utf-8")
    return out_dir / "coco_minimal.json"
