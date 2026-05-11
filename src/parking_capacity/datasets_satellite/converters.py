"""Conversions COCO, YOLO (seg / OBB), masques PNG, GeoJSON, format unifié."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from shapely.geometry import mapping, Polygon
except ImportError:  # pragma: no cover
    Polygon = None  # type: ignore
    mapping = None  # type: ignore


UNIFIED_SUBDIR = "parking_capacity_dataset"


@dataclass
class ImageRecord:
    """Une image et ses annotations pour export unifié."""

    image_id: str
    rel_image: str
    split: str
    width: int
    height: int
    polygons: List[List[Tuple[float, float]]]  # chaque polygone = liste (x,y) pixels
    category_ids: List[int]
    source_dataset: str


def ensure_unified_layout(root: Path) -> Tuple[Path, Path, Path, Path]:
    """Crée ``parking_capacity_dataset/{images,labels,masks}`` et retourne les chemins."""
    base = root / UNIFIED_SUBDIR
    img_d = base / "images"
    lbl_d = base / "labels"
    msk_d = base / "masks"
    for p in (img_d, lbl_d, msk_d):
        p.mkdir(parents=True, exist_ok=True)
    return base, img_d, lbl_d, msk_d


def write_metadata_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def polygon_to_binary_mask(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    width: int,
    height: int,
) -> np.ndarray:
    """Masque uint8 0/255 à partir de polygones (xy pixels)."""
    img = Image.new("L", (width, height), 0)
    dr = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) >= 3:
            dr.polygon([tuple(p) for p in poly], outline=255, fill=255)
    return np.array(img, dtype=np.uint8)


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def yolo_segmentation_line(
    class_id: int,
    polygon_xy: Sequence[Tuple[float, float]],
    width: int,
    height: int,
) -> str:
    """Une ligne YOLO segmentation : class x1 y1 x2 y2 ... normalisées 0–1."""
    parts = [str(class_id)]
    for x, y in polygon_xy:
        parts.append(f"{float(x) / width:.6f}")
        parts.append(f"{float(y) / height:.6f}")
    return " ".join(parts)


def yolo_obb_line(
    class_id: int,
    quad_xy: Sequence[Tuple[float, float]],
    width: int,
    height: int,
) -> str:
    """
    YOLO OBB (8 coords normalisées), ordre identique à l’entrée DOTA (horaire).
    Format : class x1 y1 x2 y2 x3 y3 x4 y4
    """
    if len(quad_xy) != 4:
        raise ValueError("OBB attend 4 sommets")
    parts = [str(class_id)]
    for x, y in quad_xy:
        parts.append(f"{float(x) / width:.6f}")
        parts.append(f"{float(y) / height:.6f}")
    return " ".join(parts)


def coco_template(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "info": {"description": "parking-capacity satellite export"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def add_coco_segmentation_instance(
    coco: Dict[str, Any],
    *,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    ann_id: int,
    category_id: int,
    segmentation: List[List[float]],
    bbox_xywh: Tuple[float, float, float, float],
    area: float,
) -> None:
    """Ajoute une image (si besoin) et une annotation segmentation RLE/polygone COCO."""
    imgs = coco["images"]
    if not any(im["id"] == image_id for im in imgs):
        imgs.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
    coco["annotations"].append(
        {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": list(bbox_xywh),
            "iscrowd": 0,
        }
    )


def bbox_from_polygon(poly: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def polygon_to_coco_polygon(poly: Sequence[Tuple[float, float]]) -> List[List[float]]:
    flat: List[float] = []
    for x, y in poly:
        flat.extend([float(x), float(y)])
    return [flat]


def export_coco_segmentation_json(
    records: Sequence[ImageRecord],
    categories: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    coco = coco_template(categories)
    next_ann = 1
    for i, rec in enumerate(records, start=1):
        coco["images"].append(
            {"id": i, "file_name": rec.rel_image, "width": rec.width, "height": rec.height}
        )
        cat = rec.category_ids[0] if rec.category_ids else 1
        for poly in rec.polygons:
            if len(poly) < 3:
                continue
            seg = polygon_to_coco_polygon(poly)
            x, y, w, h = bbox_from_polygon(poly)
            coco["annotations"].append(
                {
                    "id": next_ann,
                    "image_id": i,
                    "category_id": cat,
                    "segmentation": seg,
                    "area": float(w * h),
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }
            )
            next_ann += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")


def export_yolo_segmentation_dir(
    records: Sequence[ImageRecord],
    labels_dir: Path,
    class_id: int = 0,
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        lines = []
        for poly in rec.polygons:
            if len(poly) >= 3:
                lines.append(yolo_segmentation_line(class_id, poly, rec.width, rec.height))
        (labels_dir / f"{rec.image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_obb_dir(
    records: Sequence[Tuple[str, int, Sequence[Tuple[float, float]], int, int]],
    labels_dir: Path,
) -> None:
    """
    ``records`` : (image_id, class_id, quad_xy, width, height)
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[str]] = {}
    for image_id, cid, quad, w, h in records:
        grouped.setdefault(image_id, []).append(yolo_obb_line(cid, quad, w, h))
    for image_id, lines in grouped.items():
        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def polygons_to_geojson(
    features: List[Dict[str, Any]],
    crs_hint: Optional[str] = None,
) -> Dict[str, Any]:
    fc: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs_hint:
        fc["crs"] = {"type": "name", "properties": {"name": crs_hint}}
    return fc


def shapely_polygons_to_geojson_features(
    polygons: Sequence[Any],
    properties: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if Polygon is None or mapping is None:
        raise RuntimeError("shapely requis pour shapely_polygons_to_geojson_features")
    feats = []
    for poly in polygons:
        feats.append(
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": dict(properties or {}),
            }
        )
    return feats


def write_unified_dataset(
    records: Sequence[ImageRecord],
    out_root: Path,
    *,
    copy_images: bool = True,
    coco_path: Optional[str] = "coco_segmentation.json",
    yolo_labels: bool = True,
    class_id: int = 0,
) -> Path:
    """
    Écrit la structure ``parking_capacity_dataset`` sous ``out_root``.
    Copie les images depuis chemins absolus si ``rel_image`` est un chemin existant.
    """
    base, img_d, lbl_d, msk_d = ensure_unified_layout(out_root)
    meta_rows: List[Dict[str, Any]] = []
    coco = coco_template([{"id": 1, "name": "parking"}])
    next_ann_id = 1
    for idx, rec in enumerate(records, start=1):
        src = Path(rec.rel_image)
        if src.is_file():
            dst_name = f"{rec.image_id}{src.suffix.lower()}"
            dst = img_d / dst_name
            if copy_images:
                shutil.copy2(src, dst)
            rel_img = f"images/{dst_name}"
        else:
            rel_img = rec.rel_image
        w, h = rec.width, rec.height
        mask = polygon_to_binary_mask(rec.polygons, w, h)
        save_mask_png(mask, msk_d / f"{rec.image_id}.png")
        if yolo_labels:
            lines = []
            for poly in rec.polygons:
                if len(poly) >= 3:
                    lines.append(yolo_segmentation_line(class_id, poly, w, h))
            (lbl_d / f"{rec.image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        coco["images"].append({"id": idx, "file_name": rel_img, "width": w, "height": h})
        cat = rec.category_ids[0] if rec.category_ids else 1
        for poly in rec.polygons:
            if len(poly) < 3:
                continue
            seg = polygon_to_coco_polygon(poly)
            x, y, bw, bh = bbox_from_polygon(poly)
            area = float(bw * bh)
            coco["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": idx,
                    "category_id": cat,
                    "segmentation": seg,
                    "area": area,
                    "bbox": [x, y, bw, bh],
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1
        meta_rows.append(
            {
                "image_id": rec.image_id,
                "split": rec.split,
                "width": w,
                "height": h,
                "image": rel_img,
                "mask": f"masks/{rec.image_id}.png",
                "label": f"labels/{rec.image_id}.txt",
                "source_dataset": rec.source_dataset,
                "n_polygons": len(rec.polygons),
            }
        )
    write_metadata_jsonl(meta_rows, base / "metadata.jsonl")
    if coco_path:
        (base / coco_path).write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return base


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total
