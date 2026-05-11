#!/usr/bin/env python3
"""YOLOv8 segmentation — Ultralytics API (défauts : 20 epochs, imgsz 640, AMP, batch auto)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def detect_dataset_yaml(dataset_root: Path) -> Path:
    """Si ``dataset.yaml`` existe et décrit train/val, le réutilise."""
    if (dataset_root / "dataset.yaml").is_file():
        return dataset_root / "dataset.yaml"
    # layout YOLO standard images/train
    if (dataset_root / "images" / "train").is_dir() and (dataset_root / "labels" / "train").is_dir():
        yaml_path = dataset_root / "dataset.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {dataset_root.resolve()}",
                    "train: images/train",
                    "val: images/val",
                    "nc: 1",
                    "names: ['parking']",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return yaml_path
    raise FileNotFoundError(f"Layout YOLO introuvable sous {dataset_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--model", default="yolov8m-seg.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--output-dir", type=Path, default=Path("runs/yolo_seg"))
    p.add_argument(
        "--save-period",
        type=int,
        default=0,
        help="Sauvegarder un checkpoint tous les N epochs (0=désactivé). Utile sur Colab vers Drive.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--weights", type=Path, default=None, help="Checkpoint pour reprise")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installez ultralytics : pip install ultralytics", file=sys.stderr)
        sys.exit(2)

    from parking_capacity.datasets_satellite.yolo_seg_layout import ensure_yolo_seg_dataset

    ds_root = args.dataset_root
    if (ds_root / "metadata.jsonl").is_file():
        ds_root = ensure_yolo_seg_dataset(ds_root)

    yaml_path = detect_dataset_yaml(ds_root)
    model = YOLO(str(args.weights) if args.weights and args.weights.is_file() else args.model)

    train_kw: Dict[str, Any] = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        amp=True,
        project=str(args.output_dir.parent),
        name=args.output_dir.name,
        exist_ok=True,
    )
    if args.save_period >= 1:
        train_kw["save_period"] = args.save_period
    if args.resume:
        train_kw["resume"] = True if args.weights is None else str(args.weights)

    results = model.train(**train_kw)

    metrics_path = args.output_dir / "train_metrics.json"
    rd = getattr(results, "results_dict", None)
    if rd is not None:
        try:
            metrics_path.write_text(json.dumps(dict(rd), indent=2, default=float), encoding="utf-8")
        except Exception:
            pass

    csv_src = args.output_dir / "results.csv"
    if csv_src.is_file():
        shutil.copy2(csv_src, args.output_dir / "train_metrics.csv")

    print(f"Terminé — sortie {args.output_dir}")


if __name__ == "__main__":
    main()
