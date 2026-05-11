#!/usr/bin/env python3
"""Détection véhicules (YOLO) — signal secondaire ; préparer xView au format YOLO/COCO."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True, help="Répertoire contenant dataset.yaml")
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", type=Path, default=Path("runs/vehicle_det"))
    args = p.parse_args()

    exe = shutil.which("yolo")
    if exe is None:
        print("Ultralytics non trouvé : pip install ultralytics", file=sys.stderr)
        sys.exit(2)

    data_yaml = args.dataset_root / "dataset.yaml"
    if not data_yaml.is_file():
        print(
            f"Créez {data_yaml} pour votre jeu (voir Ultralytics detect).\n"
            "Pour xView : convertissez les annotations puis placez un dataset.yaml.",
            file=sys.stderr,
        )
        sys.exit(3)

    cmd = [
        exe,
        "detect",
        "train",
        f"data={data_yaml}",
        f"model={args.model}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"project={args.output_dir.parent}",
        f"name={args.output_dir.name}",
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
