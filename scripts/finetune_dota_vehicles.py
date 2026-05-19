"""Fine-tuning YOLOv8 sur DOTAv1 vehicles (51k bboxes réelles annotées humainement).

Stratégie anti-catastrophic-forgetting :
- Part de yolov8s.pt (COCO) — meilleure base que yolov8s-visdrone pour single-class.
- freeze=10 (gèle backbone, ne ré-entraîne que neck+head).
- lr0=1e-4 (modéré : suffisant pour adapter le head, pas assez pour casser le backbone).
- patience=8 (early stopping).
- Mosaic + mixup augmentation.
- imgsz=640 (compromis MPS vitesse/qualité).
- batch=8 (MPS plafond pratique).
"""

from __future__ import annotations

from pathlib import Path
import torch
from ultralytics import YOLO

DATASET_YAML = "/Users/mac/Yolo/datasets/DOTAv1_vehicles/dataset.yaml"
OUT_DIR = Path("/Users/mac/Yolo/data/aerial_weights/dota_finetune_v1")


def main():
    print(f"MPS : {torch.backends.mps.is_available()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # On part de yolov8s.pt (COCO) → 80 classes mais on remap en 1 classe via DOTA.
    # Cela évite le bagage VisDrone (10 classes) qu'on ne veut pas.
    m = YOLO("yolov8s.pt")
    print("Modèle source : yolov8s.pt (COCO 80 classes)")

    # Paramètres adaptés à MPS sur Mac M-series (images DOTA jusqu'à 4000×5000) :
    # - imgsz=512 (forcing ~1 sec/iter sur MPS contre ~6 sec avec 640)
    # - batch=4 (RAM raisonnable, évite swap)
    # - cache=False (évite explosion RAM 7+ GB sur DOTA)
    # - epochs=8 (suffisant avec freeze backbone + bonne base COCO)
    # - workers=2 (limite I/O contention)
    results = m.train(
        data=DATASET_YAML,
        epochs=8,
        imgsz=512,
        batch=4,
        workers=2,
        cache=False,
        device="mps",
        project=str(OUT_DIR),
        name="run2",
        freeze=10,
        lr0=1e-4,
        lrf=0.01,
        patience=4,
        mosaic=0.5,        # plus léger
        mixup=0.0,         # désactivé pour gagner mémoire
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        save=True,
        save_period=-1,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBEST WEIGHTS : {best}")


if __name__ == "__main__":
    main()
