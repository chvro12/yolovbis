#!/usr/bin/env python3
"""Mask2Former — point d’entrée ; entraînement complet via pipeline HF + COCO (voir doc)."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Mask2Former — voir docs/training_satellite_models.md")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    msg = (
        "Mask2Former : exporter vos données en COCO instance segmentation "
        "(coco_segmentation*.json depuis datasets-prepare), puis utiliser le script "
        "official Hugging Face `examples/pytorch/instance-segmentation` ou "
        "`transformers` Trainer avec Mask2FormerForUniversalSegmentation.\n"
        f"Dataset : {args.dataset_root}\n"
        f"Sortie : {args.output_dir}\n"
        f"Epochs : {args.epochs}\n"
        "Installez : pip install parking-capacity[train_satellite]\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
