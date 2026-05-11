#!/usr/bin/env python3
"""Fine-tune SegFormer sur ``parking_capacity_dataset`` (masques PNG)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def _need(extra: str) -> None:
    raise RuntimeError(
        f"Dépendance manquante : {extra}. Installez : pip install parking-capacity[train_satellite]"
    )


def load_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    meta = dataset_root / "metadata.jsonl"
    pairs: List[Tuple[Path, Path]] = []
    root = dataset_root
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            im = root / row["image"]
            msk = root / row["mask"]
            if im.is_file() and msk.is_file():
                pairs.append((im, msk))
    else:
        for mpath in (dataset_root / "masks").glob("*.png"):
            stem = mpath.stem
            for ext in (".jpg", ".jpeg", ".png"):
                ip = dataset_root / "images" / f"{stem}{ext}"
                if ip.is_file():
                    pairs.append((ip, mpath))
                    break
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Entraînement SegFormer segmentation parking")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name", default="nvidia/mit-b5")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--img-size", type=int, default=512)
    args = p.parse_args()

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        _need(str(e))

    pairs = load_pairs(args.dataset_root)
    if len(pairs) < 2:
        raise SystemExit(f"Pas assez d’images : {args.dataset_root}")

    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "background", 1: "parking"},
        label2id={"background": 0, "parking": 1},
        ignore_mismatched_sizes=True,
    )

    class MaskDataset(Dataset):
        def __init__(self, pp: List[Tuple[Path, Path]]) -> None:
            self.pairs = pp

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, i: int) -> Dict[str, Any]:
            ip, mp = self.pairs[i]
            image = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            mask = (mask > 127).astype(np.int64)
            enc = processor(images=image, segmentation_masks=mask, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = enc["labels"].squeeze(0)
            return item

    n = len(pairs)
    n_val = max(1, n // 10)
    ds_train = MaskDataset(pairs[:-n_val])
    ds_val = MaskDataset(pairs[-n_val:])

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        out: Dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collate,
    )
    trainer.train()
    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    print(f"Terminé — poids sous {args.output_dir}")


if __name__ == "__main__":
    main()
