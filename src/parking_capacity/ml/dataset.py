"""Jeu PyTorch : images de puces + capacité (régression)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _apply_target(y: float, target_transform: str) -> float:
    if target_transform == "log1p":
        return math.log1p(max(0.0, y))
    return y


class ChipRegressionDataset(Dataset):
    """Lit `manifest.csv` (colonnes `image_relative`, `capacity`) sous `root_dir`."""

    def __init__(
        self,
        manifest_csv: Path,
        root_dir: Path,
        *,
        img_size: int = 128,
        transform: Optional[Callable] = None,
        augment_train: bool = False,
        target_transform: str = "none",
    ) -> None:
        self.root = Path(root_dir)
        self.img_size = img_size
        self.transform = transform
        self.augment_train = augment_train
        self.target_transform = target_transform
        df = pd.read_csv(manifest_csv)
        if "image_relative" not in df.columns or "capacity" not in df.columns:
            raise ValueError("manifest.csv doit contenir image_relative et capacity")
        df = df.dropna(subset=["capacity"])
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        df = df.dropna(subset=["capacity"])
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = self.root / str(row["image_relative"])
        img = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.Resampling.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        if self.augment_train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.3:
                x = torch.flip(x, dims=[1])
            if random.random() < 0.2:
                jitter = 0.03 * torch.randn_like(x)
                x = (x + jitter).clamp(0.0, 1.0)
        if self.transform:
            x = self.transform(x)
        cap = float(row["capacity"])
        y = torch.tensor(_apply_target(cap, self.target_transform), dtype=torch.float32)
        return x, y


def build_synthetic_chip_dataset(
    output_dir: Path,
    n: int,
    *,
    img_size: int = 128,
    seed: int = 42,
) -> Path:
    """
    **Uniquement démo / tests CI** : génère des PNG à couleur unie (pas d’orthophoto,
    pas de parking réel). La « capacité » est une formule sur (R,G,B) pour vérifier que
    l’entraînement et l’évaluation tournent. Pour du réel : `build-chips` puis
    `train-model --chip-dir=...` sans `--synthetic-n`.
    """
    rng = random.Random(seed)
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for i in range(n):
        r = rng.randint(20, 235)
        g = rng.randint(20, 235)
        b = rng.randint(20, 235)
        capacity = int(round(0.35 * r + 0.35 * g + 0.30 * b))
        img = Image.new("RGB", (img_size, img_size), color=(r, g, b))
        fname = f"{i:06d}.png"
        rel = f"images/{fname}"
        img.save(img_dir / fname, format="PNG")
        rows.append({"image_relative": rel, "capacity": capacity, "synthetic": True, "row_index": i})
    man = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    meta = out / "synthetic_meta.json"
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "n": n,
                "formula": "capacity=int(round(0.35*r+0.35*g+0.30*b))",
            },
            indent=2,
        )
    )
    return man
