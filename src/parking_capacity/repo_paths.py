"""Chemins standard relatifs à la racine du dépôt (local, CI, Codespaces)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def aerial_weights_dir() -> Path:
    return data_dir() / "aerial_weights"


def dota_finetuned_weight_candidates() -> tuple[Path, ...]:
    base = aerial_weights_dir() / "dota_finetune_v1"
    return (
        base / "run2/weights/best.pt",
        base / "run1/weights/best.pt",
    )


def finetuned_french_weights() -> Path:
    return aerial_weights_dir() / "finetuned_v1/run1/weights/best.pt"


def parking_seg_weights() -> Path:
    return data_dir() / "runs/essai_cli_train/best.pt"


def resolve_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_file():
            return path
    return None
