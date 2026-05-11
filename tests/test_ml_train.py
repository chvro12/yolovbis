"""Entraînement minimal (synthétique) pour valider le pipeline ML."""

import math
from pathlib import Path

from parking_capacity.ml.train import run_training


def test_train_synthetic_tiny(tmp_path: Path):
    out = tmp_path / "run"
    chips = tmp_path / "chips"
    summary = run_training(
        chip_dir=chips,
        manifest_csv=None,
        output_dir=out,
        synthetic_n=64,
        architecture="tiny",
        pretrained=False,
        img_size=64,
        epochs=2,
        batch_size=16,
        lr=1e-2,
        val_frac=0.2,
        seed=0,
        device_str="cpu",
    )
    assert (out / "model.pt").exists()
    assert summary["n_samples"] == 64
    assert summary["val"]["n"] > 0
    assert math.isfinite(float(summary["val"]["mae"]))
