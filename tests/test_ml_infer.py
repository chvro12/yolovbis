"""Inférence ML après un mini-entraînement synthétique."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from parking_capacity.ml.infer import predict_capacity_from_pil, read_inference_config
from parking_capacity.ml.train import run_training


def test_read_inference_config_after_train(tmp_path: Path) -> None:
    out = tmp_path / "run"
    chips = tmp_path / "chips"
    run_training(
        chip_dir=chips,
        manifest_csv=None,
        output_dir=out,
        synthetic_n=24,
        architecture="tiny",
        pretrained=False,
        img_size=64,
        epochs=2,
        batch_size=8,
        lr=1e-2,
        val_frac=0.2,
        seed=1,
        device_str="cpu",
    )
    ckpt = out / "model.pt"
    assert ckpt.is_file()
    assert (out / "model_meta.json").is_file()
    cfg = read_inference_config(ckpt)
    assert cfg["architecture"] == "tiny"
    assert cfg["img_size"] == 64
    assert math.isfinite(float(cfg["half_side_m"]))

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    pred, info = predict_capacity_from_pil(img, ckpt, device_str="cpu")
    assert math.isfinite(pred)
    assert info["architecture"] == "tiny"
