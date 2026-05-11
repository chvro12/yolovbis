"""Tests conversions et registre datasets satellite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from parking_capacity.datasets_satellite.converters import (
    ImageRecord,
    write_unified_dataset,
)
from parking_capacity.datasets_satellite.registry import default_registry, load_registry
from parking_capacity.training.evaluate_segmentation import binary_iou


def test_binary_iou_perfect() -> None:
    a = np.ones((10, 10), dtype=bool)
    assert abs(binary_iou(a, a) - 1.0) < 1e-5


def test_write_unified_dataset(tmp_path: Path) -> None:
    img = tmp_path / "i.png"
    from PIL import Image

    Image.new("RGB", (32, 24), (128, 128, 128)).save(img)
    poly = [(4.0, 4.0), (28.0, 4.0), (28.0, 20.0), (4.0, 20.0)]
    rec = ImageRecord(
        image_id="t1",
        rel_image=str(img),
        split="train",
        width=32,
        height=24,
        polygons=[poly],
        category_ids=[1],
        source_dataset="test",
    )
    base = write_unified_dataset([rec], tmp_path / "out")
    assert (base / "metadata.jsonl").is_file()
    assert (base / "coco_segmentation.json").is_file()


def test_default_registry_has_four_datasets() -> None:
    reg = default_registry(Path("/tmp/_fake_root"))
    assert len(reg["datasets"]) == 4


def test_load_registry_creates_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reg = load_registry(tmp_path)
    assert "datasets" in reg
    assert (tmp_path / "data" / "datasets" / "dataset_registry.json").is_file()
