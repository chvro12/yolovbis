"""Tests export Colab (légers)."""

from __future__ import annotations

import shutil
import zipfile
import json
from pathlib import Path

from parking_capacity.colab_export import (
    run_doctor_build,
    run_export_colab_training,
    run_import_colab_model,
)


def test_export_colab_training_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]

    out = tmp_path / "colab_export"
    r = run_export_colab_training(
        out,
        dataset="apklot",
        include_current_dataset=False,
        include_config=True,
        include_notebooks=True,
        max_dataset_bytes=100,
        project_root=root,
    )
    assert r["ok"] is True
    assert (out / "parking_capacity_colab.zip").is_file()
    assert (out / "build_info.json").is_file()
    assert (out / "README_COLAB.md").is_file()
    assert (out / "requirements_colab.txt").is_file()
    assert (out / "commands.sh").is_file()
    assert (out / "train_colab.ipynb").is_file()
    assert (out / "project_snapshot" / "pyproject.toml").is_file()

    with zipfile.ZipFile(out / "parking_capacity_colab.zip") as zf:
        names = zf.namelist()
    assert "train_colab.ipynb" in names
    assert "build_info.json" in names
    assert "README_COLAB.md" in names

    doc = run_doctor_build(export_dir=out)
    assert doc["notebook_sha256_current"]
    assert doc["notebook_sha256_exported"] == doc["notebook_sha256_current"]
    assert doc["ok"] is True

    bi = json.loads((out / "build_info.json").read_text(encoding="utf-8"))
    cmds = bi.get("cli_commands") or []
    assert "doctor-build" in cmds
    assert "export-colab-training" in cmds


def test_import_colab_model_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "pack"
    weights = src / "run1" / "yolo_train" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"x")
    (weights / "last.pt").write_bytes(b"y")
    (src / "run1" / "yolo_train").mkdir(parents=True, exist_ok=True)
    (src / "run1" / "yolo_train" / "results.csv").write_text("epoch,mAP\n", encoding="utf-8")
    (src / "run1" / "train_metrics.json").write_text("{}", encoding="utf-8")
    (src / "ds").mkdir(parents=True)
    (src / "ds" / "dataset.yaml").write_text("path: .\n", encoding="utf-8")

    zip_base = tmp_path / "colab_results"
    shutil.make_archive(str(zip_base), "zip", root_dir=src)
    zip_path = Path(str(zip_base) + ".zip")

    dest = tmp_path / "models" / "apklot_yolo"
    r = run_import_colab_model(zip_path, dest)
    assert r["ok"]
    assert (dest / "best.pt").is_file()
    assert (dest / "dataset.yaml").is_file()
