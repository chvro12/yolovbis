"""Validation produit : benchmark, manuel, pretty, ML garde-fous, go/no-go."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parking_capacity.benchmark_addresses import run_benchmark_addresses
from parking_capacity.evaluate_manual import run_evaluate_manual_review
from parking_capacity.go_no_go import write_go_no_go_report
from parking_capacity.human_output import format_run_address_pretty
from parking_capacity.ml.model_meta import model_meta_blocks_primary_ml, should_skip_ml_inference
from parking_capacity.pipeline import RowResult


def test_benchmark_addresses_mock(tmp_path):
    csv_in = tmp_path / "in.csv"
    csv_in.write_text(
        "address,radius_m,expected_capacity,notes\n\"Test mock\",50,,\n",
        encoding="utf-8",
    )
    out = tmp_path / "bench"
    run_benchmark_addresses(csv_in, out, mock=True)
    assert (out / "results.csv").is_file()
    assert (out / "results.json").is_file()
    assert (out / "benchmark_report.md").is_file()
    assert (out / "manual_review.csv").is_file()
    subs = [p for p in out.iterdir() if p.is_dir()]
    assert subs
    assert (subs[0] / "chip.png").is_file()
    assert (subs[0] / "result.json").is_file()
    man = Path(out / "manual_review.csv").read_text(encoding="utf-8")
    assert "human_count" in man
    assert "chip_path" in man


def test_evaluate_manual_review(tmp_path):
    rev = tmp_path / "manual_review.csv"
    rev.write_text(
        "address,estimated_capacity,min_capacity,max_capacity,method_used,"
        "visual_evidence_level,image_confidence,chip_path,overlay_path,human_count,human_notes,accepted\n"
        '"Adresse A",18,10,22,area_ratio,none,low,,,20,,\n',
        encoding="utf-8",
    )
    outd = tmp_path / "ev"
    p = run_evaluate_manual_review(rev, outd)
    assert p.is_file()
    summ = json.loads((outd / "summary.json").read_text(encoding="utf-8"))
    assert summ["n_pairs"] == 1
    assert summ["mae"] == 2.0


def test_run_address_pretty_format():
    row = RowResult(
        input_address="1 rue X",
        ban_label="1 rue X, Ville",
        ban_score=0.9,
        radius_m_used=50,
        estimated_capacity=18,
        min_capacity=14,
        max_capacity=22,
        method_used="area_ratio",
        primary_confidence="medium",
        visual_evidence_level="none",
        image_used=True,
        image_confidence="low",
        capacity_provenance="test",
        sources_used="ban|apicarto|osm_overpass|ign_wms_orthophoto",
        warnings="Note test.",
    )
    s = format_run_address_pretty(row)
    assert "Adresse" in s
    assert "Fourchette" in s
    assert "14" in s and "22" in s


def test_model_meta_blocks_and_force_ml(tmp_path):
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"x")
    meta = tmp_path / "model_meta.json"
    meta.write_text(
        json.dumps(
            {
                "dataset_mode": "synthetic",
                "n_train_samples": 200,
                "val_r2": 0.5,
            }
        ),
        encoding="utf-8",
    )
    assert model_meta_blocks_primary_ml(json.loads(meta.read_text()))
    skip, _, note = should_skip_ml_inference(ck, force_ml=False)
    assert skip is True
    assert "Inférence ML ignorée" in note
    skip2, _, _ = should_skip_ml_inference(ck, force_ml=True)
    assert skip2 is False


def test_go_no_go_report_mock(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    bench.joinpath("results.csv").write_text(
        "input_address,estimated_capacity,visual_evidence_level\nX,10,none\n",
        encoding="utf-8",
    )
    bench.joinpath("benchmark_report.md").write_text("# B\n", encoding="utf-8")
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    mp = model_dir / "model.pt"
    mp.write_bytes(b"y")
    (model_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "architecture": "tiny",
                "dataset_mode": "mock",
                "n_train_samples": 30,
                "val_r2": -0.2,
                "val_mae": 99.0,
                "val_rmse": 100.0,
                "created_at": "2026-01-01T00:00:00+00:00",
                "split_method": "random",
                "target_transform": "none",
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "go.md"
    write_go_no_go_report(bench, mp, out_md)
    text = out_md.read_text(encoding="utf-8")
    assert "Décision" in text
    assert "Modèle" in text
