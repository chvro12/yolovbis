"""Tests Phase 4 — validation terrain : métriques + segmentation + rapport."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from parking_capacity.benchmark_validation import (
    compute_metrics,
    identify_failing_cases,
    load_front_validation_csv,
    segment_metrics,
    write_front_validation_report,
    write_validation_report,
)


def test_compute_metrics_basic():
    est = [10, 20, 30, 40, 50]
    hum = [12, 18, 32, 38, 55]
    m = compute_metrics(est, hum)
    assert m.n_predicted == 5
    assert m.n_refused == 0
    # erreurs : 2, 2, 2, 2, 5 → MAE = 13/5 = 2.6
    assert m.mae == pytest.approx(2.6, abs=0.05)
    # MAPE = moyenne |err|/max(hum,1)
    expected_mape = sum(abs(e - h) / max(h, 1) for e, h in zip(est, hum)) / 5
    assert m.mape == pytest.approx(expected_mape, abs=1e-3)
    assert m.rmse is not None and m.rmse >= m.mae
    assert m.r2 is not None and 0 < m.r2 <= 1


def test_compute_metrics_refusal():
    est = [10, None, 30, None, 50]
    hum = [12, 18, 32, 38, 55]
    m = compute_metrics(est, hum)
    assert m.n_addresses_total == 5
    assert m.n_predicted == 3
    assert m.n_refused == 2
    assert m.refusal_rate == pytest.approx(0.4)


def test_compute_metrics_missing_human():
    est = [10, 20, 30]
    hum = [12, None, 32]
    m = compute_metrics(est, hum)
    # Une seule ligne sans human → exclue du total
    assert m.n_addresses_total == 2
    assert m.n_predicted == 2


def test_accuracy_thresholds():
    est = [10, 11, 12, 13, 14]
    hum = [10, 10, 10, 10, 10]
    m = compute_metrics(est, hum)
    # erreurs : 0, 1, 2, 3, 4
    assert m.accuracy_pm["pm1"] == pytest.approx(0.4)  # 0 et 1
    assert m.accuracy_pm["pm3"] == pytest.approx(0.8)  # 0,1,2,3
    assert m.accuracy_pm["pm5"] == 1.0


def test_bootstrap_ci_runs_when_enough_samples():
    est = [10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    hum = [11, 13, 14, 22, 23, 32, 34, 38, 47, 52]
    m = compute_metrics(est, hum, with_bootstrap=True)
    assert m.mae_ci95 is not None
    lo, hi = m.mae_ci95
    assert lo <= m.mae <= hi


def test_segment_metrics_groups_by_column():
    df = pd.DataFrame({
        "estimated_capacity": [10, 20, 30, 40, 50, 60],
        "human_count":        [11, 18, 28, 38, 55, 65],
        "site_type":          ["small", "small", "large", "large", "large", "small"],
    })
    seg = segment_metrics(df, segment_cols=["site_type"])
    assert "__overall__" in seg
    assert "site_type" in seg
    assert "small" in seg["site_type"]
    assert "large" in seg["site_type"]
    # Chaque segment a ses propres métriques
    assert seg["site_type"]["small"]["n_predicted"] == 3
    assert seg["site_type"]["large"]["n_predicted"] == 3


def test_identify_failing_cases():
    df = pd.DataFrame({
        "address": ["a", "b", "c", "d"],
        "estimated_capacity": [10, 100, 30, 5],
        "human_count":        [12, 15, 30, 50],
    })
    failing = identify_failing_cases(df, error_threshold=20, top_k=5)
    # b a |err|=85 ; d a |err|=45 ; a et c en dessous du seuil
    assert len(failing) == 2
    assert failing[0]["address"] == "b"
    assert failing[1]["address"] == "d"


def test_write_validation_report_creates_all_files(tmp_path):
    df = pd.DataFrame({
        "address": [f"addr {i}" for i in range(8)],
        "estimated_capacity": [10, 20, 30, 40, 50, None, 70, 80],
        "human_count":        [11, 22, 28, 42, 52, 60, 75, 78],
        "site_type":          ["s"]*4 + ["l"]*4,
        "semantic_confidence":["weak"]*4 + ["medium"]*4,
    })
    md = write_validation_report(df, tmp_path)
    assert md.is_file()
    assert (tmp_path / "validation_summary.json").is_file()
    assert (tmp_path / "validation_segments.json").is_file()
    assert (tmp_path / "validation_failing_cases.json").is_file()
    assert (tmp_path / "per_address.csv").is_file()
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["n_addresses_total"] == 8
    assert summary["n_refused"] == 1
    segments = json.loads((tmp_path / "validation_segments.json").read_text())
    assert "site_type" in segments
    assert "semantic_confidence" in segments
    md_text = md.read_text()
    assert "MAE" in md_text
    assert "MAPE" in md_text
    assert "site_type" in md_text


def test_empty_dataframe_no_crash():
    m = compute_metrics([], [])
    assert m.n_predicted == 0
    assert m.mae is None


def test_load_front_validation_csv_maps_ui_columns(tmp_path):
    p = tmp_path / "front.csv"
    p.write_text(
        "input_address,predicted_capacity,predicted_min,predicted_max,true_capacity,primary_source\n"
        "addr a,10,8,12,11,private_marked_slots\n"
        "addr b,,0,0,5,no_private_parking_detected\n",
        encoding="utf-8",
    )
    df = load_front_validation_csv(p)
    assert list(df["address"]) == ["addr a", "addr b"]
    assert list(df["human_count"]) == [11, 5]
    assert df.loc[0, "estimated_capacity"] == 10
    assert pd.isna(df.loc[1, "estimated_capacity"])
    assert df.loc[0, "abs_error"] == 1


def test_write_front_validation_report(tmp_path):
    p = tmp_path / "front.csv"
    p.write_text(
        "input_address,predicted_capacity,true_capacity,primary_source,primary_confidence,verdict\n"
        "addr a,10,11,private_marked_slots,medium,acceptable\n"
        "addr b,0,0,no_private_parking_detected,medium,correct\n"
        "addr c,,5,,low,sous-estime\n"
        "addr d,30,10,private_marked_slots,low,sur-estime\n",
        encoding="utf-8",
    )
    md = write_front_validation_report(p, tmp_path / "out")
    assert md.is_file()
    summary = json.loads((tmp_path / "out" / "validation_summary.json").read_text())
    assert summary["n_addresses_total"] == 4
    assert summary["n_predicted"] == 3
    assert summary["n_refused"] == 1
