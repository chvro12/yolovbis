"""Tests classification APKLOT et garde-fous."""

from __future__ import annotations

from pathlib import Path

from parking_capacity.datasets_satellite.apklot import classify_apklot_path, _voc_allowed


def test_classify_satellite_vs_camera() -> None:
    assert classify_apklot_path(Path("/repo/1. Satellite/Dataset/foo.jpg")) == "satellite"
    assert classify_apklot_path(Path("/repo/2. Camera/segmentation_1/x.png")) == "camera"


def test_voc_allowed_modes() -> None:
    assert _voc_allowed("satellite", "satellite") is True
    assert _voc_allowed("camera", "satellite") is False
    assert _voc_allowed("camera", "camera") is True
    assert _voc_allowed("satellite", "all") is True
