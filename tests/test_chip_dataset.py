"""Tests build_chip_dataset avec WMS mocké."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from parking_capacity.imagery_wms import OrthoChip


def test_build_chip_dataset_mocked(tmp_path: Path):
    csv = tmp_path / "labels.csv"
    csv.write_text("lon,lat,capacity\n2.35,48.86,10\n2.36,48.87,20\n", encoding="utf-8")

    def _fake_chip(*args, **kwargs):
        return OrthoChip(
            image=Image.new("RGB", (8, 8), color=(10, 20, 30)),
            minx=0.0,
            miny=1.0,
            maxx=100.0,
            maxy=101.0,
            width_px=8,
            height_px=8,
            layer="TEST",
        )

    out = tmp_path / "ds"
    with patch("parking_capacity.chip_dataset.fetch_ortho_chip", side_effect=_fake_chip):
        from parking_capacity.chip_dataset import build_chip_dataset

        man = build_chip_dataset(
            csv,
            out,
            lon_column="lon",
            lat_column="lat",
            capacity_column="capacity",
            max_rows=10,
            delay_s=0.0,
            half_side_m=10.0,
            chip_pixels=8,
        )

    assert man.exists()
    mdf = pd.read_csv(man)
    assert len(mdf) == 2
    assert (out / "images" / "000000.png").exists()
    assert mdf.iloc[0]["capacity"] == 10
