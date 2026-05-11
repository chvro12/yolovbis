"""Tests parseurs harvest (sans réseau)."""

import json
from pathlib import Path

import pandas as pd

from parking_capacity.data_sources.harvest_labels import harvest_resource_file
from parking_capacity.data_sources.label_heuristics import pick_capacity_column, rows_from_dataframe


def test_pick_capacity_column():
    df = pd.DataFrame([{"Nb_places": 10, "lon": 2.3, "lat": 48.8}])
    assert pick_capacity_column(df) == "Nb_places"


def test_rows_from_dataframe():
    df = pd.DataFrame(
        [
            {"capacity": 5, "longitude": 2.1, "latitude": 48.2},
            {"capacity": 7, "longitude": 2.2, "latitude": 48.3},
        ]
    )
    meta = {"catalog_resource_url": "http://example/x"}
    rows = rows_from_dataframe(df, meta=meta)
    assert len(rows) == 2
    assert rows[0]["capacity"] == 5
    assert rows[0]["lon"] == 2.1


def test_harvest_geojson_file(tmp_path: Path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [2.35, 48.86]},
                "properties": {"nb_places": 42, "name": "P1"},
            }
        ],
    }
    p = tmp_path / "x.geojson"
    p.write_text(json.dumps(fc), encoding="utf-8")
    rows = harvest_resource_file(
        p,
        resource_format="GEOJSON",
        meta={"catalog_resource_url": "local"},
        work_dir=tmp_path / "w",
    )
    assert len(rows) == 1
    assert rows[0]["capacity"] == 42
    assert abs(rows[0]["lon"] - 2.35) < 1e-6
