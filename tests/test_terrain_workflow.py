"""Cache WMS, source-priority, diagnose-address et make-training-run (mock)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from parking_capacity.diagnose import create_diagnose_mock_transport, run_diagnose_address
from parking_capacity.imagery_wms import fetch_ortho_chip
from parking_capacity.make_training_run import make_training_run
from parking_capacity.pipeline import process_address
from parking_capacity.visual_evidence import compute_visual_evidence


def _png_bytes():
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_wms_disk_cache_second_fetch_hits_disk(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        u = str(request.url)
        if "SERVICE=WMS" in u.upper() or "GetMap" in u or "geopf" in u.lower():
            calls.append(1)
            return httpx.Response(200, content=_png_bytes(), headers={"Content-Type": "image/png"})
        return httpx.Response(404, text="unexpected")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    cache = tmp_path / "cache"
    lon, lat = 2.301234, 48.851234
    fetch_ortho_chip(
        lon,
        lat,
        half_side_m=55.0,
        width_px=64,
        height_px=64,
        client=client,
        cache_dir=cache,
        analysis_radius_m=50.0,
    )
    fetch_ortho_chip(
        lon,
        lat,
        half_side_m=55.0,
        width_px=64,
        height_px=64,
        client=client,
        cache_dir=cache,
        analysis_radius_m=50.0,
    )
    assert len(calls) == 1


def test_source_priority_osm_over_aerial_when_tagged(monkeypatch):
    """Sans vision : OSM tag capacity doit gagner en hybrid."""

    ban_json = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [2.3, 48.85]},
                "properties": {"label": "Test", "score": 0.9},
            }
        ]
    }
    parcel_json = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2.29, 48.84], [2.31, 48.84], [2.31, 48.86], [2.29, 48.86], [2.29, 48.84]]],
                },
                "properties": {"id": "0000000000000000"},
            }
        ],
    }
    overpass_json = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"amenity": "parking", "capacity": "42"},
                "geometry": [
                    {"lat": 48.849, "lon": 2.300},
                    {"lat": 48.849, "lon": 2.301},
                    {"lat": 48.851, "lon": 2.301},
                    {"lat": 48.851, "lon": 2.300},
                    {"lat": 48.849, "lon": 2.300},
                ],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api-adresse.data.gouv.fr" in url:
            return httpx.Response(200, json=ban_json)
        if "apicarto.ign.fr" in url and "parcelle" in url:
            return httpx.Response(200, json=parcel_json)
        if "overpass-api.de" in url or "interpreter" in url:
            return httpx.Response(200, json=overpass_json)
        return httpx.Response(404, text="not mocked")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    r = process_address(
        "1 rue test Paris",
        client=client,
        overpass_delay_s=0.0,
        use_vision=False,
        min_intersection_m2=1.0,
        source_priority="hybrid",
    )
    assert r.primary_source == "osm_parcelle"
    assert r.primary_capacity == 42
    assert r.source_priority_used == "hybrid"


def test_diagnose_address_mock_writes_artifacts(tmp_path):
    out = tmp_path / "diag"
    run_diagnose_address(
        "38 rue du Moulin à Vent (mock)",
        out,
        radius_m=50,
        mock=True,
        source_priority="hybrid",
    )
    assert (out / "chip.png").is_file()
    assert (out / "result.json").is_file()
    assert (out / "sources.json").is_file()
    assert (out / "warnings.txt").exists()
    assert (out / "debug_map.html").is_file()
    assert (out / "debug_overlay.png").is_file()
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert "visual_evidence_level" in data
    assert "capacity_provenance" in data
    assert "image_used" in data


def test_make_training_run_mock_report(tmp_path):
    report = make_training_run(
        tmp_path / "run1",
        bbox=None,
        preset=None,
        max_samples=80,
        mock=True,
        epochs=1,
    )
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "MAE" in text
    assert "mock" in text.lower() or "Mock" in text


def test_visual_evidence_levels():
    from parking_capacity.vision_estimate import VisionEstimate

    ve_none = compute_visual_evidence(None, osm_parking_polygon_in_scope=False, image_fetched=True)
    assert ve_none.visual_evidence_level == "none"

    v = VisionEstimate(
        parking_pixel_fraction=0.008,
        parking_area_m2=100.0,
        estimated_spaces=10,
        confidence="low",
        device="cpu",
        notes="",
    )
    ve_w = compute_visual_evidence(v, osm_parking_polygon_in_scope=False, image_fetched=True)
    assert ve_w.visual_evidence_level == "weak"
    assert ve_w.parking_spaces_detected_count is None


def test_create_diagnose_mock_transport_smoke():
    t = create_diagnose_mock_transport()
    c = httpx.Client(transport=t, timeout=10.0)
    r = c.get("https://api-adresse.data.gouv.fr/search/?q=test")
    assert r.status_code == 200
    c.close()
