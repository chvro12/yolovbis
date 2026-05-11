"""Tests configuration GIS et requête Overpass transport."""

from __future__ import annotations

from parking_capacity.osm_transport import build_osm_transport_query
from parking_capacity.providers_config import load_gis_providers_config


def test_build_osm_transport_query_contains_highway_tags():
    q = build_osm_transport_query(49.73, 1.44, 80)
    assert "parking_aisle" in q
    assert "parking_entrance" in q
    assert "highway" in q
    assert "residential" in q
    assert "building" in q


def test_load_gis_providers_config_defaults():
    cfg = load_gis_providers_config(yaml_path=None, project_root=None)
    assert cfg.ign_wfs_url.startswith("https://")
    assert "overpass" in cfg.overpass_url
