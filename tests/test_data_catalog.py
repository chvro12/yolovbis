"""Tests filtres catalogue (sans réseau)."""

from parking_capacity.data_sources.pan import filter_datasets_by_keywords


def test_filter_pan_keywords():
    ds = [
        {"title": "Parkings de Testville", "slug": "parkings-testville", "tags": []},
        {"title": "Vélib", "slug": "velib", "tags": ["bike"]},
        {"title": "Base nationale des lieux de stationnement hors voirie", "slug": "bnls", "tags": []},
    ]
    out = filter_datasets_by_keywords(ds, ["stationnement", "parking"])
    titles = {d["title"] for d in out}
    assert "Parkings de Testville" in titles
    assert "Base nationale des lieux de stationnement hors voirie" in titles
    assert "Vélib" not in titles
