"""Baseline capacity choice."""

from parking_capacity.baseline import BaselineSignals, choose_baseline_capacity, compare_ml_to_baseline


def test_choose_baseline_osm_first():
    sig = BaselineSignals(
        osm_capacity_tagged=50,
        osm_parking_space_count=0,
        area_mid_places=10,
        area_min_places=8,
        area_max_places=12,
        area_total_m2=300,
        vision_places=5,
        ml_places=3,
    )
    v, m, _ = choose_baseline_capacity(sig)
    assert v == 50 and m == "osm_capacity"


def test_compare_ml_warning():
    msg = compare_ml_to_baseline(100.0, 10)
    assert "50" in msg or "diffère" in msg
