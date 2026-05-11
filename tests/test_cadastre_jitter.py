"""Repli APICarto : plusieurs points de requête autour du géocodage."""

from parking_capacity.cadastre import iter_parcel_query_points


def test_iter_parcel_query_points_includes_center():
    pts = list(iter_parcel_query_points(4.8587, 45.730513, jitter_m=(0.0, 3.0)))
    assert len(pts) >= 5
    lo, la = pts[0]
    assert abs(lo - 4.8587) < 1e-8 and abs(la - 45.730513) < 1e-8


def test_iter_parcel_query_points_deduplicates():
    pts = list(iter_parcel_query_points(0.0, 45.0, jitter_m=(0.0, 0.0)))
    assert len(pts) == 1
