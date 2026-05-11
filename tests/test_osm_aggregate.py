"""Tests unitaires pour l'agrégation OSM (sans réseau)."""

from shapely.geometry import Point, Polygon

from parking_capacity.osm_aggregate import (
    classify_parkings,
    sum_capacity,
)
from parking_capacity.overpass import OsmParkingElement


def test_sum_capacity_parcelle():
    parcel = Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002), (0, 0)])
    buf = Point(0.001, 0.001).buffer(0.003)
    elems = [
        OsmParkingElement(
            "way",
            1,
            {"amenity": "parking", "capacity": "10"},
            [(0.0005, 0.0005), (0.0015, 0.0005), (0.0015, 0.0015), (0.0005, 0.0015), (0.0005, 0.0005)],
        ),
        OsmParkingElement(
            "way",
            2,
            {"amenity": "parking"},
            [(10, 10), (10.001, 10), (10.001, 10.001), (10, 10.001), (10, 10)],
        ),
    ]
    classified = classify_parkings(elems, parcel, buf, min_intersection_m2=5.0)
    cap_p, n_p = sum_capacity(classified, on_parcel=True)
    assert cap_p == 10
    assert n_p == 1
