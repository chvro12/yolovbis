"""Split géographique manifest."""

import numpy as np
import pandas as pd

from parking_capacity.ml.geo_split import geographic_train_val_mask, indices_from_mask


def test_geographic_train_val_mask_stable():
    df = pd.DataFrame(
        {
            "lon": [2.3, 2.31, 5.0, 5.01],
            "lat": [48.8, 48.81, 43.6, 43.61],
            "capacity": [10, 20, 30, 40],
        }
    )
    m_tr, m_va = geographic_train_val_mask(df, val_frac=0.25, seed=0, precision=2)
    assert m_tr.sum() + m_va.sum() == len(df)
    assert m_va.any() and m_tr.any()
    tr_i = indices_from_mask(m_tr)
    va_i = indices_from_mask(m_va)
    assert len(np.intersect1d(tr_i, va_i)) == 0


def test_surface_untagged():
    from shapely.geometry import Polygon

    from parking_capacity.overpass import OsmParkingElement
    from parking_capacity.osm_aggregate import ParkingClassification, surface_capacity_range_for_untagged

    ring = [(2.0, 48.0), (2.001, 48.0), (2.001, 48.001), (2.0, 48.001), (2.0, 48.0)]
    poly = Polygon(ring)
    rows = [
        ParkingClassification(
            element=OsmParkingElement("way", 1, {}, ring),
            polygon=poly,
            capacity=None,
            on_parcel=True,
            in_buffer_only=False,
        )
    ]
    mid, mn, mx, a = surface_capacity_range_for_untagged(rows, on_parcel=None)
    assert a > 0
    assert mx >= mid >= mn
