"""Split train/val géographique (grille lon/lat, sans fuite spatiale grossière)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def geographic_train_val_mask(
    df: pd.DataFrame,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    val_frac: float = 0.15,
    seed: int = 42,
    precision: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regroupe les lignes par cellule (lon, lat) arrondis puis répartit les **cellules**
    entre train et validation (évite de mélanger des puces voisines dans les deux splits).
    Retourne (mask_train, mask_val) booléens alignés sur df.index.
    """
    n = len(df)
    if n < 2:
        m = np.ones(n, dtype=bool)
        return m, ~m
    if lon_col not in df.columns or lat_col not in df.columns:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, min(n - 1, int(round(n * val_frac))))
        val_idx = perm[:n_val]
        mask_val = np.zeros(n, dtype=bool)
        mask_val[val_idx] = True
        return ~mask_val, mask_val

    buckets = (
        df[lat_col].astype(float).round(precision).astype(str)
        + "_"
        + df[lon_col].astype(float).round(precision).astype(str)
    )
    uniq = pd.unique(buckets)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_val_cells = max(1, int(round(len(uniq) * val_frac)))
    val_cells = set(uniq[order[:n_val_cells]])
    mask_val = buckets.isin(val_cells).to_numpy()
    return ~mask_val, mask_val


def indices_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.nonzero(mask)[0]
