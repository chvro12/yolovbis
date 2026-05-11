"""Heuristiques de colonnes (capacité, coordonnées) sur DataFrames / propriétés GeoJSON."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


CAPACITY_EXACT = [
    "nbplaces",
    "nbplacestotales",
    "nombreplaces",
    "nombredeplaces",
    "capacite",
    "capacity",
    "places",
    "placestotales",
    "totalplaces",
    "parkplaces",
    "nbplace",
    "nombreplacestotales",
    "capacitestationnement",
]

LON_CANDIDATES = ["lon", "lng", "longitude", "xlong", "x", "coordx", "lambertx", "lambert93x"]
LAT_CANDIDATES = ["lat", "latitude", "ylat", "y", "coordy", "lamberty", "lambert93y"]


def pick_capacity_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty or df.columns is None:
        return None
    norm_to_orig = {_norm(c): c for c in df.columns}
    for pat in CAPACITY_EXACT:
        if pat in norm_to_orig:
            return norm_to_orig[pat]
    # colonne contenant "place" et majoritairement numérique
    best: Optional[Tuple[str, float]] = None
    for c in df.columns:
        n = _norm(c)
        if "place" not in n and "capac" not in n and "nb" != n[:2]:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        ratio = float(s.notna().mean())
        if ratio < 0.2:
            continue
        if best is None or ratio > best[1]:
            best = (c, ratio)
    return best[0] if best else None


def pick_lonlat_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df.empty:
        return None, None
    norm_to_orig = {_norm(c): c for c in df.columns}
    lon_col: Optional[str] = None
    lat_col: Optional[str] = None
    for cand in LON_CANDIDATES:
        if cand in norm_to_orig:
            lon_col = norm_to_orig[cand]
            break
    for cand in LAT_CANDIDATES:
        if cand in norm_to_orig:
            lat_col = norm_to_orig[cand]
            break
    return lon_col, lat_col


def pick_capacity_key(props: Dict[str, Any]) -> Optional[str]:
    if not props:
        return None
    keys = list(props.keys())
    fake = pd.DataFrame([{k: props[k] for k in keys}])
    col = pick_capacity_column(fake)
    return col


def rows_from_dataframe(
    df: pd.DataFrame,
    *,
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Transforme un tableau en lignes normalisées (capacité + lon/lat si détectés)."""
    df2 = df.reset_index(drop=True)
    cap_col = pick_capacity_column(df2)
    lon_col, lat_col = pick_lonlat_columns(df2)
    out: List[Dict[str, Any]] = []
    for idx in range(len(df2)):
        row = df2.iloc[idx]
        rec: Dict[str, Any] = {**meta, "row_index": idx}
        rec["capacity_column_guess"] = cap_col
        rec["lon_column_guess"] = lon_col
        rec["lat_column_guess"] = lat_col
        if cap_col and cap_col in df2.columns:
            v = row.get(cap_col)
            try:
                rec["capacity"] = int(float(v)) if pd.notna(v) and str(v).strip() != "" else None
            except (ValueError, TypeError):
                rec["capacity"] = None
        else:
            rec["capacity"] = None
        if lon_col and lat_col and lon_col in df2.columns and lat_col in df2.columns:
            try:
                rec["lon"] = float(row[lon_col]) if pd.notna(row[lon_col]) else None
                rec["lat"] = float(row[lat_col]) if pd.notna(row[lat_col]) else None
            except (ValueError, TypeError):
                rec["lon"], rec["lat"] = None, None
        else:
            rec["lon"], rec["lat"] = None, None
        rec["confidence"] = (
            "high"
            if cap_col and lon_col and lat_col
            else ("medium" if cap_col else "low")
        )
        out.append(rec)
    return out


def dataframe_from_json_records(data: Any) -> Optional[pd.DataFrame]:
    """Si `data` est une liste d'objets homogènes, retourne un DataFrame."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return pd.DataFrame(data["data"])
    return None
