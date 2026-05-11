"""Génération d’un jeu de données : puce orthophoto (WMS) + métadonnées par ligne de labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from parking_capacity.imagery_wms import fetch_ortho_chip
from parking_capacity.data_sources.label_heuristics import (
    pick_capacity_column,
    pick_lonlat_columns,
)


def detect_label_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """Retourne (lon_col, lat_col, capacity_col) avec détection heuristique."""
    lon, lat = pick_lonlat_columns(df)
    if lon is None or lat is None:
        for a, b in (("lon", "lat"), ("longitude", "latitude"), ("x", "y")):
            if a in df.columns and b in df.columns:
                lon, lat = a, b
                break
    cap = pick_capacity_column(df)
    return lon, lat, cap


def build_chip_dataset(
    labels_csv: Path,
    output_dir: Path,
    *,
    lon_column: Optional[str] = None,
    lat_column: Optional[str] = None,
    capacity_column: Optional[str] = None,
    max_rows: int = 2000,
    delay_s: float = 0.6,
    half_side_m: float = 55.0,
    chip_pixels: int = 512,
    wms_base: Optional[str] = None,
    wms_layer: Optional[str] = None,
    require_capacity: bool = True,
    client: Optional[httpx.Client] = None,
    cache_dir: Optional[Path] = None,
    refresh_imagery: bool = False,
) -> Path:
    """
    Pour chaque ligne avec lon/lat valides, télécharge une puce BD ORTHO et enregistre
    `images/{i:06d}.png` + `manifest.csv` + `manifest.jsonl` (une ligne JSON par puce).

    Retourne le chemin du manifest CSV.
    """
    from parking_capacity.imagery_wms import DEFAULT_WMS_BASE, DEFAULT_WMS_LAYER

    df = pd.read_csv(labels_csv)
    if df.empty:
        raise ValueError("CSV de labels vide")

    lon_c = lon_column
    lat_c = lat_column
    cap_c = capacity_column
    if lon_c is None or lat_c is None or cap_c is None:
        d_lon, d_lat, d_cap = detect_label_columns(df)
        lon_c = lon_c or d_lon
        lat_c = lat_c or d_lat
        cap_c = cap_c if capacity_column is not None else d_cap

    if lon_c is None or lat_c is None:
        raise ValueError(
            "Colonnes lon/lat introuvables. Précisez --lon-column / --lat-column ou "
            "utilisez un CSV avec longitude/latitude ou lon/lat."
        )

    if require_capacity and not cap_c:
        raise ValueError(
            "Colonne capacité introuvable. Passez --capacity-column ou utilisez "
            "`--no-require-capacity` pour exporter des puces sans label numérique."
        )

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    own = client is None
    if own:
        client = httpx.Client(timeout=90.0, follow_redirects=True)

    manifest_rows: List[Dict[str, Any]] = []
    n_written = 0

    try:
        for idx, row in df.iterrows():
            if n_written >= max_rows:
                break
            try:
                lo = float(row[lon_c])
                la = float(row[lat_c])
            except (TypeError, ValueError, KeyError):
                continue
            if not (-180 <= lo <= 180 and -90 <= la <= 90):
                continue

            cap_val: Optional[int] = None
            if cap_c and cap_c in df.columns:
                try:
                    v = row[cap_c]
                    if pd.notna(v) and str(v).strip() != "":
                        cap_val = int(float(v))
                except (ValueError, TypeError):
                    cap_val = None
            if require_capacity and cap_val is None:
                continue

            time.sleep(delay_s)
            chip = fetch_ortho_chip(
                lo,
                la,
                half_side_m=half_side_m,
                width_px=chip_pixels,
                height_px=chip_pixels,
                wms_base=wms_base or DEFAULT_WMS_BASE,
                layer=wms_layer or DEFAULT_WMS_LAYER,
                client=client,
                cache_dir=cache_dir,
                refresh_imagery=refresh_imagery,
                analysis_radius_m=float(half_side_m),
            )
            fname = f"{n_written:06d}.png"
            fpath = img_dir / fname
            chip.image.save(fpath, format="PNG")

            meta = _row_meta(row, idx)
            meta.update(
                {
                    "image_relative": f"images/{fname}",
                    "lon": lo,
                    "lat": la,
                    "capacity": cap_val,
                    "bbox_3857_minx": chip.minx,
                    "bbox_3857_miny": chip.miny,
                    "bbox_3857_maxx": chip.maxx,
                    "bbox_3857_maxy": chip.maxy,
                    "chip_pixels": chip_pixels,
                    "half_side_m": half_side_m,
                    "wms_layer": chip.layer,
                }
            )
            for extra in ("area_m2", "osm_type", "osm_id", "tags_json"):
                if extra in row.index and pd.notna(row.get(extra)):
                    try:
                        meta[extra] = row[extra]
                    except Exception:
                        meta[extra] = str(row[extra])
            manifest_rows.append(meta)
            n_written += 1
    finally:
        if own and client is not None:
            client.close()

    man_csv = output_dir / "manifest.csv"
    man_jsonl = output_dir / "manifest.jsonl"
    pd.DataFrame(manifest_rows).to_csv(man_csv, index=False)
    with man_jsonl.open("w", encoding="utf-8") as f:
        for rec in manifest_rows:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    readme = output_dir / "README_CHIPS.txt"
    readme.write_text(
        "Jeu de puces orthophoto BD ORTHO (WMS Géoplateforme).\n"
        "Respecter les CGU IGN / Géoplateforme. Usage typique : entraînement / évaluation ML.\n"
        f"Lignes exportées : {len(manifest_rows)}.\n",
        encoding="utf-8",
    )
    return man_csv


def _row_meta(row: pd.Series, idx: Any) -> Dict[str, Any]:
    """Sérialise quelques champs utiles de la ligne source (sans objets non JSON)."""
    keys = (
        "catalog_resource_url",
        "catalog_dataset_title",
        "catalog_source",
        "confidence",
        "row_index",
        "capacity_column_guess",
    )
    out: Dict[str, Any] = {"source_row_index": idx}
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            try:
                out[k] = row[k]
            except Exception:
                out[k] = str(row[k])
    return out
