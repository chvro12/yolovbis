"""Télécharge des ressources du catalogue et extrait capacité + coordonnées (heuristique)."""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from shapely.geometry import shape

from parking_capacity.data_sources.download import download_url_to_file
from parking_capacity.data_sources.label_heuristics import (
    dataframe_from_json_records,
    pick_capacity_key,
    rows_from_dataframe,
)

ALLOWED_FORMATS = {"CSV", "GEOJSON", "JSON", "TEXT", "ZIP"}


def _read_csv_best_effort(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        for sep in (None, ",", ";", "\t"):
            try:
                if sep is None:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc)
                return pd.read_csv(path, sep=sep, encoding=enc)
            except Exception:
                continue
    try:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, encoding="latin-1")


def _parse_geojson_data(data: dict, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    if data.get("type") == "Feature":
        data = {"type": "FeatureCollection", "features": [data]}
    out: List[Dict[str, Any]] = []
    if data.get("type") == "FeatureCollection":
        feats = data.get("features") or []
        for i, feat in enumerate(feats):
            props = dict(feat.get("properties") or {})
            geom = feat.get("geometry")
            lon, lat = None, None
            if geom:
                try:
                    g = shape(geom)
                    c = g.centroid
                    lon, lat = float(c.x), float(c.y)
                except Exception:
                    pass
            cap_key = pick_capacity_key(props)
            cap = None
            if cap_key is not None:
                try:
                    cap = int(float(props[cap_key])) if props.get(cap_key) not in (None, "") else None
                except (ValueError, TypeError):
                    cap = None
            out.append(
                {
                    **meta,
                    "row_index": i,
                    "capacity": cap,
                    "lon": lon,
                    "lat": lat,
                    "capacity_column_guess": cap_key,
                    "lon_column_guess": "geometry.centroid",
                    "lat_column_guess": "geometry.centroid",
                    "confidence": "high"
                    if cap is not None and lon is not None
                    else ("medium" if lon is not None else "low"),
                }
            )
        return out
    return [{**meta, "parse_error": "geojson_type_non_supporte", "confidence": "none"}]


def _parse_geojson(path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    if not isinstance(data, dict):
        return [{**meta, "parse_error": "geojson_racine_invalide", "confidence": "none"}]
    return _parse_geojson_data(data, meta)


def _parse_json(path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    if isinstance(data, dict) and data.get("type") in ("FeatureCollection", "Feature"):
        return _parse_geojson_data(data, meta)
    df = dataframe_from_json_records(data)
    if df is not None and not df.empty:
        return rows_from_dataframe(df, meta=meta)
    return []


def _extract_zip_inner(path: Path, work: Path) -> Optional[Path]:
    work.mkdir(parents=True, exist_ok=True)
    suffixes = (".csv", ".geojson", ".json")
    with zipfile.ZipFile(path, "r") as zf:
        names = sorted(
            [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n],
            key=len,
        )
        for n in names:
            low = n.lower()
            if any(low.endswith(s) for s in suffixes):
                dest = work / Path(n).name
                dest.write_bytes(zf.read(n))
                return dest
    return None


def harvest_resource_file(
    path: Path,
    *,
    resource_format: str,
    meta: Dict[str, Any],
    work_dir: Path,
) -> List[Dict[str, Any]]:
    fmt = (resource_format or "").upper()
    if fmt == "ZIP":
        inner = _extract_zip_inner(path, work_dir / "zip_unpack")
        if inner is None:
            return [{**meta, "parse_error": "zip_sans_csv_ni_geojson", "confidence": "none"}]
        inner_fmt = "GEOJSON" if inner.suffix.lower() == ".geojson" else "JSON" if inner.suffix.lower() == ".json" else "CSV"
        return harvest_resource_file(inner, resource_format=inner_fmt, meta=meta, work_dir=work_dir)

    if fmt == "GEOJSON" or (fmt == "JSON" and path.suffix.lower() == ".geojson"):
        try:
            return _parse_geojson(path, meta)
        except Exception as e:  # noqa: BLE001
            return [{**meta, "parse_error": str(e), "confidence": "none"}]

    if fmt in {"JSON", "TEXT"}:
        try:
            rows = _parse_json(path, meta)
            if rows:
                return rows
        except Exception:
            pass
        # retenter comme CSV si lignes tabulaires
        try:
            df = _read_csv_best_effort(path)
            if not df.empty:
                return rows_from_dataframe(df, meta=meta)
        except Exception as e:  # noqa: BLE001
            return [{**meta, "parse_error": str(e), "confidence": "none"}]
        return [{**meta, "parse_error": "json_non_tabulaire", "confidence": "none"}]

    # CSV par défaut
    try:
        df = _read_csv_best_effort(path)
        if df.empty:
            return [{**meta, "parse_error": "csv_vide", "confidence": "none"}]
        return rows_from_dataframe(df, meta=meta)
    except Exception as e:  # noqa: BLE001
        return [{**meta, "parse_error": str(e), "confidence": "none"}]


def harvest_from_catalog(
    catalog_path: Path,
    output_path: Path,
    *,
    max_files: int = 200,
    max_mb_per_file: int = 40,
    delay_s: float = 0.75,
    client: Optional[httpx.Client] = None,
) -> int:
    """
    Lit le CSV `catalog_path` (sortie de `parking-capacity catalog`), télécharge
    les ressources autorisées et écrit `output_path` (CSV).
    Retourne le nombre de lignes produites.
    """
    cat = pd.read_csv(catalog_path)
    if cat.empty:
        raise ValueError("Catalogue vide")

    all_rows: List[Dict[str, Any]] = []
    own = client is None
    if own:
        client = httpx.Client(timeout=120.0, follow_redirects=True)

    tmp_root = Path.cwd() / ".harvest_tmp"
    tmp_root.mkdir(exist_ok=True)
    n_done = 0
    try:
        for _, row in cat.iterrows():
            if n_done >= max_files:
                break
            fmt = str(row.get("resource_format") or "").strip().upper()
            url = row.get("resource_url")
            if not url or not isinstance(url, str):
                continue
            if fmt not in ALLOWED_FORMATS:
                continue

            meta = {
                "catalog_source": row.get("source"),
                "catalog_dataset_title": row.get("dataset_title"),
                "catalog_dataset_page_url": row.get("dataset_page_url"),
                "catalog_resource_title": row.get("resource_title"),
                "catalog_resource_format": fmt,
                "catalog_resource_url": url,
            }

            suffix = ".zip" if fmt == "ZIP" else ".geojson" if fmt == "GEOJSON" else ".json" if fmt == "JSON" else ".csv"
            dest = tmp_root / f"dl_{n_done}{suffix}"
            try:
                time.sleep(delay_s)
                download_url_to_file(url, dest, max_bytes=max_mb_per_file * 1024 * 1024, client=client)
                work = tmp_root / f"work_{n_done}"
                rows = harvest_resource_file(dest, resource_format=fmt, meta=meta, work_dir=work)
                all_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                all_rows.append({**meta, "parse_error": str(e), "confidence": "none"})
            finally:
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
            n_done += 1
    finally:
        if own and client is not None:
            client.close()
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return len(out_df)
