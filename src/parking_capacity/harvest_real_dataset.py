"""Construction d'un jeu orthophoto + labels OSM (capacity) dans une bbox."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from parking_capacity.geometry import to_metric
from parking_capacity.imagery_wms import DEFAULT_WMS_BASE, DEFAULT_WMS_LAYER, fetch_ortho_chip
from parking_capacity.osm_aggregate import element_to_polygon
from parking_capacity.overpass import OsmParkingElement, OverpassError, query_parkings_bbox_capacity


def parse_bbox_string(s: str) -> Tuple[float, float, float, float]:
    """`min_lon,min_lat,max_lon,max_lat` (ordre GIS courant)."""
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox doit être min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox invalide : min<max requis pour lon et lat")
    return min_lon, min_lat, max_lon, max_lat


def _centroid_lonlat(el: OsmParkingElement) -> Optional[Tuple[float, float]]:
    poly = element_to_polygon(el)
    if poly is None or poly.is_empty:
        return None
    c = poly.centroid
    return float(c.x), float(c.y)


def _capacity_int(tags: dict[str, str]) -> Optional[int]:
    v = tags.get("capacity")
    if not v:
        return None
    try:
        c = int(float(v))
        return c if c > 0 else None
    except ValueError:
        return None


def _area_m2(el: OsmParkingElement) -> float:
    poly = element_to_polygon(el)
    if poly is None or poly.is_empty:
        return 0.0
    return float(to_metric(poly).area)


def _quality_flags(cap: int, area_m2: float) -> List[str]:
    flags: List[str] = []
    if cap <= 0:
        flags.append("reject_capacity_non_positive")
    if area_m2 > 30 and cap > area_m2 / 12:
        flags.append("warn_capacity_very_dense_vs_area")
    if area_m2 > 0 and cap > area_m2 / 10:
        flags.append("reject_capacity_implausible_vs_area")
    if area_m2 < 20:
        flags.append("warn_tiny_geometry")
    return flags


def harvest_real_dataset(
    out_dir: Path,
    *,
    bbox: str,
    country: str = "FR",
    half_side_m: float = 55.0,
    chip_pixels: int = 512,
    delay_s: float = 0.75,
    max_features: int = 5000,
    wms_base: Optional[str] = None,
    wms_layer: Optional[str] = None,
    client: Optional[httpx.Client] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Interroge Overpass (bbox), filtre les parkings avec capacity, télécharge les puces BD ORTHO.
    Écrit `manifest.csv`, `dataset_manifest.csv` (copie), `dataset_report.md`.
    Retourne le chemin du manifest principal.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    min_lon, min_lat, max_lon, max_lat = parse_bbox_string(bbox)
    own = client is None
    if own:
        client = httpx.Client(timeout=120.0, follow_redirects=True)

    rows_out: List[Dict[str, Any]] = []
    stats = {"raw": 0, "kept": 0, "rejected": 0}

    try:
        elems, raw = query_parkings_bbox_capacity(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            client=client,
            delay_s=0.0,
            cache_dir=cache_dir,
        )
        stats["raw"] = len(elems)
        seen: set[tuple[str, int]] = set()

        for el in elems:
            if len(rows_out) >= max_features:
                break
            key = (el.osm_type, el.osm_id)
            if key in seen:
                continue
            seen.add(key)
            cap = _capacity_int(el.tags)
            if cap is None:
                stats["rejected"] += 1
                continue
            cc = _centroid_lonlat(el)
            if cc is None:
                stats["rejected"] += 1
                continue
            lon, lat = cc
            area_m2 = _area_m2(el)
            qf = _quality_flags(cap, area_m2)
            if "reject_capacity_implausible_vs_area" in qf or "reject_capacity_non_positive" in qf:
                stats["rejected"] += 1
                continue

            time.sleep(delay_s)
            chip = fetch_ortho_chip(
                lon,
                lat,
                half_side_m=half_side_m,
                width_px=chip_pixels,
                height_px=chip_pixels,
                wms_base=wms_base or DEFAULT_WMS_BASE,
                layer=wms_layer or DEFAULT_WMS_LAYER,
                client=client,
                cache_dir=cache_dir,
                analysis_radius_m=float(half_side_m),
            )
            fname = f"{len(rows_out):06d}.png"
            rel = f"images/{fname}"
            chip.image.save(img_dir / fname, format="PNG")

            row: Dict[str, Any] = {
                "image_relative": rel,
                "lon": lon,
                "lat": lat,
                "capacity": cap,
                "source": "osm_overpass_bbox",
                "osm_type": el.osm_type,
                "osm_id": el.osm_id,
                "area_m2": round(area_m2, 2),
                "tags_json": json.dumps(el.tags, ensure_ascii=False),
                "half_side_m": half_side_m,
                "chip_pixels": chip_pixels,
                "country": country,
                "quality_flags": "|".join(qf) if qf else "",
                "geometry_ring_json": json.dumps(el.geometry_lonlat[:200]),
            }
            rows_out.append(row)
            stats["kept"] += 1
    except OverpassError as e:
        (out_dir / "dataset_report.md").write_text(f"# Échec moisson\n\n{e}\n", encoding="utf-8")
        raise
    finally:
        if own and client is not None:
            client.close()

    man = out_dir / "manifest.csv"
    pd.DataFrame(rows_out).to_csv(man, index=False)
    shutil.copyfile(man, out_dir / "dataset_manifest.csv")

    report = [
        "# Rapport dataset réel (OSM capacity + orthophoto)",
        "",
        f"- Pays demandé : **{country}** (filtre géographique = bbox utilisateur).",
        f"- Bbox : `{bbox}`",
        f"- Éléments bruts Overpass : {stats['raw']}",
        f"- Lignes retenues : {stats['kept']}",
        f"- Rejetées : {stats['rejected']}",
        "",
        "## Limites",
        "",
        "- Les tags `capacity` OSM peuvent être erronés ou obsolètes.",
        "- Pas de garantie de couverture homogène sur tout le territoire.",
        "- Respecter les CGU Overpass et IGN / Géoplateforme (débit, usage).",
        "",
    ]
    (out_dir / "dataset_report.md").write_text("\n".join(report), encoding="utf-8")
    return man
