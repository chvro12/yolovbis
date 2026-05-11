"""Commande ``check-gis-providers`` : tests réseau + rapports."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image, ImageDraw

from parking_capacity.ign_geoplateforme import wfs_get_feature_geojson, wfs_ping
from parking_capacity.mapillary_provider import mapillary_ping
from parking_capacity.osm_transport import query_transport_around
from parking_capacity.providers_config import GisProvidersConfig, load_gis_providers_config

logger = logging.getLogger(__name__)


def run_gis_providers_check(
    lat: float,
    lon: float,
    *,
    radius_m: int,
    out_dir: Path,
    cfg: Optional[GisProvidersConfig] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or load_gis_providers_config()

    report: Dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "radius_m": radius_m,
        "ign_wfs_reachable": False,
        "ign_bdtopo_buildings_sample": False,
        "ign_bdtopo_roads_sample": False,
        "osm_highways_found": False,
        "osm_highway_way_count": 0,
        "osm_named_highways_sample": [],
        "microsoft_path_present": bool(cfg.microsoft_buildings_path),
        "microsoft_bbox_features": 0,
        "mapillary_token_present": bool(cfg.mapillary_token),
        "mapillary_ping_ok": False,
        "mapillary_images_near_point": 0,
        "warnings": [],
    }
    warnings: List[str] = []

    if not cfg.microsoft_buildings_path:
        warnings.append("MICROSOFT_BUILDINGS_PATH non défini — pas d'empreintes Microsoft locales.")
    if not cfg.mapillary_token:
        warnings.append("MAPILLARY_ACCESS_TOKEN non défini — pas d'images Mapillary.")

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        try:
            report["ign_wfs_reachable"] = wfs_ping(client, cfg.ign_wfs_url, cfg.ign_bdtopo_buildings_typename)
            if report["ign_wfs_reachable"]:
                fc = wfs_get_feature_geojson(
                    client,
                    cfg.ign_wfs_url,
                    cfg.ign_bdtopo_buildings_typename,
                    (lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002),
                    max_features=5,
                    cache_dir=cache_dir,
                )
                feats = fc.get("features") or []
                report["ign_bdtopo_buildings_sample"] = len(feats) > 0
                fc2 = wfs_get_feature_geojson(
                    client,
                    cfg.ign_wfs_url,
                    cfg.ign_bdtopo_roads_typename,
                    (lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002),
                    max_features=5,
                    cache_dir=cache_dir,
                )
                report["ign_bdtopo_roads_sample"] = len(fc2.get("features") or []) > 0
        except Exception as e:  # noqa: BLE001
            warnings.append(f"IGN WFS : {e}")

        try:
            tr = query_transport_around(
                lat,
                lon,
                radius_m=radius_m,
                base_url=cfg.overpass_url,
                client=client,
                delay_s=0.0,
                cache_dir=cache_dir,
            )
            report["osm_highway_way_count"] = tr.summary.n_highway_ways
            report["osm_highways_found"] = tr.summary.n_highway_ways > 0
            report["osm_named_highways_sample"] = tr.summary.named_highways_sample
        except Exception as e:  # noqa: BLE001
            warnings.append(f"OSM transport : {e}")

        if cfg.mapillary_token:
            try:
                report["mapillary_ping_ok"] = mapillary_ping(client, cfg.mapillary_token)
                from parking_capacity.mapillary_provider import mapillary_images_in_bbox

                bbox_small = (lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002)
                imgs = mapillary_images_in_bbox(
                    client,
                    bbox_small,
                    access_token=cfg.mapillary_token,
                    graph_base=cfg.mapillary_graph_url,
                    limit=12,
                )
                report["mapillary_images_near_point"] = len(imgs)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Mapillary : {e}")

        if cfg.microsoft_buildings_path and cfg.microsoft_buildings_path.is_file():
            try:
                from parking_capacity.microsoft_buildings import load_building_geometries_for_bbox

                bbox_ms = (lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002)
                geoms = load_building_geometries_for_bbox(cfg.microsoft_buildings_path, bbox_ms)
                report["microsoft_bbox_features"] = len(geoms)
                if not geoms:
                    warnings.append(
                        "MICROSOFT_BUILDINGS_PATH : fichier présent mais aucune empreinte dans la bbox test "
                        "(extrait régional requis si fichier national)."
                    )
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Microsoft buildings bbox : {e}")
        elif os.environ.get("MICROSOFT_BUILDINGS_PATH", "").strip() and not cfg.microsoft_buildings_path:
            warnings.append("MICROSOFT_BUILDINGS_PATH défini mais chemin introuvable ou vide.")

    report["warnings"] = warnings

    json_path = out_dir / "providers_raw.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Rapport fournisseurs GIS",
        "",
        f"- Point : `{lat}`, `{lon}` — rayon **{radius_m} m**",
        "",
        "## Résultats",
        "",
        "| Contrôle | Résultat |",
        "|----------|----------|",
        f"| IGN WFS joignable | {'oui' if report['ign_wfs_reachable'] else 'non'} |",
        f"| BD TOPO bâtiments (échantillon bbox) | {'oui' if report.get('ign_bdtopo_buildings_sample') else 'non'} |",
        f"| BD TOPO routes (échantillon bbox) | {'oui' if report.get('ign_bdtopo_roads_sample') else 'non'} |",
        f"| OSM highways (Overpass transport) | {'oui' if report['osm_highways_found'] else 'non'} ({report['osm_highway_way_count']} ways) |",
        f"| Microsoft buildings path | {'oui' if report['microsoft_path_present'] else 'non'} |",
        f"| Mapillary token | {'oui' if report['mapillary_token_present'] else 'non'} |",
        f"| Mapillary ping | {'oui' if report.get('mapillary_ping_ok') else 'non'} |",
        f"| Mapillary images (bbox point) | {report.get('mapillary_images_near_point', 0)} |",
        f"| Microsoft empreintes bbox test | {report.get('microsoft_bbox_features', 0)} |",
        "",
        "### Exemples de noms de voirie OSM",
        "",
    ]
    names = report.get("osm_named_highways_sample") or []
    if names:
        for n in names:
            md_lines.append(f"- {n}")
    else:
        md_lines.append("- (aucun nom dans l'échantillon)")
    md_lines.extend(["", "## Avertissements", ""])
    if warnings:
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("- (aucun)")
    (out_dir / "providers_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    _write_debug_png(out_dir / "debug_gis_layers.png", report)
    return report


def _write_debug_png(path: Path, report: Dict[str, Any]) -> None:
    im = Image.new("RGB", (720, 520), color=(24, 26, 32))
    draw = ImageDraw.Draw(im)
    y = 12
    for line in [
        "GIS providers check",
        f"osm_highways: {report.get('osm_highways_found')} ({report.get('osm_highway_way_count')} ways)",
        f"ign_wfs: {report.get('ign_wfs_reachable')} bdtopo_bat:{report.get('ign_bdtopo_buildings_sample')}",
        f"microsoft_path: {report.get('microsoft_path_present')} mapillary: {report.get('mapillary_token_present')}",
    ]:
        draw.text((16, y), line, fill=(230, 232, 238))
        y += 28
    im.save(path, format="PNG")
