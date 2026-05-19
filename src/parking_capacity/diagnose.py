"""Diagnostic adresse : artefacts fichiers pour contrôle terrain."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from PIL import Image, ImageDraw

from parking_capacity.pipeline import process_address, row_to_json_serializable


def _tiny_png_bytes() -> bytes:
    im = Image.new("RGB", (64, 64), color=(120, 140, 100))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def create_diagnose_mock_transport() -> httpx.MockTransport:
    """HTTP entièrement mocké (BAN, APICarto, Overpass, WMS) pour tests sans réseau."""

    ban_json = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [2.3, 48.85]},
                "properties": {"label": "38 rue du Moulin à Vent (mock)", "score": 0.92},
            }
        ]
    }
    parcel_json = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2.29, 48.84], [2.31, 48.84], [2.31, 48.86], [2.29, 48.86], [2.29, 48.84]]],
                },
                "properties": {"id": "mock_parcel_1"},
            }
        ],
    }
    overpass_json = {
        "elements": [
            {
                "type": "way",
                "id": 99,
                "tags": {"amenity": "parking"},
                "geometry": [
                    {"lat": 48.849, "lon": 2.300},
                    {"lat": 48.849, "lon": 2.301},
                    {"lat": 48.851, "lon": 2.301},
                    {"lat": 48.851, "lon": 2.300},
                    {"lat": 48.849, "lon": 2.300},
                ],
            }
        ]
    }
    png = _tiny_png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api-adresse.data.gouv.fr" in url:
            return httpx.Response(200, json=ban_json)
        if "apicarto.ign.fr" in url and "parcelle" in url:
            return httpx.Response(200, json=parcel_json)
        if "overpass-api.de" in url or "interpreter" in url:
            return httpx.Response(200, json=overpass_json)
        if "data.geopf.fr" in url or "wms" in url.lower():
            return httpx.Response(200, content=png, headers={"Content-Type": "image/png"})
        return httpx.Response(404, text="not mocked")

    return httpx.MockTransport(handler)


def _sources_payload(row: Any) -> Dict[str, Any]:
    return {
        "ban": {
            "label": row.ban_label,
            "score": row.ban_score,
            "lon": row.lon,
            "lat": row.lat,
        },
        "cadastre": {"parcelle_ids": row.parcelle_ids},
        "osm": {
            "nearby_parkings_count": row.nearby_osm_parkings_count,
            "capacity_osm_parcelle": row.capacity_osm_parcelle,
            "capacity_osm_buffer": row.capacity_osm_buffer,
            "capacity_osm_parcelle_tagged": row.capacity_osm_parcelle_tagged,
            "capacity_osm_buffer_tagged": row.capacity_osm_buffer_tagged,
            "osm_parking_space_count": row.osm_parking_space_count,
            "n_parkings_parcelle": row.n_parkings_parcelle,
            "n_parkings_buffer_only": row.n_parkings_buffer_only,
        },
        "pipeline": {
            "source_priority_used": row.source_priority_used,
            "capacity_provenance": row.capacity_provenance,
            "sources_used": row.sources_used,
        },
    }


def _write_debug_map_html(path: Path, lat: float, lon: float, radius_m: float) -> None:
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Diagnostic rayon {radius_m} m</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>#map {{ height: 520px; }}</style></head><body>
<p>Centre approximatif (BAN) : {lat:.6f}, {lon:.6f} — rayon analyse : {radius_m} m</p>
<div id="map"></div>
<script>
  const map = L.map('map').setView([{lat}, {lon}], 17);
  L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '© OSM' }}).addTo(map);
  L.marker([{lat}, {lon}]).addTo(map).bindPopup('Point géocodé');
  L.circle([{lat}, {lon}], {{ radius: {radius_m}, color: '#c00', fillColor: '#f88', fillOpacity: 0.15 }}).addTo(map);
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_debug_overlay(
    chip_png: Path,
    out_png: Path,
    *,
    radius_m: float,
    half_side_m: float,
) -> None:
    im = Image.open(chip_png).convert("RGBA")
    w, h = im.size
    cx, cy = w // 2, h // 2
    r_px = int(min(w, h) * 0.5 * (radius_m / max(half_side_m, 1e-6)))
    r_px = max(4, min(r_px, min(w, h) // 2 - 2))
    draw = ImageDraw.Draw(im)
    draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=(220, 40, 40, 255), width=3)
    im.convert("RGB").save(out_png, format="PNG")


def run_diagnose_address(
    address: str,
    out_dir: Path,
    *,
    radius_m: int = 50,
    buffer_m: float = 40.0,
    chip_half_side_m: float = 55.0,
    chip_pixels: int = 512,
    cache_dir: Optional[Path] = None,
    refresh_imagery: bool = False,
    source_priority: str = "hybrid",
    no_vision: bool = False,
    mock: bool = False,
    ml_checkpoint: Optional[Path] = None,
    ml_mode: str = "fallback",
    force_ml: bool = False,
    visual_backend: str = "auto",
    visual_model_specialized_for_parking: bool = False,
    yolo_weights: Optional[Path] = None,
    providers_yaml: Optional[Path] = None,
    vehicle_yolo_weights: Optional[Path] = None,
    auto_download_vehicle_yolo: bool = False,
    auto_download_aerial_yolo: bool = False,
    use_finetuned_french_yolo: bool = False,
    use_dota_finetuned_yolo: bool = False,
    slot_yolo_weights: Optional[Path] = None,
    roboflow_api_key: Optional[str] = None,
    roboflow_model_id: Optional[str] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chip_path = out_dir / "chip.png"

    own_client = True
    client: httpx.Client
    if mock:
        transport = create_diagnose_mock_transport()
        client = httpx.Client(transport=transport, timeout=30.0)
        no_vision_eff = True
    else:
        client = httpx.Client(timeout=120.0, follow_redirects=True)
        no_vision_eff = no_vision

    fetch_half = max(float(chip_half_side_m), float(radius_m))

    try:
        row = process_address(
            address,
            client=client,
            search_radius_m=radius_m,
            point_buffer_m=buffer_m,
            use_vision=not no_vision_eff,
            chip_half_side_m=chip_half_side_m,
            chip_pixels=chip_pixels,
            cache_dir=cache_dir,
            save_chip_path=chip_path,
            refresh_imagery=refresh_imagery,
            source_priority=source_priority,
            ml_checkpoint=ml_checkpoint,
            ml_mode=ml_mode,
            force_ml=force_ml,
            min_intersection_m2=1.0 if mock else 25.0,
            visual_backend=visual_backend,
            visual_model_specialized_for_parking=visual_model_specialized_for_parking,
            yolo_weights=yolo_weights,
            providers_yaml=providers_yaml,
            vehicle_yolo_weights=vehicle_yolo_weights,
            auto_download_vehicle_yolo=auto_download_vehicle_yolo,
            auto_download_aerial_yolo=auto_download_aerial_yolo,
            use_finetuned_french_yolo=use_finetuned_french_yolo,
            use_dota_finetuned_yolo=use_dota_finetuned_yolo,
            slot_yolo_weights=slot_yolo_weights,
            roboflow_api_key=roboflow_api_key,
            roboflow_model_id=roboflow_model_id,
        )
    finally:
        if own_client:
            client.close()

    result = row_to_json_serializable(row)
    result["chip_file"] = "chip.png" if chip_path.is_file() else None
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "sources.json").write_text(
        json.dumps(_sources_payload(row), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    warn_text = row.warnings or ""
    (out_dir / "warnings.txt").write_text(warn_text + ("\n" if warn_text and not warn_text.endswith("\n") else ""), encoding="utf-8")

    if row.lat is not None and row.lon is not None:
        _write_debug_map_html(out_dir / "debug_map.html", float(row.lat), float(row.lon), float(radius_m))

    if chip_path.is_file():
        try:
            _write_debug_overlay(
                chip_path,
                out_dir / "debug_overlay.png",
                radius_m=float(radius_m),
                half_side_m=float(fetch_half),
            )
        except Exception as e:  # noqa: BLE001
            (out_dir / "warnings.txt").write_text(
                warn_text + f"\nImpossible debug_overlay.png : {e}\n",
                encoding="utf-8",
            )

        if not mock and row.lat is not None and row.lon is not None:
            cg = row.raw_debug.get("chip_geo") if isinstance(row.raw_debug, dict) else None
            if isinstance(cg, dict) and all(k in cg for k in ("minx", "miny", "maxx", "maxy", "width_px", "height_px")):
                try:
                    import math

                    import numpy as np
                    from PIL import Image as _Img

                    from parking_capacity.gis_fusion import (
                        build_gis_fusion,
                        compute_fusion_area_metrics,
                        write_gis_fusion_debug_pngs,
                    )
                    from parking_capacity.imagery_wms import OrthoChip, chip_m2_per_pixel
                    from parking_capacity.providers_config import load_gis_providers_config
                    from parking_capacity.surface_classification import classify_surfaces

                    img = _Img.open(chip_path).convert("RGB")
                    chip_geo = OrthoChip(
                        image=img,
                        minx=float(cg["minx"]),
                        miny=float(cg["miny"]),
                        maxx=float(cg["maxx"]),
                        maxy=float(cg["maxy"]),
                        width_px=int(cg["width_px"]),
                        height_px=int(cg["height_px"]),
                        layer="diagnose_gis",
                    )
                    gis_cfg = load_gis_providers_config(yaml_path=providers_yaml)
                    c2 = httpx.Client(timeout=120.0, follow_redirects=True)
                    try:
                        fusion = build_gis_fusion(
                            chip_geo,
                            float(row.lat),
                            float(row.lon),
                            radius_m=int(radius_m),
                            cfg=gis_cfg,
                            client=c2,
                            cache_dir=cache_dir,
                            overpass_delay_s=1.0,
                            access_distance_threshold_m=float(gis_cfg.access_distance_threshold_m),
                        )
                        rgb = np.asarray(img, dtype=np.uint8)
                        m_per_px = math.sqrt(max(chip_m2_per_pixel(chip_geo), 1e-9))
                        surf = classify_surfaces(rgb, m_per_px=m_per_px)
                        if (
                            fusion.road_mask_gis_hw is not None
                            and fusion.road_mask_gis_hw.shape == surf.road_mask.shape
                        ):
                            surf.road_mask = surf.road_mask | fusion.road_mask_gis_hw
                        _ex, _u, _f, final_mask = compute_fusion_area_metrics(
                            chip_geo, surf.parking_eligible_mask, surf.road_mask, fusion,
                        )
                        write_gis_fusion_debug_pngs(out_dir, chip_geo, fusion, final_mask)
                    finally:
                        c2.close()
                except Exception as e:  # noqa: BLE001
                    (out_dir / "warnings.txt").write_text(
                        (out_dir / "warnings.txt").read_text(encoding="utf-8")
                        + f"\nPNG fusion GIS : {e}\n",
                        encoding="utf-8",
                    )

        try:
            _write_geometry_debug_pngs(
                chip_path,
                out_dir,
                radius_m=float(radius_m),
                half_side_m=float(fetch_half),
                chip_pixels=int(chip_pixels),
            )
        except Exception as e:  # noqa: BLE001
            (out_dir / "warnings.txt").write_text(
                (out_dir / "warnings.txt").read_text(encoding="utf-8")
                + f"\nImpossible debug_geometry_*.png : {e}\n",
                encoding="utf-8",
            )

    _print_console_summary(row, out_dir)


def _write_geometry_debug_pngs(
    chip_path: Path,
    out_dir: Path,
    *,
    radius_m: float,
    half_side_m: float,
    chip_pixels: int,
) -> None:
    """Rejoue la chaîne géométrique + classification sur ``chip.png`` pour produire les images."""
    import numpy as np
    from PIL import Image as _Image

    from parking_capacity.imagery_wms import OrthoChip
    from parking_capacity.parking_geometry import (
        analyze_parking_geometry,
        render_geometry_debug_overlays,
    )
    from parking_capacity.parking_scenarios import analyze_parking_scenarios

    img = Image.open(chip_path).convert("RGB")
    w, h = img.size
    side_m = max(half_side_m, 1.0) * 2.0
    chip = OrthoChip(
        image=img,
        minx=0.0,
        miny=0.0,
        maxx=side_m,
        maxy=side_m,
        width_px=w,
        height_px=h,
        layer="diagnose_replay",
    )
    analysis = analyze_parking_geometry(chip)
    overlays = render_geometry_debug_overlays(chip, analysis)
    for name, im in overlays.items():
        im.convert("RGB").save(out_dir / f"{name}.png", format="PNG")

    # Classification surface + scénarios
    scenarios = analyze_parking_scenarios(chip)
    surf = scenarios.surface
    rgb = np.asarray(img, dtype=np.uint8)

    def _overlay_mask(mask, color):
        out = rgb.copy()
        if mask is None or mask.size == 0:
            return _Image.fromarray(out)
        m = mask.astype(bool)
        out[m] = (0.5 * out[m] + 0.5 * np.array(color, dtype=np.uint8)).astype(np.uint8)
        return _Image.fromarray(out)

    # surface_classification : pile colorée par type
    cls = rgb.copy()
    cls[surf.vegetation_mask] = (0.4 * cls[surf.vegetation_mask] + 0.6 * np.array([30, 200, 50], dtype=np.uint8)).astype(np.uint8)
    cls[surf.road_mask] = (0.4 * cls[surf.road_mask] + 0.6 * np.array([50, 50, 220], dtype=np.uint8)).astype(np.uint8)
    cls[surf.asphalt_mask & ~surf.road_mask & ~surf.roof_mask] = (0.5 * cls[surf.asphalt_mask & ~surf.road_mask & ~surf.roof_mask] + 0.5 * np.array([200, 200, 80], dtype=np.uint8)).astype(np.uint8)
    cls[surf.roof_mask] = (0.5 * cls[surf.roof_mask] + 0.5 * np.array([220, 50, 50], dtype=np.uint8)).astype(np.uint8)
    _Image.fromarray(cls).save(out_dir / "debug_surface_classification.png", format="PNG")

    _overlay_mask(surf.roof_mask, (220, 50, 50)).save(
        out_dir / "debug_asphalt_vs_roof.png", format="PNG"
    )
    _overlay_mask(surf.road_mask, (50, 50, 220)).save(
        out_dir / "debug_roadside_candidates.png", format="PNG"
    )
    _overlay_mask(surf.parking_eligible_mask, (250, 200, 80)).save(
        out_dir / "debug_unmarked_surface.png", format="PNG"
    )

    # === Couche sémantique : véhicules, bâtiments, accès, overlay synthèse ===
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    vehicles = scenarios.vehicles
    semantic = scenarios.semantic

    # debug_vehicle_detection.png : boîtes voitures
    veh_img = rgb.copy()
    if vehicles and vehicles.vehicles and cv2 is not None:
        for v in vehicles.vehicles:
            # construire un rectangle orienté
            cx, cy = int(v.cx), int(v.cy)
            wpx, hpx = max(int(v.width_px), 2), max(int(v.height_px), 2)
            rect = ((cx, cy), (wpx, hpx), v.angle_deg)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(veh_img, [box], 0, (0, 200, 240), 2)
        for c in vehicles.clusters:
            xs = [int(m.cx) for m in c.members]
            ys = [int(m.cy) for m in c.members]
            if len(xs) >= 2:
                cv2.line(veh_img, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 0), 2)
    _Image.fromarray(veh_img).save(out_dir / "debug_vehicle_detection.png", format="PNG")

    # debug_building_mask.png
    if semantic is not None:
        _overlay_mask(semantic.building_mask, (220, 30, 30)).save(
            out_dir / "debug_building_mask.png", format="PNG"
        )

    # debug_access_paths.png : intersection eligible/road dilatée
    if semantic is not None and cv2 is not None and surf.road_mask.sum() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eligible_dilated = cv2.dilate(
            (surf.parking_eligible_mask & ~semantic.building_mask).astype(np.uint8),
            k, iterations=2,
        ).astype(bool)
        contact = eligible_dilated & surf.road_mask
        acc_img = rgb.copy()
        acc_img[surf.road_mask] = (0.5 * acc_img[surf.road_mask] + 0.5 * np.array([50, 50, 220], dtype=np.uint8)).astype(np.uint8)
        acc_img[contact] = (0.4 * acc_img[contact] + 0.6 * np.array([255, 220, 50], dtype=np.uint8)).astype(np.uint8)
        _Image.fromarray(acc_img).save(out_dir / "debug_access_paths.png", format="PNG")
    else:
        _Image.fromarray(rgb).save(out_dir / "debug_access_paths.png", format="PNG")

    # debug_semantic_overlay.png : synthèse
    sem_img = rgb.copy()
    if semantic is not None:
        sem_img[semantic.building_mask] = (0.6 * sem_img[semantic.building_mask] + 0.4 * np.array([220, 30, 30], dtype=np.uint8)).astype(np.uint8)
        useful = surf.parking_eligible_mask & ~semantic.building_mask
        sem_img[useful] = (0.5 * sem_img[useful] + 0.5 * np.array([60, 200, 80], dtype=np.uint8)).astype(np.uint8)
    sem_img[surf.road_mask] = (0.6 * sem_img[surf.road_mask] + 0.4 * np.array([50, 50, 220], dtype=np.uint8)).astype(np.uint8)
    if vehicles and cv2 is not None:
        for v in vehicles.vehicles:
            cv2.circle(sem_img, (int(v.cx), int(v.cy)), max(3, int(v.width_px / 2)), (255, 200, 0), 2)
    _Image.fromarray(sem_img).save(out_dir / "debug_semantic_overlay.png", format="PNG")
    # Sérialise un résumé géométrique + scénarios pour debug rapide
    comp_payload = {}
    for mode, est in scenarios.components.items():
        if est is None:
            comp_payload[mode] = None
            continue
        comp_payload[mode] = {
            "capacity_estimate": est.capacity_estimate,
            "capacity_min": est.capacity_min,
            "capacity_max": est.capacity_max,
            "confidence": est.confidence,
            "notes": est.notes,
            "extras": est.extras,
        }
    geom_summary = {
        "primary_mode": scenarios.primary_mode,
        "primary_estimate": comp_payload.get(scenarios.primary_mode) if scenarios.primary_estimate else None,
        "components": comp_payload,
        "surface": {
            "asphalt_likelihood": surf.asphalt_likelihood,
            "roof_likelihood": surf.roof_likelihood,
            "road_likelihood": surf.road_likelihood,
            "vegetation_likelihood": surf.vegetation_likelihood,
            "shadow_likelihood": surf.shadow_likelihood,
            "building_edge_likelihood": surf.building_edge_likelihood,
        },
        "geometry": {
            "geometry_confidence": analysis.geometry_confidence,
            "parking_rows_detected": analysis.parking_rows_detected,
            "geometric_capacity_estimate": analysis.geometric_capacity_estimate,
            "geometric_capacity_min": analysis.geometric_capacity_min,
            "geometric_capacity_max": analysis.geometric_capacity_max,
            "repeated_pattern_score": analysis.repeated_pattern_score,
            "estimated_row_orientation_deg": analysis.estimated_row_orientation_deg,
            "estimated_slot_width_m": analysis.estimated_slot_width_m,
            "debug": {
                "meters_per_pixel": analysis.debug.meters_per_pixel,
                "raw_line_count": analysis.debug.raw_line_count,
                "filtered_line_count": analysis.debug.filtered_line_count,
                "usable_line_count": analysis.debug.usable_line_count,
                "dominant_orientations_deg": analysis.debug.dominant_orientations_deg,
                "row_candidates": analysis.debug.row_candidates,
                "accepted_rows": analysis.debug.accepted_rows,
                "rejected_rows": analysis.debug.rejected_rows,
                "rejection_reasons": analysis.debug.rejection_reasons,
                "capacity_formula_used": analysis.debug.capacity_formula_used,
                "row_lengths_m": analysis.debug.row_lengths_m,
                "chain_failure": analysis.debug.chain_failure,
            },
        },
    }
    (out_dir / "geometry_debug.json").write_text(
        json.dumps(geom_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _print_console_summary(row: Any, out_dir: Path) -> None:
    lines = [
        "--- Diagnostic adresse ---",
        f"Dossier : {out_dir.resolve()}",
        f"Adresse : {row.input_address}",
        f"BAN : {row.ban_label} (score {row.ban_score})",
        f"Priorité source : {row.source_priority_used}",
        f"Provenance capacité : {row.capacity_provenance}",
        f"Capacité estimée : {row.estimated_capacity}",
        f"Méthode : {row.method_used} (confiance : {row.primary_confidence})",
        f"Preuve visuelle : {row.visual_evidence_level} | image : {row.image_used} | conf. image : {row.image_confidence}",
        f"Géométrie : conf={row.geometry_confidence} rangées={row.parking_rows_detected} "
        f"cap={row.geometric_capacity_estimate} ({row.geometric_capacity_min}-{row.geometric_capacity_max})",
        f"Aire parking détectée (m²) : {row.parking_area_detected_m2}",
        f"Places détectées (comptage explicite) : {row.parking_spaces_detected_count}",
        f"Surface seule (hint, non primary) : {row.surface_only_capacity_hint}",
        f"Raison repli : {row.fallback_reason}",
        f"Erreur : {row.error}",
        "Fichiers : chip.png, result.json, sources.json, warnings.txt, debug_map.html, debug_overlay.png, "
        "debug_edges.png, debug_hough_lines.png, debug_parking_rows.png, debug_geometry_overlay.png, "
        "debug_surface_classification.png, debug_asphalt_vs_roof.png, debug_roadside_candidates.png, "
        "debug_unmarked_surface.png, debug_bdtopo_buildings.png, debug_bdtopo_roads.png, debug_osm_roads.png, "
        "debug_gis_fusion.png, debug_final_parking_candidate.png, geometry_debug.json",
    ]
    print("\n".join(lines))
