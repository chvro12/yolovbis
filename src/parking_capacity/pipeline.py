"""Orchestration : adresse → OSM + orthophoto + vision + ML + baseline."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import httpx
import numpy as np

from parking_capacity.baseline import BaselineSignals, choose_baseline_capacity, compare_ml_to_baseline
from parking_capacity.cadastre import fetch_parcelles, merge_parcelles_geometries
from parking_capacity.geocode import GeocodeError, geocode_address
from parking_capacity.geometry import buffer_point_m
from parking_capacity.gis_rasterize import chip_to_bbox4326
from parking_capacity.imagery_wms import fetch_ortho_chip
from parking_capacity.osm_aggregate import (
    classify_parkings,
    count_polygons,
    sum_capacity,
    surface_capacity_range_for_untagged,
)
from parking_capacity.overpass import OverpassError, query_parking_space_count, query_parkings
from parking_capacity.pipeline_resolve import pick_primary_capacity
from parking_capacity.site_classification import collect_osm_amenity_tags, infer_site_type
from parking_capacity.ml.model_meta import should_skip_ml_inference
from parking_capacity.vision_estimate import segment_parking_on_chip
from parking_capacity.visual_evidence import compute_visual_evidence, vision_notes_with_disclaimer

logger = logging.getLogger(__name__)


@dataclass
class RowResult:
    input_address: str
    ban_label: str | None = None
    lon: float | None = None
    lat: float | None = None
    ban_score: float | None = None
    parcelle_ids: str | None = None
    capacity_osm_parcelle: int = 0
    capacity_osm_parcelle_tagged: int = 0
    capacity_osm_buffer: int = 0
    capacity_osm_buffer_tagged: int = 0
    n_parkings_parcelle: int = 0
    n_parkings_buffer_only: int = 0
    primary_capacity: int | None = None
    primary_source: str | None = None
    primary_confidence: str | None = None
    vision_estimated_spaces: int | None = None
    vision_area_m2: float | None = None
    vision_fraction: float | None = None
    vision_device: str | None = None
    ml_estimated_capacity: int | None = None
    ml_estimated_raw: float | None = None
    caveats: str = ""
    error: str | None = None
    raw_debug: dict[str, Any] = field(default_factory=dict)
    radius_m_used: Optional[int] = None
    estimated_capacity: Optional[int] = None
    min_capacity: Optional[int] = None
    max_capacity: Optional[int] = None
    method_used: Optional[str] = None
    sources_used: Optional[str] = None
    nearby_osm_parkings_count: int = 0
    chip_path: Optional[str] = None
    warnings: Optional[str] = None
    baseline_estimate: Optional[int] = None
    baseline_method: Optional[str] = None
    ml_vs_baseline_note: Optional[str] = None
    osm_parking_space_count: int = 0
    area_ratio_mid: Optional[int] = None
    area_ratio_min: Optional[int] = None
    area_ratio_max: Optional[int] = None
    area_total_m2: float = 0.0
    source_priority_used: Optional[str] = None
    capacity_provenance: Optional[str] = None
    visual_evidence_level: Optional[str] = None
    image_used: bool = False
    image_confidence: Optional[str] = None
    parking_area_detected_m2: Optional[float] = None
    parking_spaces_detected_count: Optional[int] = None
    fallback_reason: Optional[str] = None
    visual_model_type: str = "segformer_generic"
    visual_model_specialized_for_parking: bool = False
    visual_backend_used: str = "auto"
    parking_rows_detected: int = 0
    estimated_row_orientation_deg: float = 0.0
    estimated_slot_width_m: float = 0.0
    estimated_slot_length_m: float = 0.0
    repeated_pattern_score: float = 0.0
    geometric_capacity_estimate: Optional[int] = None
    geometric_capacity_min: Optional[int] = None
    geometric_capacity_max: Optional[int] = None
    geometry_confidence: str = "none"
    geometry_debug: dict[str, Any] = field(default_factory=dict)
    surface_only_capacity_hint: Optional[int] = None
    surface_only_area_m2: Optional[float] = None
    visual_specialized_effective: bool = False
    # Multi-scénarios visuels
    parking_visual_mode: str = "unknown"
    asphalt_likelihood: float = 0.0
    roof_likelihood: float = 0.0
    road_likelihood: float = 0.0
    vegetation_likelihood: float = 0.0
    shadow_likelihood: float = 0.0
    building_edge_likelihood: float = 0.0
    unmarked_area_m2: Optional[float] = None
    usable_area_factor: Optional[float] = None
    unmarked_capacity_estimate: Optional[int] = None
    unmarked_capacity_min: Optional[int] = None
    unmarked_capacity_max: Optional[int] = None
    unmarked_confidence: str = "none"
    roadside_length_m: Optional[float] = None
    roadside_capacity_estimate: Optional[int] = None
    roadside_capacity_min: Optional[int] = None
    roadside_capacity_max: Optional[int] = None
    roadside_confidence: str = "none"
    courtyard_area_m2: Optional[float] = None
    courtyard_capacity_estimate: Optional[int] = None
    courtyard_capacity_min: Optional[int] = None
    courtyard_capacity_max: Optional[int] = None
    courtyard_confidence: str = "none"
    visual_capacity_components: dict[str, Any] = field(default_factory=dict)
    # Couche sémantique
    vehicle_count: int = 0
    vehicle_density_score: float = 0.0
    vehicle_alignment_score: float = 0.0
    vehicle_detection_method: str = "none"
    parked_vehicle_clusters: int = 0
    building_area_m2: Optional[float] = None
    parking_outside_buildings_ratio: Optional[float] = None
    vehicle_access_score: Optional[float] = None
    road_connection_detected: bool = False
    observed_vehicle_floor: int = 0
    plausible_capacity_ceiling: Optional[int] = None
    parking_usability_score: Optional[float] = None
    semantic_confidence: str = "none"
    semantic_scores: dict[str, Any] = field(default_factory=dict)
    overall_reliability_score: Optional[float] = None
    geometric_reliability: Optional[float] = None
    osm_reliability: Optional[float] = None
    visual_reliability: Optional[float] = None
    ml_reliability: Optional[float] = None
    range_quality_score: Optional[float] = None
    label_source: Optional[str] = None
    refuse_prediction: bool = False
    refuse_prediction_reason: Optional[str] = None
    yolo_detections_count: int = 0
    gis_report: Optional[dict[str, Any]] = None
    site_type: str = "unknown"
    supporting_evidence: Optional[dict[str, Any]] = None
    parking_capacity_estimation: Optional[dict[str, Any]] = None
    vehicles_payload: Optional[dict[str, Any]] = None

    def to_flat_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw_debug", None)
        return d


def process_address(
    address: str,
    *,
    client: httpx.Client | None = None,
    overpass_delay_s: float = 1.0,
    search_radius_m: int = 50,
    point_buffer_m: float = 40.0,
    min_intersection_m2: float = 25.0,
    use_vision: bool = True,
    vision_device: str | None = None,
    chip_half_side_m: float = 55.0,
    chip_pixels: int = 512,
    wms_base: str | None = None,
    wms_layer: str | None = None,
    m2_per_space: float = 26.0,
    vision_compare: bool = False,
    include_raw: bool = False,
    ml_checkpoint: Optional[Union[Path, str]] = None,
    ml_mode: str = "fallback",
    ml_device: str | None = None,
    cache_dir: Optional[Path] = None,
    save_chip_path: Optional[Path] = None,
    aerial_first: bool = True,
    source_priority: str = "hybrid",
    refresh_imagery: bool = False,
    force_ml: bool = False,
    visual_backend: str = "auto",
    visual_model_specialized_for_parking: bool = False,
    yolo_weights: Optional[Path] = None,
    providers_yaml: Optional[Path] = None,
) -> RowResult:
    """
    ``source_priority`` : ``hybrid`` (défaut, OSM fiable sinon image), ``aerial`` (image d'abord),
    ``osm`` (OSM d'abord).

    ``visual_backend`` : ``auto`` (géométrie + SegFormer indice), ``geometry_only``, ``segformer_generic``, ``none``, ``yolo_parking``, …
    ``visual_model_specialized_for_parking`` : si vrai, SegFormer peut contribuer aux places comme avant.
    """
    from parking_capacity.vision.backends import normalize_visual_backend, runs_geometry, runs_segformer, runs_yolo

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=120.0, follow_redirects=True)

    res = RowResult(input_address=address)
    caveats: list[str] = []
    res.radius_m_used = int(search_radius_m)

    eff_priority = (source_priority or "hybrid").strip().lower()
    if eff_priority not in ("aerial", "osm", "hybrid"):
        caveats.append(f"source_priority inconnu {source_priority!r}, utilisation de hybrid.")
        eff_priority = "hybrid"
    if not aerial_first and eff_priority == "hybrid":
        eff_priority = "osm"
    res.source_priority_used = eff_priority

    vb = normalize_visual_backend(visual_backend)
    res.visual_backend_used = vb
    res.visual_model_specialized_for_parking = bool(visual_model_specialized_for_parking)
    if vb == "yolo_parking" or (yolo_weights is not None and runs_yolo(vb)):
        res.visual_model_type = "yolo_parking"
    elif vb in ("groundingdino_sam", "future_custom"):
        res.visual_model_type = vb
    elif runs_segformer(vb):
        res.visual_model_type = "segformer_generic"
    elif runs_geometry(vb):
        res.visual_model_type = "geometry_heuristic"
    else:
        res.visual_model_type = "none"

    try:
        g = geocode_address(address, client=client)
        res.ban_label = g.label
        res.lon, res.lat = g.lon, g.lat
        res.ban_score = g.score
        if g.score < 0.5:
            caveats.append("Score BAN faible : vérifier la géolocalisation.")

        parcel_hits = fetch_parcelles(g.lon, g.lat, client=client)
        parcel_union = merge_parcelles_geometries(parcel_hits)
        if parcel_union is None:
            caveats.append("Aucune parcelle cadastrale renvoyée par APICarto pour ce point.")
        res.parcelle_ids = "|".join([h.id_parcelle or "" for h in parcel_hits if h.id_parcelle]) or None

        pt_buf = buffer_point_m(g.lon, g.lat, point_buffer_m)

        try:
            elems, raw_osm = query_parkings(
                g.lat,
                g.lon,
                radius_m=search_radius_m,
                client=client,
                delay_s=overpass_delay_s,
                cache_dir=cache_dir,
            )
        except OverpassError as e:
            res.error = f"overpass: {e}"
            res.caveats = "; ".join(caveats + ["Overpass indisponible ou erreur."])
            return res

        res.nearby_osm_parkings_count = len(elems)

        classified = classify_parkings(
            elems,
            parcel_union,
            pt_buf,
            min_intersection_m2=min_intersection_m2,
        )

        hint_parts = [g.label or ""]
        for c in classified:
            tg = c.element.tags
            for key in ("name", "operator", "description"):
                v = tg.get(key)
                if v:
                    hint_parts.append(str(v))
        res.site_type = infer_site_type(
            " | ".join(hint_parts),
            osm_amenity_tags=collect_osm_amenity_tags(classified),
        )

        cap_p, tagged_p = sum_capacity(classified, on_parcel=True)
        cap_b, tagged_b = sum_capacity(classified, on_parcel=False)

        res.capacity_osm_parcelle = cap_p
        res.capacity_osm_parcelle_tagged = tagged_p
        res.capacity_osm_buffer = cap_b
        res.capacity_osm_buffer_tagged = tagged_b
        res.n_parkings_parcelle = count_polygons(classified, on_parcel=True)
        res.n_parkings_buffer_only = count_polygons(classified, on_parcel=False)

        if tagged_p == 0 and res.n_parkings_parcelle > 0:
            caveats.append("Parkings sur parcelle sans tag capacity OSM.")

        scope_rows = [r for r in classified if r.on_parcel or r.in_buffer_only]
        mid_a, mn_a, mx_a, ta = surface_capacity_range_for_untagged(scope_rows, on_parcel=None)
        res.area_ratio_mid = mid_a if mid_a > 0 else None
        res.area_ratio_min = mn_a if mn_a > 0 else None
        res.area_ratio_max = mx_a if mx_a > 0 else None
        res.area_total_m2 = ta

        try:
            ps_n, _ps_raw = query_parking_space_count(
                g.lat,
                g.lon,
                radius_m=search_radius_m,
                client=client,
                delay_s=overpass_delay_s,
                cache_dir=cache_dir,
            )
            res.osm_parking_space_count = ps_n
        except OverpassError:
            caveats.append("Comptage parking_space OSM indisponible.")

        has_osm_capacity = (cap_p > 0) or (cap_b > 0)

        vision_est = None
        ml_int: int | None = None
        ml_raw: float | None = None

        ml_ckpt: Path | None = None
        if ml_checkpoint:
            p = Path(ml_checkpoint)
            if p.is_file():
                ml_ckpt = p
            else:
                caveats.append(f"Checkpoint ML introuvable : {p}")

        ml_skip, _ml_meta_side, ml_skip_note = should_skip_ml_inference(ml_ckpt, force_ml=force_ml)
        if ml_skip_note:
            caveats.append(ml_skip_note)

        ml_mode_l = (ml_mode or "fallback").strip().lower()
        if ml_ckpt is None:
            ml_mode_l = "off"
        elif ml_mode_l not in ("aux", "fallback", "before_vision"):
            caveats.append(f"Mode ML inconnu {ml_mode!r} : utilisation de fallback.")
            ml_mode_l = "fallback"

        use_ml = ml_ckpt is not None and ml_mode_l != "off"
        need_chip = (
            use_ml
            or use_vision
            or eff_priority in ("aerial", "hybrid")
            or runs_geometry(vb)
            or (runs_yolo(vb) and yolo_weights is not None)
        )

        chip = None
        if need_chip:
            try:
                from parking_capacity.imagery_wms import DEFAULT_WMS_BASE, DEFAULT_WMS_LAYER
                from parking_capacity.ml.infer import read_inference_config

                fetch_half = max(float(chip_half_side_m), float(search_radius_m))
                fetch_px = chip_pixels
                if use_ml and ml_ckpt is not None:
                    cfg = read_inference_config(ml_ckpt)
                    fetch_half = max(fetch_half, float(cfg.get("half_side_m", chip_half_side_m)))
                    fetch_px = max(int(cfg.get("chip_pixels", chip_pixels)), chip_pixels)
                chip = fetch_ortho_chip(
                    g.lon,
                    g.lat,
                    half_side_m=fetch_half,
                    width_px=fetch_px,
                    height_px=fetch_px,
                    wms_base=wms_base or DEFAULT_WMS_BASE,
                    layer=wms_layer or DEFAULT_WMS_LAYER,
                    client=client,
                    cache_dir=cache_dir,
                    refresh_imagery=refresh_imagery,
                    analysis_radius_m=float(search_radius_m),
                )
                if save_chip_path is not None:
                    save_chip_path = Path(save_chip_path)
                    save_chip_path.parent.mkdir(parents=True, exist_ok=True)
                    chip.image.save(save_chip_path, format="PNG")
                    res.chip_path = str(save_chip_path.resolve())
            except Exception as e:  # noqa: BLE001
                caveats.append(f"Puce orthophoto (WMS) indisponible : {e!s}")

        vision_seg_out = None
        if use_vision and chip is not None and runs_segformer(vb):
            try:
                vision_seg_out = segment_parking_on_chip(
                    chip,
                    m2_per_space=m2_per_space,
                    use_vision=True,
                    device=vision_device,
                )
                if vision_seg_out:
                    vision_est = vision_seg_out.estimate
                    res.vision_estimated_spaces = vision_est.estimated_spaces
                    res.vision_area_m2 = vision_est.parking_area_m2
                    res.vision_fraction = vision_est.parking_pixel_fraction
                    res.vision_device = vision_est.device
                    caveats.append(vision_notes_with_disclaimer(vision_est))
            except Exception as e:  # noqa: BLE001
                caveats.append(f"Vision (SegFormer) indisponible : {e!s}")

        geo_analysis = None
        scenarios = None
        if chip is not None and runs_geometry(vb):
            res.raw_debug["chip_geo"] = {
                "minx": chip.minx,
                "miny": chip.miny,
                "maxx": chip.maxx,
                "maxy": chip.maxy,
                "width_px": chip.width_px,
                "height_px": chip.height_px,
                "bbox4326": list(chip_to_bbox4326(chip)),
            }
            from parking_capacity.gis_context import fetch_chip_gis_augmentation
            from parking_capacity.parking_scenarios import analyze_parking_scenarios
            from parking_capacity.providers_config import load_gis_providers_config

            seg_mask = None
            if (
                vision_seg_out is not None
                and vision_seg_out.parking_mask_hw is not None
                and vision_seg_out.parking_mask_hw.ndim == 2
                and vision_seg_out.parking_mask_hw.shape[0] == chip.height_px
                and vision_seg_out.parking_mask_hw.shape[1] == chip.width_px
            ):
                seg_mask = vision_seg_out.parking_mask_hw

            gis_cfg = load_gis_providers_config(yaml_path=providers_yaml)
            gis_aug = None
            try:
                gis_aug = fetch_chip_gis_augmentation(
                    chip,
                    g.lat,
                    g.lon,
                    radius_m=int(search_radius_m),
                    cfg=gis_cfg,
                    client=client,
                    cache_dir=cache_dir,
                    overpass_delay_s=overpass_delay_s,
                )
                res.raw_debug["gis_fusion"] = {**gis_aug.to_debug_dict()}
            except Exception as e:  # noqa: BLE001
                caveats.append(f"Fusion GIS : {e!s}")

            scenarios = analyze_parking_scenarios(
                chip,
                segformer_parking_mask=seg_mask,
                has_osm_capacity=has_osm_capacity,
                gis_augmentation=gis_aug,
                max_plausible_slots_cap=gis_cfg.max_plausible_capacity_slots,
                site_type=res.site_type or "unknown",
            )
            geo_analysis = scenarios.geometry

            # Couche sémantique : véhicules + bâtiments + accès + usability
            if scenarios.vehicles is not None:
                vd = scenarios.vehicles
                res.vehicle_count = vd.vehicle_count
                res.vehicle_density_score = vd.vehicle_density_score
                res.vehicle_alignment_score = vd.vehicle_alignment_score
                res.vehicle_detection_method = vd.method
                res.parked_vehicle_clusters = len(vd.clusters)
            if scenarios.semantic is not None:
                sm = scenarios.semantic
                res.building_area_m2 = sm.building_area_m2
                res.parking_outside_buildings_ratio = sm.parking_outside_buildings_ratio
                res.vehicle_access_score = sm.vehicle_access_score
                res.road_connection_detected = sm.road_connection_detected
                res.observed_vehicle_floor = sm.observed_vehicle_floor
                res.plausible_capacity_ceiling = sm.plausible_capacity_ceiling
                res.parking_usability_score = sm.parking_usability_score
                res.semantic_confidence = sm.semantic_confidence
                res.semantic_scores = {
                    "parking_usability_score": sm.parking_usability_score,
                    "semantic_confidence": sm.semantic_confidence,
                    "vehicle_evidence": round(sm.evidence.vehicle_evidence, 3),
                    "vehicle_alignment_evidence": round(sm.evidence.vehicle_alignment_evidence, 3),
                    "building_exclusion_score": round(sm.evidence.building_exclusion_score, 3),
                    "road_access_score": round(sm.evidence.road_access_score, 3),
                    "compactness_score": round(sm.evidence.compactness_score, 3),
                    "separators_score": round(sm.evidence.separators_score, 3),
                    "geometry_score": round(sm.evidence.geometry_score, 3),
                    "osm_score": round(sm.evidence.osm_score, 3),
                    "notes": list(sm.notes),
                    "building_mask_source": sm.building_mask_source,
                    "road_source": sm.road_source,
                    "access_distance_m": sm.access_distance_m,
                    "road_network_score": sm.road_network_score,
                }
            if gis_aug is not None and scenarios.semantic is not None:
                res.gis_report = gis_aug.to_row_report_merged(
                    semantic=scenarios.semantic,
                    usable_parking_area_m2=scenarios.fusion_usable_parking_area_m2,
                    excluded_building_area_m2=scenarios.fusion_excluded_building_area_m2,
                    final_parking_candidate_area_m2=scenarios.fusion_final_candidate_area_m2,
                )

            # Champs géométrie historique (compat)
            if geo_analysis is not None:
                res.parking_rows_detected = geo_analysis.parking_rows_detected
                res.estimated_row_orientation_deg = geo_analysis.estimated_row_orientation_deg
                res.estimated_slot_width_m = geo_analysis.estimated_slot_width_m
                res.estimated_slot_length_m = geo_analysis.estimated_slot_length_m
                res.repeated_pattern_score = geo_analysis.repeated_pattern_score
                res.geometric_capacity_estimate = geo_analysis.geometric_capacity_estimate
                res.geometric_capacity_min = geo_analysis.geometric_capacity_min
                res.geometric_capacity_max = geo_analysis.geometric_capacity_max
                res.geometry_confidence = geo_analysis.geometry_confidence
                res.geometry_debug = {
                    "meters_per_pixel": geo_analysis.debug.meters_per_pixel,
                    "raw_line_count": geo_analysis.debug.raw_line_count,
                    "filtered_line_count": geo_analysis.debug.filtered_line_count,
                    "usable_line_count": geo_analysis.debug.usable_line_count,
                    "dominant_orientations_deg": list(geo_analysis.debug.dominant_orientations_deg),
                    "row_candidates": geo_analysis.debug.row_candidates,
                    "accepted_rows": geo_analysis.debug.accepted_rows,
                    "rejected_rows": geo_analysis.debug.rejected_rows,
                    "rejection_reasons": list(geo_analysis.debug.rejection_reasons),
                    "capacity_formula_used": geo_analysis.debug.capacity_formula_used,
                    "row_lengths_m": list(geo_analysis.debug.row_lengths_m),
                    "chain_failure": geo_analysis.debug.chain_failure,
                }
                if geo_analysis.debug.chain_failure:
                    caveats.append(
                        f"Chaîne géométrique stoppée à : {geo_analysis.debug.chain_failure}."
                    )

            # Surface classification
            surf = scenarios.surface
            res.asphalt_likelihood = round(surf.asphalt_likelihood, 3)
            res.roof_likelihood = round(surf.roof_likelihood, 3)
            res.road_likelihood = round(surf.road_likelihood, 3)
            res.vegetation_likelihood = round(surf.vegetation_likelihood, 3)
            res.shadow_likelihood = round(surf.shadow_likelihood, 3)
            res.building_edge_likelihood = round(surf.building_edge_likelihood, 3)

            # Composants par scénario
            comps_payload: dict[str, Any] = {}
            for mode, est in scenarios.components.items():
                if est is None:
                    comps_payload[mode] = None
                    continue
                comps_payload[mode] = {
                    "capacity_estimate": est.capacity_estimate,
                    "capacity_min": est.capacity_min,
                    "capacity_max": est.capacity_max,
                    "confidence": est.confidence,
                    "notes": est.notes,
                    "extras": est.extras,
                }
            res.visual_capacity_components = comps_payload

            um = scenarios.components.get("unmarked_surface")
            if um:
                res.unmarked_area_m2 = um.extras.get("unmarked_area_m2")
                res.usable_area_factor = um.extras.get("usable_area_factor_hi")
                res.unmarked_capacity_estimate = um.capacity_estimate
                res.unmarked_capacity_min = um.capacity_min
                res.unmarked_capacity_max = um.capacity_max
                res.unmarked_confidence = um.confidence

            rs = scenarios.components.get("roadside_parking")
            if rs:
                res.roadside_length_m = rs.extras.get("roadside_length_m")
                res.roadside_capacity_estimate = rs.capacity_estimate
                res.roadside_capacity_min = rs.capacity_min
                res.roadside_capacity_max = rs.capacity_max
                res.roadside_confidence = rs.confidence

            cy = scenarios.components.get("courtyard_parking")
            if cy:
                res.courtyard_area_m2 = cy.extras.get("courtyard_area_m2")
                res.courtyard_capacity_estimate = cy.capacity_estimate
                res.courtyard_capacity_min = cy.capacity_min
                res.courtyard_capacity_max = cy.capacity_max
                res.courtyard_confidence = cy.confidence

            res.parking_visual_mode = scenarios.primary_mode
            for note in scenarios.notes:
                caveats.append(f"Surface : {note}")

        if chip is not None and runs_yolo(vb) and yolo_weights:
            try:
                from parking_capacity.vision.yolo_parking import infer_yolo_parking

                yr = infer_yolo_parking(chip.image, weights=Path(yolo_weights))
                res.yolo_detections_count = len(yr.detections)
                if yr.error:
                    caveats.append(f"YOLO parking : {yr.error}")
            except Exception as e:  # noqa: BLE001
                caveats.append(f"YOLO parking indisponible : {e!s}")

        if use_ml and chip is not None and ml_ckpt is not None and not ml_skip:
            try:
                from parking_capacity.ml.infer import predict_capacity_from_pil

                ml_raw, ml_info = predict_capacity_from_pil(
                    chip.image,
                    ml_ckpt,
                    device_str=ml_device,
                )
                ml_int = max(0, int(round(ml_raw)))
                res.ml_estimated_raw = ml_raw
                res.ml_estimated_capacity = ml_int
                caveats.append(
                    f"ML régression ({ml_info['architecture']}, {ml_info['device']}) : ~{ml_raw:.1f} places."
                )
            except Exception as e:  # noqa: BLE001
                caveats.append(f"Inférence ML indisponible : {e!s}")

        osm_poly_in_scope = len(classified) > 0
        image_fetched = chip is not None
        # Spécialisation effective : uniquement si backend = YOLO/custom ET poids chargés.
        specialized_weights_loaded = bool(
            res.visual_model_type == "yolo_parking" and yolo_weights is not None
        )
        scenario_estimate = scenarios.primary_estimate if scenarios else None
        scenario_mode = scenarios.primary_mode if scenarios else "unknown"
        ve = compute_visual_evidence(
            vision_est,
            osm_parking_polygon_in_scope=osm_poly_in_scope,
            image_fetched=image_fetched,
            geometry=geo_analysis,
            visual_model_type=res.visual_model_type,
            visual_model_specialized_for_parking=res.visual_model_specialized_for_parking,
            specialized_weights_loaded=specialized_weights_loaded,
            scenario_mode=scenario_mode,
            scenario_estimate=scenario_estimate,
        )
        res.visual_evidence_level = ve.visual_evidence_level
        res.image_used = ve.image_used
        res.image_confidence = ve.image_confidence
        res.parking_area_detected_m2 = ve.parking_area_detected_m2
        res.parking_spaces_detected_count = ve.parking_spaces_detected_count
        res.fallback_reason = ve.fallback_reason
        res.surface_only_capacity_hint = ve.surface_only_capacity_hint
        res.surface_only_area_m2 = ve.parking_area_detected_m2 if ve.surface_only_capacity_hint else None
        res.visual_specialized_effective = bool(ve.specialized)
        if (
            res.visual_model_specialized_for_parking
            and not ve.specialized
            and res.visual_model_type == "segformer_generic"
        ):
            caveats.append(
                "--visual-model-specialized ignoré : SegFormer générique n'est pas un compteur de places ; "
                "la spécialisation requiert un backend yolo_parking / custom avec poids chargés."
            )

        if (
            ve.visual_evidence_level in ("none", "weak")
            and vision_est
            and vision_est.estimated_spaces
            and res.visual_model_specialized_for_parking
        ):
            caveats.append(
                "Confiance visuelle faible : le nombre de places affiché par SegFormer n'est pas fiable "
                "(pas de comptage de marquages)."
            )

        # Marked_slots : confiance scénario (post-sanity + post-sémantique).
        geometry_places: int | None = None
        geom_conf_pick = "none"
        marked_est = scenarios.components.get("marked_slots") if scenarios else None
        if marked_est and marked_est.capacity_estimate is not None:
            geometry_places = int(marked_est.capacity_estimate)
            geom_conf_pick = marked_est.confidence
            # Renforcement par sémantique : si géométrie medium ET preuve sémantique strong,
            # on remonte à strong. Inversement, si sémantique trop basse, on plafonne.
            if (
                scenarios and scenarios.semantic
                and scenarios.semantic.semantic_confidence == "strong"
                and geom_conf_pick == "medium"
            ):
                geom_conf_pick = "strong"
            elif (
                scenarios and scenarios.semantic
                and scenarios.semantic.parking_usability_score < 25.0
                and not has_osm_capacity
            ):
                if geom_conf_pick in ("medium", "strong"):
                    geom_conf_pick = "weak"
        elif geo_analysis and geo_analysis.geometric_capacity_estimate:
            geometry_places = int(geo_analysis.geometric_capacity_estimate)
            geom_conf_pick = geo_analysis.geometry_confidence

        # Multi-scénario : on expose les autres modes (non-marked) au resolver.
        scenario_primary_capacity = None
        scenario_primary_source = None
        scenario_primary_conf = "none"
        if scenarios and scenarios.primary_estimate and scenarios.primary_mode != "marked_slots":
            sp_est = scenarios.primary_estimate
            scenario_primary_capacity = int(sp_est.capacity_estimate) if sp_est.capacity_estimate else None
            scenario_primary_source = f"scenario_{scenarios.primary_mode}"
            scenario_primary_conf = sp_est.confidence

        primary, source, conf, provenance = pick_primary_capacity(
            source_priority=eff_priority,
            has_osm_capacity=has_osm_capacity,
            cap_p=cap_p,
            cap_b=cap_b,
            tagged_p=tagged_p,
            tagged_b=tagged_b,
            ban_score=g.score,
            vision_est=vision_est,
            vision_primary_places=ve.parking_spaces_detected_count,
            geometry_places=geometry_places,
            geometry_confidence=geom_conf_pick,
            ml_int=ml_int,
            ml_mode_l=ml_mode_l,
            mid_a=mid_a,
            mn_a=mn_a,
            mx_a=mx_a,
            ta=ta,
            osm_parking_space_count=res.osm_parking_space_count,
            visual_evidence_level=ve.visual_evidence_level,
            visual_specialized_effective=res.visual_specialized_effective,
            scenario_primary_capacity=scenario_primary_capacity,
            scenario_primary_source=scenario_primary_source,
            scenario_primary_confidence=scenario_primary_conf,
        )
        res.capacity_provenance = provenance

        if vision_compare and has_osm_capacity and vision_est and primary == cap_p:
            caveats.append(
                f"Comparaison : vision SegFormer ~{vision_est.estimated_spaces} pl. vs OSM parcelle {cap_p}."
            )

        if primary is None:
            caveats.append(
                "Aucune capacité déduite : signaux OSM, orthophoto et ML épuisés ou confiance trop faible."
            )

        sig = BaselineSignals(
            osm_capacity_tagged=cap_p if tagged_p > 0 else (cap_b if tagged_b > 0 else 0),
            osm_parking_space_count=res.osm_parking_space_count,
            area_mid_places=mid_a,
            area_min_places=mn_a,
            area_max_places=mx_a,
            area_total_m2=ta,
            vision_places=vision_est.estimated_spaces if vision_est else None,
            ml_places=ml_int,
        )
        b_est, b_method, b_note = choose_baseline_capacity(sig)
        res.baseline_estimate = b_est
        res.baseline_method = b_method
        if b_note:
            caveats.append(f"Baseline : {b_note}")

        note_ml = compare_ml_to_baseline(ml_raw, b_est)
        if note_ml:
            res.ml_vs_baseline_note = note_ml
            caveats.append(note_ml)

        res.primary_capacity = primary
        res.primary_source = source
        res.primary_confidence = conf if primary is not None else "low_confidence"
        res.estimated_capacity = primary
        res.method_used = source
        if osm_poly_in_scope and not has_osm_capacity:
            res.label_source = "osm_pseudo"
        elif source == "ml_regressor" and primary is not None:
            res.label_source = "model"
        elif primary is not None and (
            source in ("parking_geometry", "vision_specialized", "vision_marked_visible")
            or (source and source.startswith("scenario_"))
        ):
            res.label_source = "model"
        elif has_osm_capacity and res.geometric_capacity_estimate:
            res.label_source = "hybrid"
        elif has_osm_capacity:
            res.label_source = "human"
        parts = ["ban", "apicarto", "osm_overpass"]
        if chip is not None:
            parts.append("ign_wms_orthophoto")
        gf = res.raw_debug.get("gis_fusion") if isinstance(res.raw_debug, dict) else None
        if isinstance(gf, dict):
            if gf.get("ign_bdtopo_buildings") or gf.get("bdtopo_buildings_used"):
                parts.append("ign_wfs_bdtopo")
            if gf.get("osm_transport_lines") is not None or gf.get("osm_highway_ways") or gf.get("osm_highways_used"):
                parts.append("osm_transport")
        if vision_est is not None:
            parts.append("segformer_parking")
        if geo_analysis is not None and runs_geometry(vb):
            parts.append("parking_geometry")
        if ml_raw is not None:
            parts.append("ml_regression")
        res.sources_used = "|".join(parts)
        if primary is not None:
            if source in ("osm_parcelle", "osm_buffer", "osm_parking_space_count"):
                res.min_capacity = res.max_capacity = int(primary)
            elif source in ("vision_specialized", "vision_marked_visible"):
                res.min_capacity = max(0, int(primary * 0.7))
                res.max_capacity = int(primary * 1.35) + 1
            elif source == "ml_regressor":
                res.min_capacity = max(0, int(primary * 0.75))
                res.max_capacity = int(primary * 1.25) + 1
            elif source == "parking_geometry":
                if (
                    geo_analysis
                    and geo_analysis.geometric_capacity_min is not None
                    and geo_analysis.geometric_capacity_max is not None
                ):
                    res.min_capacity = int(geo_analysis.geometric_capacity_min)
                    res.max_capacity = int(geo_analysis.geometric_capacity_max)
                else:
                    res.min_capacity = max(0, int(primary * 0.78))
                    res.max_capacity = int(primary * 1.22) + 2
            elif source and source.startswith("scenario_"):
                # Fourchette du scénario si dispo
                mode_name = source[len("scenario_"):]
                comp = res.visual_capacity_components.get(mode_name) if isinstance(res.visual_capacity_components, dict) else None
                if comp and comp.get("capacity_min") is not None and comp.get("capacity_max") is not None:
                    res.min_capacity = int(comp["capacity_min"])
                    res.max_capacity = int(comp["capacity_max"])
                else:
                    res.min_capacity = max(0, int(primary * 0.6))
                    res.max_capacity = int(primary * 1.6) + 2

        # Surface seule (area_ratio) : on calcule un hint mais on ne le promeut JAMAIS en primary.
        if mid_a > 0 and ta >= 80 and res.surface_only_capacity_hint is None:
            res.surface_only_capacity_hint = int(mid_a)
            res.surface_only_area_m2 = float(ta)

        if (
            ve.visual_evidence_level == "none"
            and not has_osm_capacity
            and primary is not None
            and source not in (
                "osm_parcelle",
                "osm_buffer",
                "osm_parking_space_count",
                "parking_geometry",
                "vision_specialized",
                "vision_marked_visible",
            )
            and not (source and source.startswith("scenario_"))
        ):
            caveats.append(
                "Aucune preuve orthophoto exploitable et aucun tag OSM capacity fiable sur la zone : "
                "fourchette élargie, confiance basse ; à vérifier manuellement."
            )
            res.primary_confidence = "low_confidence"
            est = int(primary)
            lo = max(0, int(est * 0.4))
            hi = int(est * 1.8) + 10
            cur_min = res.min_capacity if res.min_capacity is not None else lo
            cur_max = res.max_capacity if res.max_capacity is not None else hi
            res.min_capacity = min(int(cur_min), lo)
            res.max_capacity = max(int(cur_max), hi)

        from parking_capacity.ml.model_meta import load_model_meta
        from parking_capacity.reliability import (
            capacity_range_quality_score,
            geometric_reliability_score,
            ml_reliability_score,
            osm_reliability_score,
            overall_reliability_score,
            visual_reliability_score,
        )

        gr = geometric_reliability_score(geo_analysis)
        os_rel = osm_reliability_score(
            has_tagged_capacity=(tagged_p > 0 or tagged_b > 0),
            tagged_polygon_count=res.n_parkings_parcelle,
            ban_score=float(g.score or 0.0),
        )
        seg_ran = bool(vision_est is not None or (use_vision and chip is not None and runs_segformer(vb)))
        vr = visual_reliability_score(
            visual_evidence_level=str(ve.visual_evidence_level or "none"),
            specialized_parking=bool(res.visual_model_specialized_for_parking),
            segformer_ran=seg_ran,
        )
        meta_ml = load_model_meta(ml_ckpt) if ml_ckpt else None
        mr = ml_reliability_score(meta_ml is not None, meta_ml or {})
        rq_val = res.range_quality_score
        if rq_val is None and res.area_ratio_mid and res.min_capacity is not None and res.max_capacity is not None:
            rq_val = capacity_range_quality_score(res.area_ratio_mid, res.min_capacity, res.max_capacity)
            res.range_quality_score = rq_val
        elif rq_val is None:
            rq_val = 0.0
        ov = overall_reliability_score(g=gr, o=os_rel, v=vr, m=mr, range_q=float(rq_val or 0.0))
        res.geometric_reliability = gr
        res.osm_reliability = os_rel
        res.visual_reliability = vr
        res.ml_reliability = mr
        res.overall_reliability_score = ov

        refuse = False
        rreason: list[str] = []
        if ov < 22.0 and primary is None:
            # déjà aucun primary : pas besoin de refuse, mais on garde la trace.
            pass
        if (
            primary is not None
            and source == "parking_geometry"
            and geom_conf_pick == "weak"
            and not has_osm_capacity
        ):
            # Géométrie weak sans OSM : on garde la valeur comme hint, pas comme primary fiable.
            refuse = True
            rreason.append("geometrie_faible_sans_osm_capacity")
            rreason.append("surface_garable_non_isolee_certitude_insuffisante")
        if (
            res.min_capacity is not None
            and res.max_capacity is not None
            and res.min_capacity > 0
            and (res.max_capacity - res.min_capacity) > max(80, int(res.min_capacity) * 4)
        ):
            refuse = True
            rreason.append("fourchette_trop_large")
        if (
            ve.visual_evidence_level == "none"
            and not has_osm_capacity
            and primary is not None
            and source not in (
                "osm_parcelle",
                "osm_buffer",
                "osm_parking_space_count",
                "parking_geometry",
                "vision_specialized",
                "vision_marked_visible",
            )
            and not (source and source.startswith("scenario_"))
        ):
            refuse = True
            rreason.append("aucune_preuve_visuelle_fiable")
        if refuse and primary is not None:
            res.refuse_prediction = True
            res.refuse_prediction_reason = "; ".join(rreason)
            res.estimated_capacity = None
            res.primary_confidence = "low_confidence"
            caveats.append("Prédiction refusée : " + res.refuse_prediction_reason)

        efc = "none"
        if res.vehicle_count > 0:
            if res.vehicle_alignment_score >= 0.55:
                efc = "high"
            elif res.vehicle_alignment_score >= 0.28:
                efc = "medium"
            else:
                efc = "low"
        res.supporting_evidence = {
            "vehicle_presence": {
                "count": res.vehicle_count,
                "method": res.vehicle_detection_method,
                "alignment_score": round(res.vehicle_alignment_score, 4),
                "clusters": res.parked_vehicle_clusters,
            },
        }
        res.vehicles_payload = {
            "count": res.vehicle_count,
            "confidence": efc,
            "used_as_supporting_evidence": res.vehicle_count > 0,
            "effect_on_capacity": "none",
            "effect_on_confidence": efc,
        }
        hyp: Optional[str] = None
        if scenarios and scenarios.components.get("unmarked_surface"):
            um0 = scenarios.components["unmarked_surface"]
            if um0 and isinstance(um0.extras, dict):
                m2r = um0.extras.get("m2_per_space_range")
                if isinstance(m2r, (list, tuple)) and len(m2r) >= 2:
                    hyp = f"{float(m2r[0]):.0f}–{float(m2r[1]):.0f} m²/place (surface non marquée, site_type={res.site_type})"
        if hyp is None:
            hyp = f"ratios calibrés sur site_type={res.site_type} (voir composants scénarios)"
        excl_parts: list[str] = []
        if scenarios and scenarios.fusion_excluded_building_area_m2:
            excl_parts.append("emprises bâtiments (BD TOPO / masque)")
        if res.road_likelihood and res.road_likelihood > 0.2:
            excl_parts.append("chaussée (partie non stationnable)")
        if res.vegetation_likelihood > 0.15:
            excl_parts.append("végétation")
        res.parking_capacity_estimation = {
            "paradigm": "parking_capacity_estimation",
            "theoretical_primary_method": res.method_used,
            "theoretical_primary_capacity": res.primary_capacity,
            "published_estimate_places": res.estimated_capacity,
            "min_capacity": res.min_capacity,
            "max_capacity": res.max_capacity,
            "hypothesis_m2_per_space": hyp,
            "usable_parking_area_m2": scenarios.fusion_usable_parking_area_m2 if scenarios else None,
            "linear_parking_length_m": res.roadside_length_m,
            "exclusions_summary": " ; ".join(excl_parts) if excl_parts else None,
            "site_type": res.site_type,
        }

        res.warnings = "; ".join(caveats)
        res.caveats = "; ".join(caveats)

        if include_raw:
            res.raw_debug.update(
                {
                    "osm_elements": len(elems),
                    "classified": len(classified),
                    "overpass_remark": raw_osm.get("remark"),
                }
            )
    except GeocodeError as e:
        res.error = f"geocode: {e}"
        res.caveats = str(e)
        res.warnings = str(e)
    except Exception as e:  # noqa: BLE001
        res.error = f"error: {e!s}"
        res.caveats = str(e)
        res.warnings = str(e)
        logger.exception("process_address")
    finally:
        if own_client:
            client.close()

    return res


def row_to_json_serializable(row: RowResult) -> dict[str, Any]:
    d = row.to_flat_dict()
    d.pop("gis_report", None)
    d.pop("supporting_evidence", None)
    d.pop("vehicles_payload", None)
    d.pop("parking_capacity_estimation", None)
    gr = getattr(row, "gis_report", None)
    if isinstance(gr, dict):
        for k in ("gis_sources", "access", "buildings", "fusion"):
            if k in gr and gr[k] is not None:
                d[k] = gr[k]
    if getattr(row, "supporting_evidence", None):
        d["supporting_evidence"] = row.supporting_evidence
    if getattr(row, "vehicles_payload", None):
        d["vehicles"] = row.vehicles_payload
    if getattr(row, "parking_capacity_estimation", None):
        d["parking_capacity_estimation"] = row.parking_capacity_estimation
    if row.raw_debug:
        d["raw_debug"] = row.raw_debug
    return d
