"""Fusion réelle IGN BD TOPO + OSM + optionnels (Microsoft, Mapillary) sur une puce orthophoto."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from parking_capacity.geometry import to_metric
from parking_capacity.gis_rasterize import chip_to_bbox4326, rasterize_lines_on_chip, rasterize_polygons_on_chip
from parking_capacity.ign_geoplateforme import (
    geojson_feature_collection_to_features,
    wfs_get_feature_geojson,
    wfs_ping,
)
from parking_capacity.imagery_wms import OrthoChip, chip_m2_per_pixel
from parking_capacity.mapillary_provider import mapillary_images_in_bbox
from parking_capacity.microsoft_buildings import load_building_geometries_for_bbox
from parking_capacity.osm_transport import TransportFetchResult, query_transport_around
from parking_capacity.providers_config import GisProvidersConfig

logger = logging.getLogger(__name__)

DEFAULT_ACCESS_DISTANCE_THRESHOLD_M = 40.0


def _expand_bbox(b: Tuple[float, float, float, float], delta: float) -> Tuple[float, float, float, float]:
    return b[0] - delta, b[1] - delta, b[2] + delta, b[3] + delta


def _bdtopo_road_line_dicts(fc: Dict[str, Any]) -> List[Dict[str, Any]]:
    line_dicts: List[Dict[str, Any]] = []
    for f in geojson_feature_collection_to_features(fc):
        g = f.get("geometry")
        if not g:
            continue
        gt = g.get("type")
        if gt == "LineString":
            line_dicts.append(g)
        elif gt == "MultiLineString":
            for part in g.get("coordinates") or []:
                line_dicts.append({"type": "LineString", "coordinates": part})
    return line_dicts


def _lines_metric_union(line_geojsons: List[Any]) -> Any:
    geoms = []
    for g in line_geojsons:
        geom = shape(g) if isinstance(g, dict) else g
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            geoms.append(to_metric(geom))
        elif geom.geom_type == "MultiLineString":
            for ln in geom.geoms:
                geoms.append(to_metric(ln))
    if not geoms:
        return None
    return unary_union(geoms)


def _chip_polygon_wgs84(chip: OrthoChip) -> Any:
    min_lon, min_lat, max_lon, max_lat = chip_to_bbox4326(chip)
    return box(min_lon, min_lat, max_lon, max_lat)


def _road_access_metric(
    chip: OrthoChip,
    lon: float,
    lat: float,
    osm_line_geoms: List[Any],
    bdtopo_line_dicts: List[Dict[str, Any]],
) -> Tuple[float, bool, str]:
    """(distance_centre_routes_m, connexion_par_proximité, road_source_tag)."""
    parts_metric = []
    src_bits: List[str] = []
    if bdtopo_line_dicts:
        u = _lines_metric_union(bdtopo_line_dicts)
        if u is not None and not u.is_empty:
            parts_metric.append(u)
            src_bits.append("bdtopo")
    if osm_line_geoms:
        u2 = _lines_metric_union(osm_line_geoms)
        if u2 is not None and not u2.is_empty:
            parts_metric.append(u2)
            src_bits.append("osm")
    if not parts_metric:
        return 1e9, False, "none"
    union_m = unary_union(parts_metric)
    chip_poly_m = to_metric(_chip_polygon_wgs84(chip))
    pt_m = to_metric(Point(lon, lat))
    c_m = chip_poly_m.centroid
    d_center = float(c_m.distance(union_m))
    d_point = float(pt_m.distance(union_m))
    d_min = min(d_center, d_point)
    touches = chip_poly_m.distance(union_m) < 1e-3 or chip_poly_m.intersects(union_m.buffer(2.0))
    road_src = "+".join(src_bits) if len(src_bits) > 1 else src_bits[0]
    return d_min, touches, road_src


def _semantic_consistency(
    *,
    bdtopo_b: bool,
    bdtopo_r: bool,
    osm_hw: bool,
    road_gis: bool,
    mapillary_n: int,
) -> float:
    s = 0.0
    if bdtopo_b:
        s += 0.35
    if bdtopo_r:
        s += 0.25
    if osm_hw:
        s += 0.25
    if road_gis:
        s += 0.1
    if mapillary_n > 0:
        s += 0.05
    return float(np.clip(s, 0.0, 1.0))


@dataclass
class GisFusionResult:
    """Sortie fusion : masques + métriques d’accès + drapeaux sources."""

    building_mask_hw: Optional[np.ndarray] = None
    building_mask_source: str = "heuristic"
    road_mask_gis_hw: Optional[np.ndarray] = None
    road_source: str = "none"
    bdtopo_building_mask_hw: Optional[np.ndarray] = None
    bdtopo_road_mask_hw: Optional[np.ndarray] = None
    osm_road_mask_hw: Optional[np.ndarray] = None
    ign_wfs_reachable: bool = False
    bdtopo_buildings_used: bool = False
    bdtopo_roads_used: bool = False
    osm_highways_used: bool = False
    mapillary_used: bool = False
    mapillary_images_nearby: int = 0
    microsoft_buildings_used: bool = False
    microsoft_warning: Optional[str] = None
    access_distance_m: float = 1e9
    road_network_score: float = 0.0
    road_connection_gis: bool = False
    semantic_consistency_score: float = 0.0
    osm_transport: Optional[TransportFetchResult] = None
    notes: List[str] = field(default_factory=list)
    raw_trace: Dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> Dict[str, Any]:
        """Trace JSON-sérialisable (sans ndarray)."""
        return {
            **self.raw_trace,
            "notes": list(self.notes),
            "building_mask_source": self.building_mask_source,
            "road_source": self.road_source,
            "ign_wfs_reachable": self.ign_wfs_reachable,
            "bdtopo_buildings_used": self.bdtopo_buildings_used,
            "bdtopo_roads_used": self.bdtopo_roads_used,
            "osm_highways_used": self.osm_highways_used,
            "mapillary_used": self.mapillary_used,
            "mapillary_images_nearby": self.mapillary_images_nearby,
            "microsoft_buildings_used": self.microsoft_buildings_used,
            "microsoft_warning": self.microsoft_warning,
            "access_distance_m": None if self.access_distance_m > 1e8 else round(self.access_distance_m, 2),
            "road_network_score": round(self.road_network_score, 4),
            "road_connection_gis": self.road_connection_gis,
            "semantic_consistency_score": round(self.semantic_consistency_score, 4),
        }

    def to_row_report(
        self,
        *,
        usable_parking_area_m2: Optional[float],
        excluded_building_area_m2: Optional[float],
        parking_outside_buildings_ratio: Optional[float],
        final_parking_candidate_area_m2: Optional[float],
        semantic_consistency_score: float,
    ) -> Dict[str, Any]:
        return {
            "gis_sources": {
                "ign_wfs_reachable": self.ign_wfs_reachable,
                "bdtopo_buildings_used": self.bdtopo_buildings_used,
                "bdtopo_roads_used": self.bdtopo_roads_used,
                "osm_highways_used": self.osm_highways_used,
                "mapillary_used": self.mapillary_used,
                "microsoft_buildings_used": self.microsoft_buildings_used,
                "microsoft_warning": self.microsoft_warning,
            },
            "access": {
                "road_connection_detected": None,  # rempli pipeline après sémantique
                "access_distance_m": round(self.access_distance_m, 2) if self.access_distance_m < 1e8 else None,
                "road_network_score": round(self.road_network_score, 4),
                "road_source": self.road_source,
                "road_connection_gis": self.road_connection_gis,
            },
            "buildings": {
                "building_mask_source": self.building_mask_source,
                "building_area_m2": None,
                "excluded_building_area_m2": excluded_building_area_m2,
                "parking_outside_buildings_ratio": parking_outside_buildings_ratio,
            },
            "fusion": {
                "usable_parking_area_m2": usable_parking_area_m2,
                "final_parking_candidate_area_m2": final_parking_candidate_area_m2,
                "semantic_consistency_score": round(semantic_consistency_score, 4),
            },
        }

    def to_row_report_merged(
        self,
        *,
        semantic: Any,
        usable_parking_area_m2: Optional[float],
        excluded_building_area_m2: Optional[float],
        final_parking_candidate_area_m2: Optional[float],
    ) -> Dict[str, Any]:
        """Rapport complet pour ``result.json`` (sémantique + fusion)."""
        base = self.to_row_report(
            usable_parking_area_m2=usable_parking_area_m2,
            excluded_building_area_m2=excluded_building_area_m2,
            parking_outside_buildings_ratio=getattr(semantic, "parking_outside_buildings_ratio", None),
            final_parking_candidate_area_m2=final_parking_candidate_area_m2,
            semantic_consistency_score=self.semantic_consistency_score,
        )
        base["buildings"]["building_area_m2"] = getattr(semantic, "building_area_m2", None)
        acc = base["access"]
        acc["road_connection_detected"] = bool(getattr(semantic, "road_connection_detected", False))
        ad = getattr(semantic, "access_distance_m", None)
        if ad is not None:
            acc["access_distance_m"] = round(float(ad), 2)
        acc["road_network_score"] = round(
            max(float(acc.get("road_network_score") or 0.0), float(getattr(semantic, "road_network_score", 0.0))),
            4,
        )
        acc["road_source"] = getattr(semantic, "road_source", acc.get("road_source"))
        base["buildings"]["building_mask_source"] = getattr(semantic, "building_mask_source", self.building_mask_source)
        return base


def build_gis_fusion(
    chip: OrthoChip,
    lat: float,
    lon: float,
    *,
    radius_m: int,
    cfg: GisProvidersConfig,
    client: httpx.Client,
    cache_dir: Optional[Path],
    overpass_delay_s: float,
    access_distance_threshold_m: float = DEFAULT_ACCESS_DISTANCE_THRESHOLD_M,
) -> GisFusionResult:
    out = GisFusionResult()
    trace: Dict[str, Any] = {}
    bbox = chip_to_bbox4326(chip)

    osm_lines: List[Any] = []
    bd_lines: List[Dict[str, Any]] = []

    if cfg.osm_enabled:
        try:
            tr = query_transport_around(
                lat,
                lon,
                radius_m=radius_m,
                base_url=cfg.overpass_url,
                client=client,
                delay_s=overpass_delay_s,
                cache_dir=cache_dir,
            )
            out.osm_transport = tr
            osm_lines = list(tr.line_geoms)
            if osm_lines:
                rm = rasterize_lines_on_chip(chip, osm_lines, thickness_px=8)
                if rm.any():
                    out.osm_road_mask_hw = rm
                    out.osm_highways_used = True
                    trace["osm_transport_lines"] = len(osm_lines)
            trace["osm_highway_ways"] = tr.summary.n_highway_ways
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"osm_transport: {e}")

    if cfg.ign_enabled and cfg.ign_use_bdtopo:
        try:
            out.ign_wfs_reachable = wfs_ping(client, cfg.ign_wfs_url, cfg.ign_bdtopo_buildings_typename)
        except Exception as e:  # noqa: BLE001
            out.ign_wfs_reachable = False
            out.notes.append(f"ign_wfs_ping: {e}")
        try:
            fc_b = wfs_get_feature_geojson(
                client,
                cfg.ign_wfs_url,
                cfg.ign_bdtopo_buildings_typename,
                _expand_bbox(bbox, 0.00025),
                max_features=cfg.ign_wfs_max_features,
                cache_dir=cache_dir,
            )
            feats_b = geojson_feature_collection_to_features(fc_b)
            geoms_b = [f.get("geometry") for f in feats_b if f.get("geometry")]
            if geoms_b:
                bm = rasterize_polygons_on_chip(chip, geoms_b)
                if bm.any():
                    out.bdtopo_building_mask_hw = bm
                    out.building_mask_hw = bm
                    out.building_mask_source = "bdtopo"
                    out.bdtopo_buildings_used = True
                    trace["ign_bdtopo_buildings"] = len(geoms_b)
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"ign_bdtopo_batiments: {e}")

        try:
            fc_r = wfs_get_feature_geojson(
                client,
                cfg.ign_wfs_url,
                cfg.ign_bdtopo_roads_typename,
                _expand_bbox(bbox, 0.00025),
                max_features=min(cfg.ign_wfs_max_features, 400),
                cache_dir=cache_dir,
            )
            bd_lines = _bdtopo_road_line_dicts(fc_r)
            if bd_lines:
                rm_b = rasterize_lines_on_chip(chip, bd_lines, thickness_px=5)
                if rm_b.any():
                    out.bdtopo_road_mask_hw = rm_b
                    out.bdtopo_roads_used = True
                    trace["ign_bdtopo_road_lines"] = len(bd_lines)
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"ign_bdtopo_routes: {e}")

    # Fusion routes : OSM ∪ BD TOPO
    road_parts: List[np.ndarray] = []
    if out.osm_road_mask_hw is not None:
        road_parts.append(out.osm_road_mask_hw)
    if out.bdtopo_road_mask_hw is not None:
        road_parts.append(out.bdtopo_road_mask_hw)
    if road_parts:
        out.road_mask_gis_hw = road_parts[0].copy()
        for p in road_parts[1:]:
            out.road_mask_gis_hw |= p
        if out.bdtopo_roads_used and out.osm_highways_used:
            out.road_source = "bdtopo+osm"
        elif out.bdtopo_roads_used:
            out.road_source = "bdtopo"
        elif out.osm_highways_used:
            out.road_source = "osm"

    dist_m, touches_chip, _src = _road_access_metric(chip, lon, lat, osm_lines, bd_lines)
    out.access_distance_m = dist_m
    out.road_connection_gis = bool(touches_chip or dist_m <= access_distance_threshold_m)
    out.road_network_score = float(
        np.clip(1.0 - min(dist_m, 200.0) / 200.0, 0.0, 1.0) * (0.5 + 0.5 * float(bool(out.bdtopo_roads_used or out.osm_highways_used)))
    )

    # Bâtiments : OSM si pas BD TOPO ; Microsoft si toujours rien
    if out.building_mask_hw is None and out.osm_transport and out.osm_transport.building_geoms:
        try:
            bm2 = rasterize_polygons_on_chip(chip, out.osm_transport.building_geoms)
            if bm2.any():
                out.building_mask_hw = bm2
                out.building_mask_source = "osm"
                trace["osm_building_polys"] = len(out.osm_transport.building_geoms)
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"osm_buildings_raster: {e}")

    if out.building_mask_hw is None and cfg.microsoft_enabled and cfg.microsoft_buildings_path:
        try:
            geoms_ms = load_building_geometries_for_bbox(cfg.microsoft_buildings_path, bbox)
            if not geoms_ms:
                out.microsoft_warning = "MICROSOFT_BUILDINGS_PATH : aucune géométrie dans la bbox (fichier vide ou trop gros / extrait requis)."
            else:
                bm3 = rasterize_polygons_on_chip(chip, geoms_ms)
                if bm3.any():
                    out.building_mask_hw = bm3
                    out.building_mask_source = "microsoft"
                    out.microsoft_buildings_used = True
                    trace["microsoft_buildings"] = len(geoms_ms)
        except Exception as e:  # noqa: BLE001
            out.microsoft_warning = str(e)
            out.notes.append(f"microsoft_buildings: {e}")
    elif cfg.microsoft_buildings_path and not cfg.microsoft_enabled:
        out.microsoft_warning = "Microsoft buildings : activez microsoft_buildings.enabled dans providers.yaml."

    # Mapillary : indice uniquement
    if cfg.mapillary_enabled and cfg.mapillary_token:
        try:
            imgs = mapillary_images_in_bbox(
                client,
                bbox,
                access_token=cfg.mapillary_token,
                graph_base=cfg.mapillary_graph_url,
                limit=15,
            )
            out.mapillary_images_nearby = len(imgs)
            out.mapillary_used = out.mapillary_images_nearby > 0
            trace["mapillary_images"] = out.mapillary_images_nearby
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"mapillary: {e}")

    out.semantic_consistency_score = _semantic_consistency(
        bdtopo_b=out.bdtopo_buildings_used,
        bdtopo_r=out.bdtopo_roads_used,
        osm_hw=out.osm_highways_used,
        road_gis=out.road_connection_gis,
        mapillary_n=out.mapillary_images_nearby,
    )

    out.raw_trace = trace
    return out


def compute_fusion_area_metrics(
    chip: OrthoChip,
    surface_parking_eligible: np.ndarray,
    surface_road: np.ndarray,
    fusion: GisFusionResult,
) -> Tuple[float, float, float, np.ndarray]:
    """(excluded_building_m2, usable_parking_m2, final_candidate_m2, final_mask)."""
    m2 = chip_m2_per_pixel(chip)
    h, w = surface_parking_eligible.shape[:2]
    bmask = fusion.building_mask_hw
    if bmask is None or bmask.shape != (h, w):
        bmask = np.zeros((h, w), dtype=bool)
    else:
        bmask = bmask.astype(bool)
    excluded = float(bmask.sum()) * m2
    eligible = surface_parking_eligible.astype(bool)
    usable = eligible & ~bmask
    usable_m2 = float(usable.sum()) * m2
    rgis = fusion.road_mask_gis_hw
    if rgis is not None and rgis.shape == (h, w):
        dil = rgis.astype(bool)
    else:
        dil = np.zeros_like(usable)
    try:
        import cv2  # type: ignore

        if dil.any():
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dil = cv2.dilate(dil.astype(np.uint8), k, iterations=2).astype(bool)
    except Exception:
        pass
    final_cand = usable & (surface_road.astype(bool) | dil)
    final_m2 = float(final_cand.sum()) * m2
    return excluded, usable_m2, final_m2, final_cand


def write_gis_fusion_debug_pngs(
    out_dir: Path,
    chip: OrthoChip,
    fusion: GisFusionResult,
    final_candidate_mask: Optional[np.ndarray],
) -> None:
    """Écrit les PNG debug fusion (diagnostic)."""
    from PIL import Image

    out_dir = Path(out_dir)
    base = np.asarray(chip.image.convert("RGB"), dtype=np.uint8)

    def _save(mask: Optional[np.ndarray], name: str, color: Tuple[int, int, int]) -> None:
        if mask is None or mask.size == 0 or not mask.any():
            return
        o = base.copy()
        m = mask.astype(bool)
        o[m] = (0.45 * o[m] + 0.55 * np.array(color, dtype=np.uint8)).astype(np.uint8)
        Image.fromarray(o).save(out_dir / name, format="PNG")

    _save(fusion.bdtopo_building_mask_hw, "debug_bdtopo_buildings.png", (220, 60, 60))
    _save(fusion.bdtopo_road_mask_hw, "debug_bdtopo_roads.png", (60, 120, 255))
    _save(fusion.osm_road_mask_hw, "debug_osm_roads.png", (80, 200, 120))
    _save(fusion.road_mask_gis_hw, "debug_gis_fusion.png", (255, 180, 40))
    _save(final_candidate_mask, "debug_final_parking_candidate.png", (200, 255, 100))
