"""Test segmentation YOLO + fusion GIS sur orthophoto IGN réelle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import numpy as np

from parking_capacity.geocode import geocode_address
from parking_capacity.gis_context import fetch_chip_gis_augmentation
from parking_capacity.imagery_wms import chip_m2_per_pixel, fetch_ortho_chip
from parking_capacity.pipeline import process_address
from parking_capacity.multiscale_inference import (
    multiscale_chips_and_masks,
    save_debug_multiscale_fusion,
)
from parking_capacity.providers_config import load_gis_providers_config
from parking_capacity.segmentation_gis_fusion import (
    fuse_segmentation_and_gis,
    write_fusion_debug_bundle,
)
from parking_capacity.yolo_seg_inference import predict_parking_mask


def run_test_segmentation_real(
    address: str,
    weights: Path,
    out_dir: Path,
    *,
    radius_m: int = 80,
    chip_pixels: int = 640,
    half_side_m: Optional[float] = None,
    m2_per_space: float = 26.0,
    cache_dir: Optional[Path] = None,
    providers_yaml: Optional[Path] = None,
    multiscale: bool = False,
    overpass_delay_s: float = 1.0,
) -> Dict[str, Any]:
    """
    Orthophoto → YOLO seg → GIS fusion → GeoJSON + PNG debug.
    Si ``multiscale``, fusionne des masques 25/50/80 m (approximation image).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hs = float(half_side_m if half_side_m is not None else max(radius_m, 40.0))

    gc = geocode_address(address)
    lon, lat = gc.lon, gc.lat

    client = httpx.Client(timeout=120.0)
    try:
        chip = fetch_ortho_chip(
            lon,
            lat,
            half_side_m=hs,
            width_px=chip_pixels,
            height_px=chip_pixels,
            client=client,
            cache_dir=cache_dir,
        )
        cfg = load_gis_providers_config(yaml_path=providers_yaml)
        gis = fetch_chip_gis_augmentation(
            chip,
            lat,
            lon,
            radius_m=radius_m,
            cfg=cfg,
            client=client,
            cache_dir=cache_dir,
            overpass_delay_s=overpass_delay_s,
        )

        if multiscale:

            def _pred(c):
                return predict_parking_mask(c.image, weights, imgsz=chip_pixels)

            chips, masks, fused = multiscale_chips_and_masks(
                lon,
                lat,
                _pred,
                half_sides_m=(25.0, 50.0, hs),
                chip_pixels=chip_pixels,
                client=client,
                cache_dir=cache_dir,
            )
            mask_hw = fused
            ref_chip = chips[-1]
            save_debug_multiscale_fusion(ref_chip.image, mask_hw, out_dir / "debug_multiscale_fusion.png")
            chip_for_metrics = ref_chip
        else:
            mask_hw = predict_parking_mask(chip.image, weights, imgsz=chip_pixels)
            chip_for_metrics = chip

        fusion = fuse_segmentation_and_gis(
            chip_for_metrics,
            mask_hw,
            gis,
            m2_per_space=m2_per_space,
        )
        dbg = write_fusion_debug_bundle(chip_for_metrics, fusion, out_dir)

        m2px = chip_m2_per_pixel(chip_for_metrics)
        seg_px = float(np.sum(fusion.parking_mask_hw.astype(np.float64)))
        bench = {
            "address": address,
            "modes": {
                "B_segmentation_only": {
                    "usable_area_m2": seg_px * m2px,
                    "theoretical_spaces_seg_only": int(max(0, (seg_px * m2px) / m2_per_space)),
                },
                "C_segmentation_plus_gis": {
                    "usable_area_m2": fusion.usable_area_m2,
                    "theoretical_spaces": fusion.theoretical_spaces,
                },
            },
        }
        row = process_address(
            address,
            client=client,
            search_radius_m=radius_m,
            chip_half_side_m=hs,
            chip_pixels=chip_pixels,
            cache_dir=cache_dir,
            source_priority="hybrid",
            yolo_weights=weights,
            visual_backend="yolo_parking",
            providers_yaml=providers_yaml,
            overpass_delay_s=overpass_delay_s,
        )
        bench["modes"]["A_heuristic_pipeline"] = {
            "estimated_capacity": row.estimated_capacity,
            "primary_capacity": row.primary_capacity,
            "refuse_prediction": getattr(row, "refuse_prediction", None),
            "method_used": row.method_used,
        }
        bench["modes"]["D_segmentation_gis_geometry"] = {
            "estimated_capacity": row.estimated_capacity,
            "geometric_capacity_estimate": getattr(row, "geometric_capacity_estimate", None),
            "method_used": row.method_used,
        }
        (out_dir / "segmentation_benchmark.json").write_text(json.dumps(bench, indent=2, default=str), encoding="utf-8")

        summary = {
            "address": address,
            "lon": lon,
            "lat": lat,
            "weights": str(weights),
            "usable_area_m2": fusion.usable_area_m2,
            "theoretical_spaces": fusion.theoretical_spaces,
            "debug": dbg,
            "segmentation_benchmark_path": str(out_dir / "segmentation_benchmark.json"),
        }
        (out_dir / "result_segmentation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        client.close()


def add_heuristic_mode_to_benchmark(
    out_dir: Path,
    *,
    estimated_capacity: Optional[float],
    primary_capacity: Optional[float],
    refuse_prediction: Optional[bool],
) -> None:
    """Ajoute le mode A (pipeline classique) au fichier segmentation_benchmark.json si présent."""
    p = out_dir / "segmentation_benchmark.json"
    if not p.is_file():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    modes = data.setdefault("modes", {})
    modes["A_heuristic_pipeline"] = {
        "estimated_capacity": estimated_capacity,
        "primary_capacity": primary_capacity,
        "refuse_prediction": refuse_prediction,
    }
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
