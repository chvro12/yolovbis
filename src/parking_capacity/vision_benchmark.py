"""Benchmark comparatif : surface OSM vs géométrie vs SegFormer vs ML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from parking_capacity.pipeline import process_address, row_to_json_serializable


def run_benchmark_vision_modes(
    address: str,
    out_dir: Path,
    *,
    client: Optional[httpx.Client] = None,
    radius_m: int = 50,
    ml_checkpoint: Optional[Path] = None,
    yolo_weights: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Pour une même adresse, enchaîne plusieurs ``visual_backend`` (+ ML optionnel) et écrit ``vision_benchmark.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    if own:
        client = httpx.Client(timeout=120.0, follow_redirects=True)
    modes: List[tuple[str, Dict[str, Any]]] = [
        ("area_osm_baseline", {"visual_backend": "none", "use_vision": False}),
        ("geometry_only", {"visual_backend": "geometry_only", "use_vision": False}),
        ("segformer_generic", {"visual_backend": "segformer_generic", "use_vision": True}),
    ]
    if ml_checkpoint is not None and Path(ml_checkpoint).is_file():
        modes.append(("ml_regression", {"visual_backend": "auto", "use_vision": True}))
    if yolo_weights is not None and Path(yolo_weights).is_file():
        modes.append(("yolo_parking", {"visual_backend": "yolo_parking", "use_vision": True}))

    rows: List[Dict[str, Any]] = []
    try:
        for name, kwargs in modes:
            r = process_address(
                address,
                client=client,
                search_radius_m=radius_m,
                cache_dir=cache_dir,
                ml_checkpoint=ml_checkpoint if name == "ml_regression" else None,
                yolo_weights=yolo_weights if name == "yolo_parking" else None,
                **kwargs,
            )
            d = row_to_json_serializable(r)
            d["mode"] = name
            rows.append(d)
    finally:
        if own and client:
            client.close()

    path = out_dir / "vision_benchmark.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = out_dir / "vision_benchmark.md"
    lines = ["# Benchmark modes vision", "", f"Adresse : `{address}`", ""]
    for d in rows:
        lines.append(f"## {d.get('mode')}")
        lines.append(f"- Capacité estimée : {d.get('estimated_capacity')}")
        lines.append(f"- Méthode : {d.get('method_used')}")
        lines.append(f"- Géométrie : {d.get('geometric_capacity_estimate')} ({d.get('geometry_confidence')})")
        lines.append(f"- Fiabilité globale : {d.get('overall_reliability_score')}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return path
