"""Benchmark sur un lot d’adresses réelles + export inspection manuelle."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from parking_capacity.diagnose import (
    _write_debug_map_html,
    _write_debug_overlay,
    create_diagnose_mock_transport,
)
from parking_capacity.pipeline import process_address, row_to_json_serializable


def _slug(address: str, idx: int) -> str:
    s = re.sub(r"[^\w\s-]", "", address, flags=re.UNICODE)
    s = re.sub(r"[-\s]+", "_", s).strip("_")[:80]
    return f"{idx:03d}_{s}" if s else f"{idx:03d}_addr"


def _accuracy_rates(err: np.ndarray, thresholds: Tuple[int, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(err)
    if n == 0:
        for t in thresholds:
            out[f"acc_pm{t}"] = float("nan")
        return out
    for t in thresholds:
        out[f"acc_pm{t}"] = float(np.mean(err <= t))
    return out


def run_benchmark_addresses(
    input_csv: Path,
    out_dir: Path,
    *,
    cache_dir: Optional[Path] = None,
    refresh_imagery: bool = False,
    source_priority: str = "hybrid",
    mock: bool = False,
    overpass_delay_s: float = 1.0,
    force_ml: bool = False,
    visual_backend: str = "auto",
    visual_model_specialized_for_parking: bool = False,
    yolo_weights: Optional[Path] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    addr_col = "address" if "address" in df.columns else ("Address" if "Address" in df.columns else None)
    if addr_col is None:
        raise ValueError("CSV : colonne « address » (ou Address) obligatoire.")

    rows_out: List[Dict[str, Any]] = []
    manual_rows: List[Dict[str, Any]] = []

    if mock:
        client = httpx.Client(transport=create_diagnose_mock_transport(), timeout=60.0)
        own = True
        delay_use = 0.0
        min_isec = 1.0
        use_vision = False
    else:
        client = httpx.Client(timeout=120.0, follow_redirects=True)
        own = True
        delay_use = overpass_delay_s
        min_isec = 25.0
        use_vision = True

    try:
        for i, (_, row) in enumerate(df.iterrows()):
            address = str(row[addr_col]).strip()
            if not address:
                continue
            radius_m = int(row["radius_m"]) if "radius_m" in df.columns and pd.notna(row.get("radius_m")) else 50
            exp = row.get("expected_capacity")
            expected: Optional[float] = None
            if exp is not None and str(exp).strip() != "" and not (isinstance(exp, float) and np.isnan(exp)):
                try:
                    expected = float(exp)
                except (TypeError, ValueError):
                    expected = None
            notes_in = row.get("notes")
            notes_s = str(notes_in) if notes_in is not None and not (isinstance(notes_in, float) and np.isnan(notes_in)) else ""

            sub = out_dir / _slug(address, i)
            sub.mkdir(parents=True, exist_ok=True)
            chip_path = sub / "chip.png"

            r = process_address(
                address,
                client=client,
                search_radius_m=radius_m,
                use_vision=use_vision,
                cache_dir=cache_dir,
                save_chip_path=chip_path,
                refresh_imagery=refresh_imagery,
                source_priority=source_priority,
                min_intersection_m2=min_isec,
                overpass_delay_s=delay_use,
                force_ml=force_ml,
                visual_backend=visual_backend,
                visual_model_specialized_for_parking=visual_model_specialized_for_parking,
                yolo_weights=yolo_weights,
            )

            rec = row_to_json_serializable(r)
            rec["input_row_index"] = i
            rec["expected_capacity"] = expected
            rec["benchmark_notes"] = notes_s
            est = r.estimated_capacity
            src = r.method_used
            if expected is not None and est is not None:
                ae = abs(float(est) - expected)
                rec["absolute_error"] = ae
                rec["relative_error"] = ae / max(expected, 1.0)
                rec["theoretical_capacity_error"] = ae
                rec["surface_capacity_error"] = ae if src == "scenario_unmarked_surface" else None
                rec["marked_slot_count_error"] = ae if src == "parking_geometry" else None
                rec["linear_parking_error"] = ae if src == "scenario_roadside_parking" else None
            else:
                rec["absolute_error"] = None
                rec["relative_error"] = None
                rec["theoretical_capacity_error"] = None
                rec["surface_capacity_error"] = None
                rec["marked_slot_count_error"] = None
                rec["linear_parking_error"] = None

            (sub / "result.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            (sub / "warnings.txt").write_text((r.warnings or "") + "\n", encoding="utf-8")

            fetch_half = max(55.0, float(radius_m))
            if chip_path.is_file() and r.lat is not None and r.lon is not None:
                _write_debug_map_html(sub / "debug_map.html", float(r.lat), float(r.lon), float(radius_m))
                try:
                    _write_debug_overlay(
                        chip_path,
                        sub / "debug_overlay.png",
                        radius_m=float(radius_m),
                        half_side_m=float(fetch_half),
                    )
                except Exception:
                    pass

            rows_out.append(rec)

            rel_chip = str(chip_path.relative_to(out_dir)) if chip_path.is_file() else ""
            rel_ov = str((sub / "debug_overlay.png").relative_to(out_dir)) if (sub / "debug_overlay.png").is_file() else ""
            manual_rows.append(
                {
                    "address": address,
                    "estimated_capacity": r.estimated_capacity,
                    "min_capacity": r.min_capacity,
                    "max_capacity": r.max_capacity,
                    "method_used": r.method_used,
                    "visual_evidence_level": r.visual_evidence_level,
                    "image_confidence": r.image_confidence,
                    "chip_path": rel_chip,
                    "overlay_path": rel_ov,
                    "human_count": "",
                    "human_notes": "",
                    "accepted": "",
                }
            )
    finally:
        if own:
            client.close()

    res_df = pd.DataFrame(rows_out)
    res_df.to_csv(out_dir / "results.csv", index=False)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2, default=str)

    pd.DataFrame(manual_rows).to_csv(out_dir / "manual_review.csv", index=False)

    _write_benchmark_report(out_dir, res_df, rows_out)


def _write_benchmark_report(out_dir: Path, res_df: pd.DataFrame, rows_out: List[Dict[str, Any]]) -> None:
    lines = [
        "# Rapport benchmark adresses",
        "",
        f"- Nombre d’adresses : **{len(res_df)}**",
        "",
    ]
    sub = res_df.dropna(subset=["expected_capacity"])
    if not sub.empty and "absolute_error" in sub.columns:
        ae = sub["absolute_error"].astype(float).values
        if len(ae) > 0:
            mae = float(np.mean(ae))
            rmse = float(np.sqrt(np.mean(ae**2)))
            lines.extend(
                [
                    "## Métriques vs `expected_capacity`",
                    "",
                    "Colonnes CSV : `absolute_error`, `theoretical_capacity_error` (capacité publiée), "
                    "`surface_capacity_error` / `marked_slot_count_error` / `linear_parking_error` "
                    "(erreur lorsque la méthode correspond).",
                    "",
                    f"- **MAE** : {mae:.3f}",
                    f"- **RMSE** : {rmse:.3f}",
                ]
            )
            acc = _accuracy_rates(ae, (2, 5, 10))
            for k, v in acc.items():
                lines.append(f"- **{k}** : {v:.2%}")
            ok = mae <= 8.0 and acc.get("acc_pm10", 0) >= 0.5
            lines.extend(["", "## Conclusion", "", "**Fiable**" if ok else "**À améliorer**", ""])
        else:
            lines.append("## Métriques : aucune erreur calculable")
    else:
        lines.extend(
            [
                "## Sans vérité terrain (`expected_capacity` vide)",
                "",
                "Diagnostic produit pour chaque adresse : confiance, preuve visuelle, méthode — "
                "remplissez `expected_capacity` dans le CSV d’entrée pour des métriques quantitatives.",
                "",
            ]
        )
    (out_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
