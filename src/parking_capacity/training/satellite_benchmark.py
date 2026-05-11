"""Benchmark modes : heuristiques vs segmentation vs segmentation+GIS (agrégation JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def scan_benchmark_directory(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in root.rglob("result.json"):
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "path": str(result),
                "estimated_capacity": data.get("estimated_capacity"),
                "refuse_prediction": data.get("refuse_prediction"),
                "primary_capacity": data.get("primary_capacity"),
                "expected_capacity": data.get("expected_capacity"),
                "method_used": data.get("method_used"),
                "geometric_capacity_estimate": data.get("geometric_capacity_estimate"),
                "unmarked_capacity_estimate": data.get("unmarked_capacity_estimate"),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]], truth_key: str = "expected_capacity") -> Dict[str, Any]:
    errs: List[float] = []
    refused = 0
    over = 0
    for r in rows:
        if r.get("refuse_prediction"):
            refused += 1
        exp = r.get(truth_key)
        est = r.get("estimated_capacity")
        if exp is not None and est is not None:
            errs.append(float(est) - float(exp))
            if float(est) > float(exp):
                over += 1
    out: Dict[str, Any] = {
        "n": len(rows),
        "refusal_rate": refused / len(rows) if rows else 0.0,
    }
    if errs:
        e = np.array(errs, dtype=np.float64)
        out["mae"] = float(np.mean(np.abs(e)))
        out["rmse"] = float(np.sqrt(np.mean(e**2)))
        out["overestimation_ratio"] = over / len(errs)
    return out


def write_satellite_benchmark_report(root: Path, out_md: Path) -> Dict[str, Any]:
    rows = scan_benchmark_directory(root)
    summ = summarize(rows)
    lines = [
        "# Benchmark satellite (agrégation result.json)",
        "",
        f"- Dossiers analysés : `{root}`",
        f"- Échantillons : {summ.get('n', 0)}",
        f"- Taux refus : {summ.get('refusal_rate', 0):.3f}",
    ]
    if "mae" in summ:
        lines.extend(
            [
                f"- MAE capacité : {summ['mae']:.3f}",
                f"- RMSE : {summ['rmse']:.3f}",
                f"- Surestimation (est > attendu) : {summ['overestimation_ratio']:.3f}",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summ


def scan_segmentation_benchmark_files(root: Path) -> List[Dict[str, Any]]:
    """Lit les ``segmentation_benchmark.json`` (modes A/B/C/D)."""
    found: List[Dict[str, Any]] = []
    for p in root.rglob("segmentation_benchmark.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        found.append({"path": str(p), "data": data})
    return found


def _scalar_for_mae(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in (
            "theoretical_spaces",
            "estimated_capacity",
            "theoretical_spaces_seg_only",
            "primary_capacity",
        ):
            if k in v and v[k] is not None:
                return float(v[k])
    return None


def summarize_mode_benchmarks(
    files: List[Dict[str, Any]],
    truth_key: str = "expected_capacity",
) -> Dict[str, Any]:
    """Agrège par mode si ``expected_capacity`` est fourni dans un ``result.json`` parent."""
    from collections import defaultdict

    per_mode: Dict[str, List[float]] = defaultdict(list)
    massive = 0
    n_truth = 0
    for item in files:
        parent = Path(item["path"]).parent
        rj = parent / "result.json"
        exp: Optional[float] = None
        if rj.is_file():
            try:
                rdata = json.loads(rj.read_text(encoding="utf-8"))
                exp = rdata.get(truth_key)
            except (json.JSONDecodeError, OSError):
                pass
        if exp is None:
            continue
        n_truth += 1
        modes = item["data"].get("modes") or {}
        for mode_name, payload in modes.items():
            est = _scalar_for_mae(payload)
            if est is None:
                continue
            per_mode[mode_name].append(float(est) - float(exp))
            if est > 5 * max(float(exp), 1.0):
                massive += 1

    out: Dict[str, Any] = {"n_with_ground_truth": n_truth, "false_massive_estimates": massive}
    for mode, errs in per_mode.items():
        e = np.array(errs, dtype=np.float64)
        out[mode] = {
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "overestimation_ratio": float(np.sum(e > 0) / max(len(e), 1)),
            "n": int(len(e)),
        }
    return out


def write_extended_satellite_benchmark_report(
    root: Path,
    out_md: Path,
    *,
    truth_key: str = "expected_capacity",
) -> Dict[str, Any]:
    rows = scan_benchmark_directory(root)
    summ = summarize(rows)
    files = scan_segmentation_benchmark_files(root)
    ext = summarize_mode_benchmarks(files, truth_key=truth_key)
    lines = [
        "# Benchmark satellite étendu",
        "",
        "## Agrégat pipeline classique (result.json)",
        "",
        f"- Échantillons : {summ.get('n', 0)}",
        f"- Taux refus : {summ.get('refusal_rate', 0):.3f}",
    ]
    if "mae" in summ:
        lines.extend(
            [
                f"- MAE capacité : {summ['mae']:.3f}",
                f"- RMSE : {summ['rmse']:.3f}",
                f"- Surestimation : {summ['overestimation_ratio']:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Modes segmentation_benchmark.json (vérité = expected_capacity dans result.json voisin)",
            "",
            "```json",
            json.dumps(ext, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"base": summ, "modes": ext}
