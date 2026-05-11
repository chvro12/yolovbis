"""Métriques de régression étendues (MAPE, médiane, buckets)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def regression_metrics_extended(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = [float(x) for x in y_true]
    yp = [float(x) for x in y_pred]
    n = len(yt)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = [abs(a - b) for a, b in zip(yt, yp)]
    mae = sum(err) / n
    se = [(a - b) ** 2 for a, b in zip(yt, yp)]
    rmse = math.sqrt(sum(se) / n)
    mean_y = sum(yt) / n
    ss_tot = sum((v - mean_y) ** 2 for v in yt) or 1.0
    ss_res = sum((a - b) ** 2 for a, b in zip(yt, yp))
    r2 = 1.0 - ss_res / ss_tot
    med = sorted(err)[len(err) // 2]
    mape_vals = [abs(a - b) / a for a, b in zip(yt, yp) if a > 0]
    mape = sum(mape_vals) / len(mape_vals) if mape_vals else float("nan")

    buckets: Dict[str, List[float]] = {"0-10": [], "10-30": [], "30-100": [], "100+": []}
    for a, b in zip(yt, yp):
        key = "100+" if a >= 100 else "30-100" if a >= 30 else "10-30" if a >= 10 else "0-10"
        buckets[key].append(abs(a - b))
    bucket_mae = {k: (sum(v) / len(v) if v else float("nan")) for k, v in buckets.items()}

    out: Dict[str, float] = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": float(med),
        "mape": float(mape),
        "n": float(n),
    }
    for k, v in bucket_mae.items():
        out[f"bucket_mae_{k.replace('-', '_')}"] = v
    return out
