"""Métriques de régression."""

from __future__ import annotations

import math
from typing import List

import torch

from parking_capacity.ml.regression_metrics import regression_metrics_extended


def regression_metrics(y_true: List[float], y_pred: List[float]) -> dict:
    n = len(y_true)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    mae = sum(err) / n
    se = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    rmse = math.sqrt(sum(se) / n)
    mean_y = sum(y_true) / n
    ss_tot = sum((v - mean_y) ** 2 for v in y_true) or 1.0
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


def _inverse_space(vals: List[float], target_transform: str) -> List[float]:
    if target_transform == "log1p":
        return [max(0.0, math.expm1(float(v))) for v in vals]
    return [float(v) for v in vals]


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    target_transform: str = "none",
) -> dict:
    model.train(False)
    ys: List[float] = []
    ps: List[float] = []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu()
        ys.extend(y.float().tolist())
        ps.extend(pred.tolist())
    ys_r = _inverse_space(ys, target_transform)
    ps_r = _inverse_space(ps, target_transform)
    base = regression_metrics(ys_r, ps_r)
    ext = regression_metrics_extended(ys_r, ps_r)
    ext.pop("n", None)
    base.update(ext)
    return base


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    target_transform: str = "none",
) -> tuple[List[float], List[float]]:
    model.train(False)
    ys: List[float] = []
    ps: List[float] = []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu()
        ys.extend(y.float().tolist())
        ps.extend(pred.tolist())
    return _inverse_space(ys, target_transform), _inverse_space(ps, target_transform)
