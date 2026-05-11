"""Métriques segmentation (IoU, Dice) et utilitaires évaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def binary_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / (union + eps))


def binary_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + eps))


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred.astype(np.bool_) == gt.astype(np.bool_)).mean())


def aggregate_scalar_metrics(errors: np.ndarray) -> Dict[str, float]:
    """MAE / RMSE sur vecteur d’erreurs (prédiction − vérité)."""
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {"mae": mae, "rmse": rmse, "n": float(errors.size)}
