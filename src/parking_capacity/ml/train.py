"""Boucle d'entraînement + export checkpoint et métriques."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from parking_capacity.ml.dataset import ChipRegressionDataset, build_synthetic_chip_dataset
from parking_capacity.ml.geo_split import geographic_train_val_mask, indices_from_mask
from parking_capacity.ml.metrics import collect_predictions, evaluate_model
from parking_capacity.ml.models import ARCHITECTURES, build_model

DEFAULT_HALF_SIDE_M = 55.0
DEFAULT_CHIP_PIXELS = 512


def _chip_geometry_from_manifest(manifest_csv: Path) -> Tuple[float, int]:
    try:
        df = pd.read_csv(manifest_csv, nrows=1)
        if df.empty:
            return DEFAULT_HALF_SIDE_M, DEFAULT_CHIP_PIXELS
        row = df.iloc[0]
        hs = DEFAULT_HALF_SIDE_M
        px = DEFAULT_CHIP_PIXELS
        if "half_side_m" in df.columns and pd.notna(row.get("half_side_m")):
            hs = float(row["half_side_m"])
        if "chip_pixels" in df.columns and pd.notna(row.get("chip_pixels")):
            px = int(row["chip_pixels"])
        return hs, px
    except Exception:
        return DEFAULT_HALF_SIDE_M, DEFAULT_CHIP_PIXELS


def _split_indices_random(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_val = max(1, min(n - 1, int(round(n * val_frac))))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def run_training(
    *,
    chip_dir: Path,
    manifest_csv: Optional[Path],
    output_dir: Path,
    synthetic_n: int = 0,
    architecture: str = "tiny",
    pretrained: bool = True,
    img_size: int = 128,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    seed: int = 42,
    device_str: Optional[str] = None,
    half_side_m: Optional[float] = None,
    chip_pixels: Optional[int] = None,
    loss: str = "mse",
    target_transform: str = "none",
    split_mode: str = "random",
    augment: bool = True,
    training_meta_extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Entraîne une régression capacité.

    ``split_mode`` : ``random`` (défaut) ou ``geo`` (masque par cellule lon/lat arrondie).
    ``loss`` : ``mse`` ou ``huber``.
    ``target_transform`` : ``none`` ou ``log1p`` (cible log1p(capacity), métriques en places réelles).
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture doit être parmi {ARCHITECTURES}")
    if loss not in ("mse", "huber"):
        raise ValueError("loss doit être mse ou huber")
    if target_transform not in ("none", "log1p"):
        raise ValueError("target_transform doit être none ou log1p")
    if split_mode not in ("random", "geo"):
        raise ValueError("split_mode doit être random ou geo")

    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist_path = out / "metrics_history.jsonl"
    if hist_path.exists():
        hist_path.unlink()

    train_half_side = half_side_m
    train_chip_px = chip_pixels

    if synthetic_n > 0:
        man = build_synthetic_chip_dataset(chip_dir, synthetic_n, img_size=img_size, seed=seed)
        if train_half_side is None:
            train_half_side = DEFAULT_HALF_SIDE_M
        if train_chip_px is None:
            train_chip_px = DEFAULT_CHIP_PIXELS
    else:
        man = manifest_csv or (Path(chip_dir) / "manifest.csv")
        if not man.exists():
            raise FileNotFoundError(f"Manifest introuvable : {man}")
        if train_half_side is None or train_chip_px is None:
            m_hs, m_px = _chip_geometry_from_manifest(Path(man))
            if train_half_side is None:
                train_half_side = m_hs
            if train_chip_px is None:
                train_chip_px = m_px

    shutil.copyfile(man, out / "dataset_manifest.csv")

    df_manifest = pd.read_csv(man)
    n = len(df_manifest)
    if n < 2:
        raise ValueError("Pas assez d'échantillons pour entraîner")

    if split_mode == "geo" and "lon" in df_manifest.columns and "lat" in df_manifest.columns:
        m_tr, m_va = geographic_train_val_mask(
            df_manifest, lon_col="lon", lat_col="lat", val_frac=val_frac, seed=seed
        )
        tr_idx = indices_from_mask(m_tr)
        va_idx = indices_from_mask(m_va)
    else:
        tr_idx, va_idx = _split_indices_random(n, val_frac, seed)

    ds_train = ChipRegressionDataset(
        man,
        Path(chip_dir),
        img_size=img_size,
        augment_train=bool(augment and architecture != "tiny"),
        target_transform=target_transform,
    )
    ds_val = ChipRegressionDataset(
        man,
        Path(chip_dir),
        img_size=img_size,
        augment_train=False,
        target_transform=target_transform,
    )
    train_ds = Subset(ds_train, tr_idx.tolist())
    val_ds = Subset(ds_val, va_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(architecture, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if loss == "huber":
        loss_fn = nn.HuberLoss(delta=1.0)
    else:
        loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train(True)
        total_loss = 0.0
        nb = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            l = loss_fn(pred, y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            total_loss += float(l.item()) * x.size(0)
            nb += x.size(0)
        train_loss = total_loss / max(nb, 1)

        val_metrics = evaluate_model(model, val_loader, device, target_transform=target_transform)
        row = {"epoch": ep, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        if val_metrics.get("mae", float("nan")) < best_mae:
            best_mae = float(val_metrics["mae"])
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        with (out / "metrics_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out / "model.pt"
    final_val = evaluate_model(model, val_loader, device, target_transform=target_transform)
    yt, yp = collect_predictions(model, val_loader, device, target_transform=target_transform)
    val_rows = df_manifest.iloc[va_idx.tolist()].reset_index(drop=True)
    pred_df = val_rows.copy()
    pred_df["y_true"] = yt
    pred_df["y_pred"] = yp
    pred_df["abs_error"] = [abs(a - b) for a, b in zip(yt, yp)]
    pred_df.to_csv(out / "predictions_val.csv", index=False)

    pd.DataFrame([final_val]).to_csv(out / "metrics.csv", index=False)

    dataset_mode = "synthetic" if synthetic_n > 0 else "real"
    infer_meta: Dict[str, Any] = {
        "architecture": architecture,
        "img_size": img_size,
        "half_side_m": float(train_half_side),
        "chip_pixels": int(train_chip_px),
        "target_transform": target_transform,
        "loss": loss,
        "split_mode": split_mode,
        "split_method": split_mode,
        "n_train_samples": int(len(tr_idx)),
        "n_val_samples": int(len(va_idx)),
        "n_samples": n,
        "dataset_mode": dataset_mode,
        "val_mae": float(final_val.get("mae", float("nan"))),
        "val_rmse": float(final_val.get("rmse", float("nan"))),
        "val_r2": float(final_val.get("r2", float("nan"))),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if training_meta_extra:
        infer_meta.update({k: v for k, v in training_meta_extra.items() if v is not None})
    (out / "model_meta.json").write_text(json.dumps(infer_meta, indent=2, default=float), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "img_size": img_size,
            "half_side_m": float(train_half_side),
            "chip_pixels": int(train_chip_px),
            "manifest": str(man),
            "chip_dir": str(chip_dir),
            "target_transform": target_transform,
            "loss": loss,
            "split_mode": split_mode,
            "n_train_samples": int(len(tr_idx)),
            "n_val_samples": int(len(va_idx)),
            "dataset_mode": infer_meta.get("dataset_mode"),
            "val_r2": infer_meta.get("val_r2"),
        },
        ckpt_path,
    )

    summary = {
        "architecture": architecture,
        "img_size": img_size,
        "half_side_m": float(train_half_side),
        "chip_pixels": int(train_chip_px),
        "loss": loss,
        "target_transform": target_transform,
        "split_mode": split_mode,
        "epochs": epochs,
        "n_samples": n,
        "val": final_val,
        "best_val_mae": best_mae,
        "checkpoint": str(ckpt_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def run_eval_from_checkpoint(
    checkpoint: Path,
    chip_dir: Path,
    manifest_csv: Optional[Path] = None,
    *,
    batch_size: int = 32,
    img_size: Optional[int] = None,
    device_str: Optional[str] = None,
) -> dict:
    """Charge `model.pt` et calcule les métriques sur tout le manifest."""
    payload = torch.load(checkpoint, map_location="cpu")
    arch = str(payload.get("architecture", "tiny"))
    if arch not in ARCHITECTURES:
        arch = "tiny"
    imsz = int(payload.get("img_size", img_size or 128))
    tt = str(payload.get("target_transform", "none"))
    model = build_model(arch, pretrained=False)  # type: ignore[arg-type]
    model.load_state_dict(payload["state_dict"])
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    man = manifest_csv or Path(str(payload.get("manifest", "")))
    if not man.exists():
        man = Path(chip_dir) / "manifest.csv"
    root = Path(str(payload.get("chip_dir", chip_dir)))
    ds_eval = ChipRegressionDataset(
        man, root, img_size=imsz, augment_train=False, target_transform=tt
    )
    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=0)
    return evaluate_model(model, loader, device, target_transform=tt)
