"""Workflow unique : APKLOT → layout YOLO → entraînement YOLOv8-seg → artefacts."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from parking_capacity.datasets_satellite.apklot import (
    _default_prepared,
    _default_raw,
    download_apklot,
    prepare_apklot_dataset,
)
from parking_capacity.datasets_satellite.yolo_seg_layout import build_yolo_seg_layout


def _project_root(project_root: Optional[Path]) -> Path:
    return project_root or Path(__file__).resolve().parents[2]


def run_quickstart_apklot_yolo(
    *,
    run_name: Optional[str] = None,
    dataset_subset: str = "small",
    subset_max_images: int = 36,
    epochs: int = 20,
    imgsz: int = 640,
    model: str = "yolov8m-seg.pt",
    patience: int = 15,
    resume: bool = False,
    weights_resume: Optional[Path] = None,
    project_root: Optional[Path] = None,
    skip_train: bool = False,
    apklot_view: str = "satellite",
    force_incompatible_dataset: bool = False,
) -> Dict[str, Any]:
    """
    1) Télécharge APKLOT si besoin (clone git).
    2) Prépare dataset (+ subset ``small`` pour aller vite).
    3) Layout YOLO ``images/train|val``.
    4) Entraîne YOLOv8-seg (Ultralytics).
    5) Copie métriques + exemples sous ``data/runs/<run>/``.
    """
    root = _project_root(project_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rn = run_name or f"apklot_yolo_{ts}"
    run_dir = root / "data" / "runs" / rn
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = _default_raw(root)
    prepared = _default_prepared(root)
    unified = prepared / "parking_capacity_dataset"

    dl = download_apklot(project_root=root)
    if not dl.get("ok"):
        return {"ok": False, "stage": "download", "detail": dl}

    prep = prepare_apklot_dataset(
        raw_root=raw,
        out_dir=prepared,
        dataset_subset=dataset_subset,
        subset_max_images=subset_max_images,
        project_root=root,
        apklot_view=apklot_view,
    )
    if not prep.get("ok"):
        return {"ok": False, "stage": "prepare", "detail": prep}

    yolo_ds = run_dir / "yolo_dataset"
    layout_info = build_yolo_seg_layout(unified, yolo_ds, copy_files=True)
    yaml_path = Path(layout_info["dataset_yaml"])

    train_out: Dict[str, Any] = {"skipped": skip_train}
    best_weights: Optional[Path] = None

    if not skip_train:
        from parking_capacity.datasets_satellite.dataset_types import (
            assert_satellite_segmentation_training_allowed,
        )

        try:
            assert_satellite_segmentation_training_allowed(
                "apklot",
                force=force_incompatible_dataset,
                project_root=root,
            )
        except ValueError as e:
            return {"ok": False, "stage": "training_gate", "error": str(e)}

        try:
            from ultralytics import YOLO
        except ImportError as e:
            return {"ok": False, "stage": "ultralytics", "error": str(e)}

        model_obj = YOLO(model if str(model).endswith(".pt") else f"{model}")
        train_kw: Dict[str, Any] = dict(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            patience=patience,
            amp=True,
            batch=-1,
            project=str(run_dir),
            name="yolo_train",
            exist_ok=True,
        )
        if resume and weights_resume is not None:
            train_kw["resume"] = str(weights_resume)
        elif resume:
            train_kw["resume"] = True
        results = model_obj.train(**train_kw)
        train_out["results_type"] = str(type(results))
        wdir = run_dir / "yolo_train" / "weights"
        best_pt = wdir / "best.pt"
        if best_pt.is_file():
            best_weights = best_pt
            shutil.copy2(best_pt, run_dir / "best.pt")

        # metrics export
        metrics_obj = getattr(results, "results_dict", None) or getattr(results, "metrics", None)
        metrics_path = run_dir / "train_metrics.json"
        try:
            if metrics_obj is not None and hasattr(metrics_obj, "items"):
                metrics_path.write_text(json.dumps(dict(metrics_obj), indent=2, default=float), encoding="utf-8")
            else:
                metrics_path.write_text(json.dumps({"note": "Voir runs/yolo_train/results.csv"}, indent=2))
        except Exception:
            metrics_path.write_text("{}")

        # copier graphiques Ultralytics
        ultra_dir = run_dir / "yolo_train"
        rc = ultra_dir / "results.csv"
        if rc.is_file():
            shutil.copy2(rc, run_dir / "train_metrics.csv")

        # copier graphiques Ultralytics
        for pat in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "MaskPR_curve.png"):
            src = ultra_dir / pat
            if src.is_file():
                shutil.copy2(src, run_dir / pat)

        # prédictions validation + exemples
        sample_masks = run_dir / "sample_masks"
        sample_overlays = run_dir / "sample_overlays"
        val_pred = run_dir / "val_predictions"
        pred_ex = run_dir / "prediction_examples"
        for d in (sample_masks, sample_overlays, val_pred, pred_ex):
            d.mkdir(parents=True, exist_ok=True)

        val_img_dir = yolo_ds / "images" / "val"
        if best_pt.is_file() and val_img_dir.is_dir():
            import numpy as np
            from PIL import Image as I

            infer = YOLO(str(best_pt))
            infer.predict(
                source=str(val_img_dir),
                imgsz=imgsz,
                save=True,
                project=str(val_pred),
                name="batch",
            )
            for img_p in sorted(val_img_dir.glob("*"))[:12]:
                if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                r0 = infer.predict(str(img_p), imgsz=imgsz, verbose=False)[0]
                if r0.masks is None:
                    continue
                import cv2

                md = r0.masks.data.cpu().numpy()
                comb = (md.max(axis=0) > 0.5).astype(np.uint8) * 255
                if comb.shape[0] != r0.orig_shape[0]:
                    comb = cv2.resize(comb, (r0.orig_shape[1], r0.orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                I.fromarray(comb).save(sample_masks / f"{img_p.stem}_mask.png")
                base = I.open(img_p).convert("RGB")
                arr = np.array(base).astype(np.float32)
                green = np.zeros_like(arr)
                green[:, :, 1] = comb.astype(np.float32)
                over = np.clip(arr * 0.55 + green * 0.45, 0, 255).astype(np.uint8)
                I.fromarray(over).save(sample_overlays / f"{img_p.stem}_overlay.png")
                shutil.copy2(img_p, pred_ex / img_p.name)

    manifest = {
        "run_name": rn,
        "run_dir": str(run_dir.resolve()),
        "dataset_subset": dataset_subset,
        "layout": layout_info,
        "download": dl,
        "prepare": prep,
        "train": train_out,
        "best_weights": str(best_weights) if best_weights else None,
    }
    (run_dir / "quickstart_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"ok": True, **manifest}
