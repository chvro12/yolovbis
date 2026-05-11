"""Orchestration : moisson → puces → entraînement resnet18 → évaluation → rapport."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from parking_capacity.chip_dataset import build_chip_dataset
from parking_capacity.france_presets import resolve_bbox_arg
from parking_capacity.harvest_real_dataset import harvest_real_dataset
from parking_capacity.ml.train import run_eval_from_checkpoint, run_training


def _fmt_examples(df: pd.DataFrame, *, best: bool, k: int = 3) -> str:
    if df.empty or "y_true" not in df.columns:
        return "_Aucun exemple._\n"
    d = df.copy()
    d["abs_error"] = (d["y_true"] - d["y_pred"]).abs()
    d = d.sort_values("abs_error", ascending=best)
    lines: List[str] = []
    for _, r in d.head(k).iterrows():
        lines.append(
            f"- vrai **{r['y_true']:.1f}** → prédit **{r['y_pred']:.1f}** (|erreur|={r['abs_error']:.1f})"
        )
    return "\n".join(lines) + "\n"


def _baseline_mae_vs_truth(pred_val: pd.DataFrame, m2_per_space: float = 26.0) -> Optional[Dict[str, float]]:
    if "area_m2" not in pred_val.columns or "y_true" not in pred_val.columns:
        return None
    sub = pred_val.dropna(subset=["area_m2", "y_true"])
    if sub.empty:
        return None
    base = (sub["area_m2"].astype(float) / m2_per_space).clip(lower=0)
    yt = sub["y_true"].astype(float)
    err = (base - yt).abs()
    return {"mae_baseline_area_ratio": float(err.mean()), "n": int(len(sub))}


def write_real_run_report(
    out_dir: Path,
    *,
    harvest_stats: Dict[str, Any],
    chip_manifest_rows: int,
    train_summary: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    mock_mode: bool,
    m2_per_space: float = 26.0,
) -> Path:
    out_dir = Path(out_dir)
    pred_path = out_dir / "predictions_val.csv"
    pred_val = pd.read_csv(pred_path) if pred_path.is_file() else pd.DataFrame()

    n_samples = int(train_summary.get("n_samples", 0))
    val = train_summary.get("val") or eval_metrics
    mae = float(val.get("mae", float("nan")))
    rmse = float(val.get("rmse", float("nan")))
    r2 = float(val.get("r2", float("nan")))

    baseline_info = _baseline_mae_vs_truth(pred_val, m2_per_space=m2_per_space)
    mae_ml = mae

    lines: List[str] = [
        "# Rapport d’exécution terrain (`make-training-run`)",
        "",
        f"- **Mode mock** : {'oui (données synthétiques, sans moisson réel)' if mock_mode else 'non'}.",
        f"- **Architecture** : `{train_summary.get('architecture', '?')}`"
        + (" (mock CI : `tiny` au lieu de resnet18)" if mock_mode else " (`resnet18` en mode réel)."),
        f"- **Parkings OSM avec capacity (moisson)** : {harvest_stats.get('osm_with_capacity', 'n/a')}.",
        f"- **Puces téléchargées / manifest puces** : {chip_manifest_rows}.",
        f"- **Échantillons entraînement (manifest)** : {n_samples}.",
        f"- **Split** : `{train_summary.get('split_mode', '?')}` (validation intégrée à l’entraînement).",
        "",
        "## Métriques validation (dernier état / jeu val)",
        "",
        f"- **MAE** : {mae:.3f}",
        f"- **RMSE** : {rmse:.3f}",
        f"- **R²** : {r2:.4f}",
        "",
    ]

    if baseline_info:
        lines.extend(
            [
                "## Baseline surface OSM / ratio m² par place",
                "",
                f"- **MAE baseline** (capacité ≈ area_m2 / {m2_per_space}) : **{baseline_info['mae_baseline_area_ratio']:.3f}** (n={baseline_info['n']}).",
                f"- **MAE modèle ML** : **{mae_ml:.3f}**.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Baseline surface / ratio",
                "",
                "_Colonne `area_m2` absente du jeu val : baseline non calculée._",
                "",
            ]
        )

    lines.extend(
        [
            "## Exemples (validation)",
            "",
            "### Meilleures prédictions (faible |erreur|)",
            _fmt_examples(pred_val, best=True),
            "### Pires prédictions",
            _fmt_examples(pred_val, best=False),
            "",
            "## Conclusion (honnête)",
            "",
        ]
    )

    warnings: List[str] = []
    if n_samples < 100:
        warnings.append(
            f"Le jeu ne contient que **{n_samples}** exemples valides (< 100) : "
            "**ne pas** recommander le modèle pour la production ; privilégier OSM, surface et orthophoto."
        )
    if math.isnan(r2) or r2 < 0:
        warnings.append("Le **R² est négatif ou non défini** : le modèle **ne généralise pas** mieux qu’une moyenne naïve sur ce split.")
    if not math.isnan(mae) and mae > 40:
        warnings.append(f"La **MAE ({mae:.1f})** reste élevée : le modèle **n’est pas fiable** pour l’instant.")

    if baseline_info and not math.isnan(mae_ml) and mae_ml >= baseline_info["mae_baseline_area_ratio"] * 0.98:
        warnings.append(
            "Le ML **n’améliore pas clairement** la simple baseline `area_m2 / m²_par_place` sur la validation."
        )

    if warnings:
        lines.append("### Avertissements\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
        lines.append("**Recommandation** : utiliser `run-address` avec `--ml-checkpoint` seulement après amélioration des données et des métriques ; sinon rester en **hybrid / OSM / surface**.")
    else:
        lines.append(
            "Les métriques sont **potentiellement exploitables** pour des tests internes ; "
            "valider tout de même sur des adresses réelles (`diagnose-address`) avant production."
        )

    lines.append("")
    path = out_dir / "real_run_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_training_run(
    out_dir: Path,
    *,
    bbox: Optional[str] = None,
    preset: Optional[str] = None,
    max_samples: int = 500,
    cache_dir: Optional[Path] = None,
    mock: bool = False,
    epochs: int = 8,
    country: str = "FR",
    harvest_delay: float = 0.75,
    chip_delay: float = 0.6,
    half_side_m: float = 55.0,
    chip_pixels: int = 512,
) -> Path:
    """
    Enchaîne harvest-real-dataset → build-chips → train-model (resnet18) → eval-model → ``real_run_report.md``.

    En ``mock=True``, saute le réseau : petit jeu synthétique + ``tiny`` pour la vitesse CI.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    harvest_stats: Dict[str, Any] = {"osm_with_capacity": 0}
    chip_rows = 0

    if mock:
        chip_dir = out_dir / "chips"
        chip_dir.mkdir(parents=True, exist_ok=True)
        n_syn = min(max(30, max_samples), 120)
        arch = "tiny"
        harvest_stats["osm_with_capacity"] = n_syn
        summary = run_training(
            chip_dir=chip_dir,
            manifest_csv=None,
            output_dir=out_dir,
            synthetic_n=n_syn,
            architecture=arch,
            epochs=min(2, epochs),
            batch_size=16,
            split_mode="random",
            augment=False,
            training_meta_extra={"dataset_mode": "mock", "source_preset": preset},
        )
        chip_rows = int(summary.get("n_samples", n_syn))
        eval_metrics = run_eval_from_checkpoint(out_dir / "model.pt", chip_dir)
    else:
        bbox_str = resolve_bbox_arg(bbox=bbox, preset=preset)
        harvest_dir = out_dir / "harvest"
        man_harvest = harvest_real_dataset(
            harvest_dir,
            bbox=bbox_str,
            country=country,
            half_side_m=half_side_m,
            chip_pixels=chip_pixels,
            delay_s=harvest_delay,
            max_features=max_samples,
            cache_dir=cache_dir,
        )
        hdf = pd.read_csv(man_harvest)
        harvest_stats["osm_with_capacity"] = len(hdf)
        chip_dir = out_dir / "chips"
        man_chips = build_chip_dataset(
            man_harvest,
            chip_dir,
            max_rows=max_samples,
            delay_s=chip_delay,
            half_side_m=half_side_m,
            chip_pixels=chip_pixels,
            cache_dir=cache_dir,
        )
        cdf = pd.read_csv(man_chips)
        chip_rows = len(cdf)
        summary = run_training(
            chip_dir=chip_dir,
            manifest_csv=man_chips,
            output_dir=out_dir,
            synthetic_n=0,
            architecture="resnet18",
            epochs=epochs,
            batch_size=16,
            split_mode="geo",
            augment=True,
            training_meta_extra={"source_bbox": bbox_str, "source_preset": preset},
        )
        eval_metrics = run_eval_from_checkpoint(out_dir / "model.pt", chip_dir)

    (out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2, default=float), encoding="utf-8")

    report_path = write_real_run_report(
        out_dir,
        harvest_stats=harvest_stats,
        chip_manifest_rows=chip_rows,
        train_summary=summary,
        eval_metrics=eval_metrics,
        mock_mode=mock,
    )
    return report_path
