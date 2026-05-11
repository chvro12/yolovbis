"""Synthèse go / no-go pour usage produit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from parking_capacity.ml.model_meta import load_model_meta, model_meta_blocks_primary_ml


def write_go_no_go_report(
    benchmark_dir: Path,
    model_pt: Path,
    out_md: Path,
    *,
    manual_eval_dir: Optional[Path] = None,
) -> Path:
    benchmark_dir = Path(benchmark_dir)
    model_pt = Path(model_pt)
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    meta = load_model_meta(model_pt)
    results_csv = benchmark_dir / "results.csv"
    bench_rep = benchmark_dir / "benchmark_report.md"

    lines: List[str] = [
        "# Rapport go / no-go",
        "",
        "## Modèle",
        "",
    ]

    model_ok = True
    reasons_block: List[str] = []

    if meta is None:
        lines.append("_Pas de `model_meta.json` à côté du checkpoint : impossible de valider automatiquement la qualité d’entraînement._")
        model_ok = False
        reasons_block.append("Métadonnées modèle absentes")
    else:
        lines.append(f"- Architecture : `{meta.get('architecture')}`")
        lines.append(f"- `dataset_mode` : `{meta.get('dataset_mode')}`")
        lines.append(f"- `n_train_samples` : {meta.get('n_train_samples')}")
        lines.append(f"- Validation MAE / RMSE / R² : {meta.get('val_mae')} / {meta.get('val_rmse')} / {meta.get('val_r2')}")
        lines.append(f"- Créé : {meta.get('created_at', '—')}")
        if model_meta_blocks_primary_ml(meta):
            model_ok = False
            reasons_block.append("Modèle signalé comme faible (synthétique, <100 ex., ou R² val < 0)")
        lines.append("")

    lines.extend(["## Benchmark (fichiers)", "", f"- Dossier : `{benchmark_dir}`", ""])
    if results_csv.is_file():
        df = pd.read_csv(results_csv)
        lines.append(f"- Lignes résultats : **{len(df)}**")
        if "expected_capacity" in df.columns and df["expected_capacity"].notna().any():
            sub = df.dropna(subset=["expected_capacity", "estimated_capacity"])
            if not sub.empty and "absolute_error" in df.columns:
                mae = sub["absolute_error"].astype(float).mean()
                lines.append(f"- MAE sur vérité CSV : **{mae:.2f}**")
        lines.append("")
    else:
        lines.append("_Pas de `results.csv`._")
        lines.append("")

    if bench_rep.is_file():
        lines.extend(["### Extrait benchmark interne", "", "```", bench_rep.read_text(encoding="utf-8")[:2500], "```", ""])

    me_dir = Path(manual_eval_dir) if manual_eval_dir else benchmark_dir.parent / "manual_eval"
    summ_path = me_dir / "summary.json"
    if summ_path.is_file():
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Inspection manuelle",
                "",
                f"- Dossier évaluation : `{me_dir}`",
                f"- Paires humain / estimé : {summ.get('n_pairs')}",
                f"- MAE humain : {summ.get('mae')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Décision",
            "",
            "**Modèle utilisable en source principale ?** " + ("**Oui (avec prudence)**" if model_ok else "**Non (bloquant sans `--force-ml`)**"),
            "",
            "### Cas où refuser de prédire une valeur unique",
            "",
            "- Preuve orthophoto `none` et aucun tag OSM `capacity` fiable sur la zone.",
            "- Checkpoint ML sans métadonnées ou avec jeu d’entraînement insuffisant.",
            "- Adresse mal géocodée (score BAN faible).",
            "",
            "### Prochaines données à collecter",
            "",
            "- Plus de points avec `expected_capacity` ou `manual_review.csv` rempli.",
            "- Moisson OSM sur bbox plus large avec contrôle terrain.",
            "",
            "### Baseline",
            "",
            "Comparer systématiquement ML à la baseline surface (`area_m2` / m² par place) dans `make-training-run` / `benchmark_report.md`.",
            "",
        ]
    )

    if reasons_block:
        lines.append("### Raisons de prudence / blocage")
        lines.extend([f"- {r}" for r in reasons_block])
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md
