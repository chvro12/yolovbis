"""Évaluation des comptages humains (manual_review.csv)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def run_evaluate_manual_review(input_csv: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    for col in ("human_count", "estimated_capacity"):
        if col not in df.columns:
            df[col] = np.nan

    def _to_num(x: Any) -> float | None:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    df["human_f"] = df["human_count"].map(_to_num)
    df["est_f"] = df["estimated_capacity"].map(_to_num)
    sub = df.dropna(subset=["human_f", "est_f"])
    sub = sub[sub["human_f"] >= 0]

    lines = [
        "# Évaluation inspection manuelle",
        "",
        f"- Lignes avec `human_count` renseigné : **{len(sub)}**",
        "",
    ]

    summary: Dict[str, Any] = {"n_pairs": int(len(sub))}

    if sub.empty:
        lines.append("_Aucune paire (human_count + estimated_capacity) à comparer._")
        summary["mae"] = None
    else:
        err = (sub["est_f"] - sub["human_f"]).abs()
        summary["mae"] = float(err.mean())
        summary["rmse"] = float(np.sqrt(((sub["est_f"] - sub["human_f"]) ** 2).mean()))
        summary["median_abs_error"] = float(err.median())
        summary["acc_pm2"] = float((err <= 2).mean())
        summary["acc_pm5"] = float((err <= 5).mean())
        summary["acc_pm10"] = float((err <= 10).mean())
        lines.extend(
            [
                "## Métriques",
                "",
                f"- **MAE** : {summary['mae']:.3f}",
                f"- **RMSE** : {summary['rmse']:.3f}",
                f"- **Médiane erreur absolue** : {summary['median_abs_error']:.3f}",
                f"- **Taux ±2 places** : {summary['acc_pm2']:.2%}",
                f"- **Taux ±5 places** : {summary['acc_pm5']:.2%}",
                f"- **Taux ±10 places** : {summary['acc_pm10']:.2%}",
                "",
                "## Erreurs (pire → meilleur)",
                "",
            ]
        )
        sub2 = sub.assign(abs_error=(sub["est_f"] - sub["human_f"]).abs()).sort_values("abs_error", ascending=False)
        for _, r in sub2.iterrows():
            lines.append(
                f"- `{r.get('address', '')}` : estimé **{r['est_f']:.0f}** vs humain **{r['human_f']:.0f}** "
                f"(|Δ|={r['abs_error']:.1f})"
            )
        lines.append("")

    (out_dir / "manual_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out_dir / "manual_eval_report.md"
