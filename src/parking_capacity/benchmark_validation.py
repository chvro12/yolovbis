"""Validation terrain robuste : MAE / MAPE / R² + bootstrap CI + segmentation par typologie.

Phase 4 du plan : transformer un prototype en produit.

Format d'entrée attendu :

    address,human_count[,notes][,site_type]
    "2 Bd Industriel, 76270 Neufchâtel-en-Bray",20,,
    "10 rue de la Paix, 75002 Paris",,parking commerce,
    ...

Format de sortie :

- ``validation_summary.json`` : agrégats globaux + intervalles bootstrap.
- ``validation_segments.json`` : métriques par typologie / confidence / mode visuel.
- ``validation_report.md`` : rapport markdown lisible.
- ``per_address.csv`` : ligne par adresse avec erreur, ratios, raisons.

Métriques :
- **MAE** (Mean Absolute Error) : moyenne |estimé − humain|.
- **MAPE** (Mean Absolute Percentage Error) : moyenne |estimé − humain| / max(humain, 1).
- **RMSE** (Root Mean Square Error) : racine moyenne (estimé − humain)².
- **R²** (coefficient de détermination) : 1 − SS_res / SS_tot.
- **acc_±N** : fraction des cas avec |erreur| ≤ N places (N = 1, 3, 5, 10).
- **Refusal rate** : fraction d'adresses où le système a refusé (None).
- **Bootstrap CI 95 %** sur MAE et MAPE (1000 resamples).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ACCURACY_THRESHOLDS = (1, 3, 5, 10, 20)
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 1234


@dataclass
class ValidationMetrics:
    """Métriques calculées sur un échantillon avec valeurs humaines."""

    n_addresses_total: int = 0
    n_predicted: int = 0
    n_refused: int = 0
    refusal_rate: float = 0.0
    mae: Optional[float] = None
    mape: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    accuracy_pm: Dict[str, float] = field(default_factory=dict)
    mae_ci95: Optional[Tuple[float, float]] = None
    mape_ci95: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n_addresses_total": self.n_addresses_total,
            "n_predicted": self.n_predicted,
            "n_refused": self.n_refused,
            "refusal_rate": round(self.refusal_rate, 4),
            "mae": None if self.mae is None else round(self.mae, 3),
            "mape": None if self.mape is None else round(self.mape, 4),
            "rmse": None if self.rmse is None else round(self.rmse, 3),
            "r2": None if self.r2 is None else round(self.r2, 4),
            "accuracy_pm": {k: round(v, 4) for k, v in self.accuracy_pm.items()},
        }
        if self.mae_ci95 is not None:
            out["mae_ci95"] = [round(self.mae_ci95[0], 3), round(self.mae_ci95[1], 3)]
        if self.mape_ci95 is not None:
            out["mape_ci95"] = [round(self.mape_ci95[0], 4), round(self.mape_ci95[1], 4)]
        return out


def _bootstrap_ci(values: np.ndarray, stat_fn, *, n: int = BOOTSTRAP_SAMPLES,
                  seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> Tuple[float, float]:
    """Intervalle de confiance bootstrap (percentile)."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, values.size, values.size)
        s = stat_fn(values[idx])
        if math.isfinite(s):
            stats.append(s)
    if not stats:
        return (float("nan"), float("nan"))
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def compute_metrics(
    estimates: List[Optional[float]],
    humans: List[Optional[float]],
    *,
    thresholds: Tuple[int, ...] = ACCURACY_THRESHOLDS,
    with_bootstrap: bool = True,
) -> ValidationMetrics:
    """Calcule MAE/MAPE/RMSE/R²/accuracy à partir de deux listes alignées.

    - ``None`` ou NaN dans ``estimates`` → refus (compté dans refusal_rate, exclu des métriques).
    - ``None`` ou NaN dans ``humans`` → exclu complètement (pas de vérité terrain).
    """
    n_total = sum(1 for h in humans if h is not None and not (isinstance(h, float) and math.isnan(h)))
    pairs = []
    n_refused = 0
    for est, hum in zip(estimates, humans):
        if hum is None or (isinstance(hum, float) and math.isnan(hum)):
            continue
        if est is None or (isinstance(est, float) and math.isnan(est)):
            n_refused += 1
            continue
        pairs.append((float(est), float(hum)))

    m = ValidationMetrics(
        n_addresses_total=n_total,
        n_predicted=len(pairs),
        n_refused=n_refused,
        refusal_rate=(n_refused / n_total) if n_total > 0 else 0.0,
    )

    if not pairs:
        return m

    est_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    hum_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    err = est_arr - hum_arr
    abs_err = np.abs(err)

    m.mae = float(np.mean(abs_err))
    m.rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = np.maximum(hum_arr, 1.0)
    m.mape = float(np.mean(abs_err / denom))

    # R²
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((hum_arr - np.mean(hum_arr)) ** 2))
    m.r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    # Accuracy ±N
    for t in thresholds:
        m.accuracy_pm[f"pm{t}"] = float(np.mean(abs_err <= t))

    if with_bootstrap and len(pairs) >= 4:
        m.mae_ci95 = _bootstrap_ci(abs_err, lambda v: float(np.mean(v)))
        m.mape_ci95 = _bootstrap_ci(abs_err / denom, lambda v: float(np.mean(v)))

    return m


def segment_metrics(
    df: pd.DataFrame,
    *,
    estimate_col: str = "estimated_capacity",
    human_col: str = "human_count",
    segment_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Calcule les métriques globales + par valeur de chaque ``segment_cols``."""
    estimates = df[estimate_col].tolist() if estimate_col in df.columns else [None] * len(df)
    humans = df[human_col].tolist() if human_col in df.columns else [None] * len(df)

    out: Dict[str, Dict[str, Any]] = {}
    out["__overall__"] = compute_metrics(estimates, humans).to_dict()

    if segment_cols is None:
        segment_cols = []

    for col in segment_cols:
        if col not in df.columns:
            continue
        seg: Dict[str, Any] = {}
        for val, sub in df.groupby(df[col].fillna("__none__")):
            est_sub = sub[estimate_col].tolist() if estimate_col in sub.columns else [None] * len(sub)
            hum_sub = sub[human_col].tolist() if human_col in sub.columns else [None] * len(sub)
            seg[str(val)] = compute_metrics(est_sub, hum_sub).to_dict()
        out[col] = seg
    return out


def identify_failing_cases(
    df: pd.DataFrame,
    *,
    estimate_col: str = "estimated_capacity",
    human_col: str = "human_count",
    address_col: str = "address",
    error_threshold: float = 15.0,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Retourne les cas avec |erreur| max, classés."""
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        est = r.get(estimate_col)
        hum = r.get(human_col)
        if est is None or hum is None:
            continue
        try:
            est_f = float(est)
            hum_f = float(hum)
        except (TypeError, ValueError):
            continue
        if math.isnan(est_f) or math.isnan(hum_f):
            continue
        err = est_f - hum_f
        rows.append({
            "address": str(r.get(address_col, "")),
            "estimated": est_f,
            "human": hum_f,
            "abs_error": float(abs(err)),
            "rel_error": float(abs(err) / max(hum_f, 1.0)),
        })
    rows.sort(key=lambda x: -x["abs_error"])
    return [r for r in rows[:top_k] if r["abs_error"] >= error_threshold]


def write_validation_report(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    estimate_col: str = "estimated_capacity",
    human_col: str = "human_count",
    address_col: str = "address",
    segment_cols: Optional[List[str]] = None,
) -> Path:
    """Écrit JSON + markdown + per_address.csv. Retourne le chemin du rapport markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if segment_cols is None:
        segment_cols = [c for c in (
            "site_type", "parking_visual_mode", "semantic_confidence",
            "primary_source", "primary_confidence",
        ) if c in df.columns]

    seg_metrics = segment_metrics(df, estimate_col=estimate_col, human_col=human_col, segment_cols=segment_cols)
    overall = seg_metrics["__overall__"]
    failing = identify_failing_cases(df, estimate_col=estimate_col, human_col=human_col, address_col=address_col)

    # Persist JSON
    (out_dir / "validation_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "validation_segments.json").write_text(
        json.dumps(seg_metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "validation_failing_cases.json").write_text(
        json.dumps(failing, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # per_address CSV
    keep_cols = [c for c in (address_col, human_col, estimate_col, "min_capacity", "max_capacity",
                             "parking_visual_mode", "semantic_confidence",
                             "primary_source", "primary_confidence",
                             "site_type", "refuse_prediction", "refuse_prediction_reason") if c in df.columns]
    df[keep_cols].to_csv(out_dir / "per_address.csv", index=False)

    # Markdown
    md_path = out_dir / "validation_report.md"
    md = _render_markdown(overall, seg_metrics, failing, segment_cols)
    md_path.write_text(md, encoding="utf-8")
    return md_path


def load_front_validation_csv(input_csv: Path) -> pd.DataFrame:
    """Charge le CSV produit par l'UI et le normalise au format validation.

    L'UI écrit ``predicted_capacity`` / ``true_capacity`` / ``input_address``.
    Le moteur de rapport attend ``estimated_capacity`` / ``human_count`` / ``address``.
    """
    df = pd.read_csv(input_csv)
    rename = {
        "input_address": "address",
        "predicted_capacity": "estimated_capacity",
        "predicted_min": "min_capacity",
        "predicted_max": "max_capacity",
        "true_capacity": "human_count",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "address" not in df.columns:
        raise ValueError("CSV front invalide : colonne input_address/address absente.")
    if "human_count" not in df.columns:
        raise ValueError("CSV front invalide : colonne true_capacity/human_count absente.")
    if "estimated_capacity" not in df.columns:
        df["estimated_capacity"] = np.nan

    for col in ("estimated_capacity", "human_count", "min_capacity", "max_capacity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "abs_error" not in df.columns:
        df["abs_error"] = (df["estimated_capacity"] - df["human_count"]).abs()
    else:
        df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
        missing = df["abs_error"].isna() & df["estimated_capacity"].notna() & df["human_count"].notna()
        df.loc[missing, "abs_error"] = (
            df.loc[missing, "estimated_capacity"] - df.loc[missing, "human_count"]
        ).abs()
    df["signed_error"] = df["estimated_capacity"] - df["human_count"]
    return df


def write_front_validation_report(input_csv: Path, out_dir: Path) -> Path:
    """Écrit un rapport de validation à partir du CSV d'annotations de l'UI."""
    df = load_front_validation_csv(input_csv)
    segment_cols = [
        c for c in (
            "primary_source",
            "primary_confidence",
            "method_used",
            "verdict",
            "parcel_ok",
            "private_boundary_ok",
            "correction_reason",
            "vehicle_detection_method",
            "slot_detection_method",
        )
        if c in df.columns
    ]
    return write_validation_report(
        df,
        out_dir,
        estimate_col="estimated_capacity",
        human_col="human_count",
        address_col="address",
        segment_cols=segment_cols,
    )


def _format_metric_row(label: str, m: Dict[str, Any]) -> str:
    n = m.get("n_predicted", 0)
    nt = m.get("n_addresses_total", 0)
    mae = m.get("mae")
    mape = m.get("mape")
    r2 = m.get("r2")
    refus = m.get("refusal_rate", 0.0)
    acc5 = (m.get("accuracy_pm") or {}).get("pm5")
    parts = [
        f"`{label}`",
        f"{n}/{nt}",
        f"{refus*100:.0f}%",
        "–" if mae is None else f"{mae:.2f}",
        "–" if mape is None else f"{mape*100:.1f}%",
        "–" if r2 is None else f"{r2:.2f}",
        "–" if acc5 is None else f"{acc5*100:.0f}%",
    ]
    return "| " + " | ".join(parts) + " |"


def _render_markdown(overall: Dict[str, Any], seg: Dict[str, Dict[str, Any]],
                     failing: List[Dict[str, Any]], segment_cols: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Rapport de validation terrain")
    lines.append("")
    lines.append("## Vue d'ensemble")
    lines.append("")
    lines.append(f"- **Adresses avec comptage humain** : {overall.get('n_addresses_total', 0)}")
    lines.append(f"- **Prédictions retournées** : {overall.get('n_predicted', 0)}")
    lines.append(f"- **Refus système** : {overall.get('n_refused', 0)} ({overall.get('refusal_rate', 0)*100:.1f} %)")
    if overall.get("mae") is not None:
        ci = overall.get("mae_ci95")
        ci_str = f" (IC 95 % : {ci[0]:.2f}–{ci[1]:.2f})" if ci else ""
        lines.append(f"- **MAE** : {overall['mae']:.2f} places{ci_str}")
    if overall.get("mape") is not None:
        ci = overall.get("mape_ci95")
        ci_str = f" (IC 95 % : {ci[0]*100:.1f}–{ci[1]*100:.1f} %)" if ci else ""
        lines.append(f"- **MAPE** : {overall['mape']*100:.1f} %{ci_str}")
    if overall.get("rmse") is not None:
        lines.append(f"- **RMSE** : {overall['rmse']:.2f} places")
    if overall.get("r2") is not None:
        lines.append(f"- **R²** : {overall['r2']:.3f}")
    acc = overall.get("accuracy_pm") or {}
    if acc:
        for t in ACCURACY_THRESHOLDS:
            v = acc.get(f"pm{t}")
            if v is not None:
                lines.append(f"- **Accuracy ±{t}** : {v*100:.0f} %")

    lines.append("")
    lines.append("## Par segment")
    lines.append("")
    lines.append("Format : `n_predicted / n_total | refus | MAE | MAPE | R² | acc±5`")

    for col in segment_cols:
        if col not in seg:
            continue
        lines.append("")
        lines.append(f"### `{col}`")
        lines.append("")
        lines.append("| Valeur | n / total | refus | MAE | MAPE | R² | ±5 |")
        lines.append("|---|---|---|---|---|---|---|")
        for val, m in sorted(seg[col].items(), key=lambda kv: -(kv[1].get("n_predicted") or 0)):
            lines.append(_format_metric_row(val, m))

    if failing:
        lines.append("")
        lines.append("## Pires cas (|erreur| ≥ 15 places)")
        lines.append("")
        lines.append("| Adresse | humain | estimé | |err| | err rel. |")
        lines.append("|---|---|---|---|---|")
        for f in failing:
            lines.append(f"| {f['address']} | {f['human']:.0f} | {f['estimated']:.0f} | "
                         f"{f['abs_error']:.0f} | {f['rel_error']*100:.0f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Fichiers exportés :")
    lines.append("- `validation_summary.json` (agrégats globaux)")
    lines.append("- `validation_segments.json` (segmentation typologie/confiance)")
    lines.append("- `validation_failing_cases.json` (top cas en échec)")
    lines.append("- `per_address.csv` (1 ligne par adresse)")
    return "\n".join(lines) + "\n"
