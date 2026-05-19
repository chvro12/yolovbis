"""Compare les prédictions du rerun (après fixes) à celles de la 1ère validation.

Lit :
- ``data/benchmark/rerun_after_fixes/results.csv`` (nouveau)
- ``data/benchmark/rerun_input.csv`` (vérité terrain + anciennes prédictions)

Produit un tableau diff par adresse + métriques globales.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _to_num(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bucket(true_cap: Optional[float]) -> str:
    if true_cap is None:
        return "?"
    t = int(true_cap)
    if t == 0:
        return "0"
    if 1 <= t <= 5:
        return "1-5"
    if 6 <= t <= 20:
        return "6-20"
    return "21+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rerun-results",
        type=Path,
        default=Path("data/benchmark/rerun_after_fixes_v2/results.csv"),
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark/rerun_input.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmark/rerun_after_fixes_v2/comparison.md"),
    )
    args = ap.parse_args()

    new = pd.read_csv(args.rerun_results)
    old = pd.read_csv(args.input)

    if "input_address" in new.columns:
        addr_col_new = "input_address"
    elif "address" in new.columns:
        addr_col_new = "address"
    else:
        raise SystemExit("Pas de colonne address dans results.csv")

    new = new.rename(columns={addr_col_new: "address"})
    df = old.merge(new, on="address", suffixes=("", "_new"), how="left")

    rows = []
    for _, r in df.iterrows():
        true_cap = _to_num(r.get("expected_capacity"))
        old_pred = _to_num(r.get("old_predicted"))
        new_pred = _to_num(r.get("estimated_capacity"))
        rows.append(
            {
                "address": r["address"],
                "true": true_cap,
                "old": old_pred,
                "new": new_pred,
                "old_err": abs(old_pred - true_cap) if old_pred is not None and true_cap is not None else None,
                "new_err": abs(new_pred - true_cap) if new_pred is not None and true_cap is not None else None,
                "old_src": r.get("old_primary_source"),
                "new_src": r.get("primary_source"),
                "new_conf": r.get("primary_confidence"),
                "new_vehicle_method": r.get("vehicle_detection_method"),
                "ceiling_new": _to_num(r.get("plausible_capacity_ceiling")),
                "nearby_pub_new": _to_num(r.get("nearby_public_capacity_estimate")),
                "consistency_max_severity": r.get("consistency_max_severity"),
                "consistency_high_count": _to_num(r.get("consistency_high_count")),
                "consistency_needs_review": r.get("consistency_needs_review"),
                "bucket": _bucket(true_cap),
            }
        )
    res = pd.DataFrame(rows)

    # Métriques globales (sur lignes avec true + pred non null des DEUX côtés)
    both = res.dropna(subset=["old", "new", "true"])
    n_both = len(both)
    if n_both > 0:
        old_mae = float(both["old_err"].mean())
        new_mae = float(both["new_err"].mean())
        old_acc5 = float((both["old_err"] <= 5).mean())
        new_acc5 = float((both["new_err"] <= 5).mean())
        delta_mae = new_mae - old_mae
        improved = int(((both["new_err"] + 0.5) < both["old_err"]).sum())
        worsened = int(((both["new_err"] - 0.5) > both["old_err"]).sum())
        unchanged = n_both - improved - worsened
    else:
        old_mae = new_mae = old_acc5 = new_acc5 = delta_mae = float("nan")
        improved = worsened = unchanged = 0

    # Stratifié par bucket
    bucket_rows = []
    for b in ["0", "1-5", "6-20", "21+"]:
        sub = both[both["bucket"] == b]
        if sub.empty:
            continue
        bucket_rows.append(
            {
                "bucket": b,
                "n": len(sub),
                "old_mae": float(sub["old_err"].mean()),
                "new_mae": float(sub["new_err"].mean()),
                "delta": float(sub["new_err"].mean() - sub["old_err"].mean()),
            }
        )

    # Top 4 anciens pires cas
    worst_old = both.nlargest(4, "old_err")[
        ["address", "true", "old", "new", "old_err", "new_err", "new_src", "ceiling_new"]
    ]

    # Cas devenus pires
    regressions = both[
        (both["new_err"] > both["old_err"] + 1) & (both["old_err"].notna())
    ].sort_values("new_err", ascending=False)

    # Coverage flags
    has_flag = res["consistency_max_severity"].fillna("none") != "none"
    n_with_flag = int(has_flag.sum())
    n_review = int((res["consistency_needs_review"].fillna(False).astype(str).str.lower() == "true").sum())

    lines = []
    lines.append("# Comparaison rerun vs validation manuelle initiale\n")
    lines.append(f"- Adresses comparables (pred old + pred new + true) : **{n_both}**")
    lines.append(f"- **MAE avant fixes : {old_mae:.3f}**")
    lines.append(f"- **MAE après fixes : {new_mae:.3f}**")
    if not math.isnan(delta_mae):
        sign = "↓" if delta_mae < 0 else "↑"
        lines.append(f"- **Δ MAE : {delta_mae:+.3f} {sign}**")
    lines.append(f"- Acc ±5 avant : {old_acc5*100:.1f}% → après : {new_acc5*100:.1f}%")
    lines.append(f"- Améliorées : **{improved}** / Aggravées : **{worsened}** / Inchangées : {unchanged}")
    lines.append("")
    lines.append(f"- Adresses avec ≥1 flag de cohérence : **{n_with_flag}** / {len(res)}")
    lines.append(f"- Adresses needs_review (≥1 flag high) : **{n_review}**")
    lines.append("")

    lines.append("## MAE stratifiée par taille réelle")
    lines.append("")
    lines.append("| Bucket | n | MAE avant | MAE après | Δ |")
    lines.append("|---|---|---|---|---|")
    for br in bucket_rows:
        lines.append(
            f"| {br['bucket']} | {br['n']} | {br['old_mae']:.2f} | {br['new_mae']:.2f} | {br['delta']:+.2f} |"
        )
    lines.append("")

    lines.append("## Top 4 anciens pires cas — devenus ?")
    lines.append("")
    lines.append("| Adresse | vrai | ancien | nouveau | err_old | err_new | src_new | ceiling_new |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in worst_old.iterrows():
        new_v = "" if r["new"] is None or math.isnan(r["new"]) else str(int(r["new"]))
        new_e = "" if r["new_err"] is None or math.isnan(r["new_err"]) else f"{r['new_err']:.0f}"
        ceil_v = "" if r["ceiling_new"] is None or math.isnan(r["ceiling_new"]) else str(int(r["ceiling_new"]))
        lines.append(
            f"| {r['address']} | {int(r['true'])} | {int(r['old'])} | {new_v} | "
            f"{r['old_err']:.0f} | {new_e} | {r['new_src']} | {ceil_v} |"
        )
    lines.append("")

    if not regressions.empty:
        lines.append(f"## Régressions ({len(regressions)} cas)")
        lines.append("")
        lines.append("| Adresse | vrai | ancien | nouveau | err_old | err_new | src_new | flag |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in regressions.head(10).iterrows():
            lines.append(
                f"| {r['address']} | {int(r['true'])} | {int(r['old'])} | {int(r['new'])} | "
                f"{r['old_err']:.0f} | {r['new_err']:.0f} | {r['new_src']} | "
                f"{r['consistency_max_severity'] or '—'} |"
            )
        lines.append("")

    # CSV diff aussi
    res.to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.csv')}")
    print()
    print("\n".join(lines[:25]))


if __name__ == "__main__":
    main()
