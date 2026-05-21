"""Projette la sortie de ``parking-capacity run`` sur 4 colonnes lisibles.

Colonnes produites :
    - clinic_id, clinic_name, adresse  (rejoint depuis le CSV d'entrée nettoyé)
    - capacite_estimee         : nombre de places prédit
    - capacite_intervalle      : « min-max » (ex. ``8-14``) ou vide si non disponible
    - parking_a_cote           : oui / non — au moins un parking OSM autour du point
    - nb_places_parking_osm    : somme des places OSM connues (capacity OSM + parking_space comptés)
    - methode, confiance       : pour pouvoir filtrer/auditer en aval

Usage :
    python scripts/summarize_bulk_output.py <run_output.csv> <input_clean.csv> <summary.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _interval(row: pd.Series) -> str:
    lo = row.get("min_capacity")
    hi = row.get("max_capacity")
    if pd.isna(lo) or pd.isna(hi):
        return ""
    try:
        lo_i, hi_i = int(float(lo)), int(float(hi))
    except (TypeError, ValueError):
        return ""
    if lo_i == hi_i:
        return str(lo_i)
    return f"{lo_i}-{hi_i}"


def _capacity(row: pd.Series) -> str:
    # On préfère estimated_capacity (sortie consolidée). Fallback : primary_capacity.
    for col in ("estimated_capacity", "primary_capacity"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ("", "nan", "None"):
            try:
                return str(int(float(val)))
            except (TypeError, ValueError):
                continue
    return ""


def _osm_count(row: pd.Series) -> int:
    # Places OSM connues : on prend le max entre les compteurs parcelle / buffer / parking_space.
    candidates = []
    for col in (
        "capacity_osm_parcelle",
        "capacity_osm_buffer",
        "osm_parking_space_count",
    ):
        val = row.get(col)
        try:
            candidates.append(int(float(val)) if pd.notna(val) else 0)
        except (TypeError, ValueError):
            continue
    return max(candidates) if candidates else 0


def summarize(run_csv: Path, input_csv: Path, output_csv: Path) -> dict:
    run_df = pd.read_csv(run_csv)
    in_df = pd.read_csv(input_csv)

    # Le run préserve l'ordre du CSV d'entrée via source_row_index → jointure stable.
    if "source_row_index" in run_df.columns:
        run_df = run_df.sort_values("source_row_index").reset_index(drop=True)
    in_df = in_df.reset_index(drop=True)
    n = min(len(in_df), len(run_df))
    in_df = in_df.iloc[:n].copy()
    run_df = run_df.iloc[:n].copy()

    out = pd.DataFrame(
        {
            "clinic_id": in_df.get("clinic_id", pd.Series([""] * n)),
            "clinic_name": in_df.get("clinic_name", pd.Series([""] * n)),
            "adresse": in_df["adresse"],
            "capacite_estimee": run_df.apply(_capacity, axis=1),
            "capacite_intervalle": run_df.apply(_interval, axis=1),
            "parking_a_cote": run_df["nearby_osm_parkings_count"].fillna(0).astype(int).gt(0).map({True: "oui", False: "non"}),
            "nb_places_parking_osm": run_df.apply(_osm_count, axis=1),
            "methode": run_df.get("method_used", pd.Series([""] * n)).fillna(""),
            "confiance": run_df.get("primary_confidence", pd.Series([""] * n)).fillna(""),
            "ban_score": run_df.get("ban_score", pd.Series([""] * n)),
            "error": run_df.get("error", pd.Series([""] * n)).fillna(""),
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")

    stats = {
        "rows": len(out),
        "with_estimate": int((out["capacite_estimee"] != "").sum()),
        "with_nearby_parking": int((out["parking_a_cote"] == "oui").sum()),
        "errors": int((out["error"] != "").sum()),
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_csv", type=Path, help="Sortie brute de `parking-capacity run`")
    p.add_argument("input_csv", type=Path, help="CSV nettoyé en entrée du run (pour clinic_id / clinic_name)")
    p.add_argument("output_csv", type=Path, help="CSV final projeté")
    args = p.parse_args(argv)
    stats = summarize(args.run_csv, args.input_csv, args.output_csv)
    print("Résumé écrit :", args.output_csv)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
