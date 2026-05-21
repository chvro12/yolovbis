"""Prépare un export vet-clinic (CSV `;`, cp1252) pour `parking-capacity run`.

- Filtre les lignes dont `Name` commence par ``ANNULER`` (cliniques annulées).
- Drop les rues sans numéro en tête (``PL DE L'EGLISE``, ``LIEU DIT LE BOITIER``, ``RTE D'ALENCON`` …) :
  la BAN les géocode mal et le pipeline gaspille du temps dessus.
- Compose une colonne ``adresse`` = ``<Address line 1>, <Postal Code> <City>``.
- Drop les sentinelles ``NOT DEFINED IN ORACLE`` dans ``Address line 2``.
- Re-encode en UTF-8 et préserve l'ID (clinic_id) pour la jointure finale.

Usage :
    python scripts/clean_vet_csv.py <input.csv> <output.csv> [--encoding cp1252]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ADDRESS_COL = "Address: Address line 1"
LINE2_COL = "Address line 2"
POSTAL_COL = "Postal Code"
CITY_COL = "City"
NAME_COL = "Name"
ID_COL = "ID"

SENTINELS = {"NOT DEFINED IN ORACLE", ""}


def _norm(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s in SENTINELS else s


def clean(input_path: Path, output_path: Path, *, encoding: str = "cp1252") -> dict:
    raw = pd.read_csv(input_path, sep=";", encoding=encoding, dtype=str, keep_default_na=False)
    stats = {"input_rows": len(raw)}

    # Travaille sur un DataFrame compact dérivé : évite les pièges d'alignement entre Series et df filtré.
    work = pd.DataFrame(
        {
            "clinic_id": raw[ID_COL],
            "clinic_name": raw[NAME_COL],
            "street": raw[ADDRESS_COL].map(_norm),
            "postal_raw": raw[POSTAL_COL].map(_norm),
            "city": raw[CITY_COL].map(_norm),
        }
    )

    # 1. Filtre ANNULER (insensible à la casse, début de chaîne).
    annul_mask = work["clinic_name"].fillna("").str.upper().str.strip().str.startswith("ANNULER")
    stats["dropped_annuler"] = int(annul_mask.sum())
    work = work[~annul_mask]

    # 2. Drop les lignes sans rue ou sans ville (rien à géocoder).
    empty_mask = (work["street"] == "") | (work["city"] == "")
    stats["dropped_empty"] = int(empty_mask.sum())
    work = work[~empty_mask]

    # 3. Drop les rues sans numéro en tête (« PL DE L'EGLISE », « LIEU DIT LE BOITIER », « RTE D'ALENCON »).
    #    Why: la BAN les géocode mal ou retombe sur le centre de la commune, ce qui pollue les résultats.
    #    Pattern : commence par un ou plusieurs chiffres ; « 6A », « 175 B », « 1796 » sont OK.
    no_number_mask = ~work["street"].str.match(r"^\d+")
    stats["dropped_no_street_number"] = int(no_number_mask.sum())
    work = work[~no_number_mask]

    # 4. Compose l'adresse finale : « <street>, <postal> <city> ».
    postal = work["postal_raw"].where(work["postal_raw"] == "", work["postal_raw"].str.zfill(5))
    adresse = (
        work["street"] + ", " + postal + " " + work["city"]
    ).str.strip().str.replace(r"\s+", " ", regex=True)

    df_out = pd.DataFrame(
        {
            "clinic_id": work["clinic_id"].values,
            "clinic_name": work["clinic_name"].values,
            "adresse": adresse.values,
            "postal_code": postal.values,
            "city": work["city"].values,
        }
    )

    # 5. Dédoublonnage strict sur adresse (garde la 1ère occurrence).
    before = len(df_out)
    df_out = df_out.drop_duplicates(subset=["adresse"], keep="first").reset_index(drop=True)
    stats["dropped_duplicate_address"] = before - len(df_out)
    stats["output_rows"] = len(df_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--encoding", default="cp1252")
    args = p.parse_args(argv)
    stats = clean(args.input, args.output, encoding=args.encoding)
    print("Fichier nettoyé :", args.output)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
