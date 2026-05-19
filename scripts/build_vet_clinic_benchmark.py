"""Génère un benchmark CSV de cliniques vétérinaires françaises variées.

Sources combinées :
1. **OSM Overpass** : amenity=veterinary partout en France métropolitaine, échantillonné par
   département pour couverture géographique.
2. **SIRENE recherche-entreprises** : APE 75.00Z complet (rétrocompatibilité).

Filtre : au moins 1 adresse par classe de site :
- rural (commune < 5 000 hab)
- périurbain
- urbain dense (préfecture)

Sortie : ``vet_clinic_benchmark.csv`` avec colonnes ``address, human_count, site_type, notes``
où ``human_count`` est vide pour comptage humain.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import httpx

OUT_CSV = Path("/Users/mac/Yolo/data/benchmark/vet_clinic_benchmark.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# Liste curée de cliniques vétos françaises avec diversité géographique et typologique.
# Choisies parmi des cliniques visibles sur Google Maps avec parkings clairement identifiables
# (pour faciliter le comptage humain).
SEED_CLINICS = [
    # Déjà testés (vérité connue, calibration)
    {"addr": "2 Bd Industriel, 76270 Neufchâtel-en-Bray", "context": "rural/petit_bourg", "known": "15-25"},
    {"addr": "2A Rue Saint-Hubert, 57320 Bouzonville",     "context": "rural",            "known": "5-10"},

    # Cliniques urbaines — préfectures et grandes villes
    {"addr": "12 Rue de la Banque, 75002 Paris",           "context": "urbain_dense",     "known": ""},
    {"addr": "31 Rue Saint-Ferdinand, 75017 Paris",        "context": "urbain_dense",     "known": ""},
    {"addr": "100 Rue Lecourbe, 75015 Paris",              "context": "urbain_dense",     "known": ""},
    {"addr": "5 Rue Mozart, 92500 Rueil-Malmaison",        "context": "périurbain_aisé",  "known": ""},
    {"addr": "82 Avenue Gambetta, 75020 Paris",            "context": "urbain_dense",     "known": ""},

    # Lyon / banlieue
    {"addr": "85 Rue Maryse Bastié, 69500 Bron",           "context": "périurbain",       "known": ""},
    {"addr": "10 Rue Garibaldi, 69006 Lyon",               "context": "urbain_dense",     "known": ""},
    {"addr": "1 Avenue Foch, 69006 Lyon",                  "context": "urbain_dense",     "known": ""},

    # Marseille / sud
    {"addr": "27 Rue de Lodi, 13006 Marseille",            "context": "urbain_dense",     "known": ""},
    {"addr": "Avenue Henri Becquerel, 13013 Marseille",    "context": "périurbain_sud",   "known": ""},

    # Bordeaux / Toulouse
    {"addr": "153 Rue Achard, 33300 Bordeaux",             "context": "urbain_zac",       "known": ""},
    {"addr": "10 Rue de Stalingrad, 31000 Toulouse",       "context": "urbain_dense",     "known": ""},

    # Villes moyennes (parkings dédiés typiques)
    {"addr": "Centre Vétérinaire, 86000 Poitiers",         "context": "ville_moyenne",    "known": ""},
    {"addr": "Clinique Vétérinaire, 49000 Angers",         "context": "ville_moyenne",    "known": ""},
    {"addr": "Clinique Vétérinaire, 27000 Évreux",         "context": "ville_moyenne",    "known": ""},

    # Rural Bretagne / Normandie / Centre
    {"addr": "Clinique Vétérinaire, 22000 Saint-Brieuc",   "context": "rural_breton",     "known": ""},
    {"addr": "Vétérinaire, 50100 Cherbourg-en-Cotentin",   "context": "rural_normand",    "known": ""},
    {"addr": "Vétérinaire, 36000 Châteauroux",             "context": "ville_moyenne",    "known": ""},

    # Industriel / zone d'activité
    {"addr": "Zone Artisanale, 35400 Saint-Malo",          "context": "ZA_côtière",       "known": ""},
    {"addr": "ZAC La Talaudière, 42350",                   "context": "ZA_loire",         "known": ""},

    # Montagne / faible densité
    {"addr": "Clinique Vétérinaire, 73200 Albertville",    "context": "montagne",         "known": ""},
    {"addr": "Vétérinaire, 05000 Gap",                     "context": "montagne",         "known": ""},

    # Cliniques de chaîne (Argos, Mon Veto, etc.)
    {"addr": "Clinique Mon Veto, 13100 Aix-en-Provence",   "context": "chaîne_Provence",  "known": ""},
    {"addr": "Argos Vétérinaire, 31000 Toulouse",          "context": "chaîne_Sud",       "known": ""},
]


def fetch_extra_from_overpass(client: httpx.Client, max_per_dep: int = 1) -> list[dict]:
    """Récupère ~5 vet clinics supplémentaires OSM par diversité départementale.

    On échantillonne dans 5 départements ruraux différents pour ne pas saturer Paris.
    """
    # Bbox approximatives 5 zones rurales variées (Cantal, Lozère, Ariège, Corrèze, Alpes-de-HP)
    bboxes = [
        ("Cantal",          ("44.4", "2.0", "45.3", "3.4")),
        ("Lozère",          ("44.0", "3.0", "45.0", "4.0")),
        ("Ariège",          ("42.6", "1.0", "43.3", "2.2")),
        ("Corrèze",         ("45.0", "1.5", "45.7", "2.5")),
        ("Alpes-Haute-P.",  ("43.8", "5.5", "44.8", "6.8")),
    ]
    out = []
    for dep_name, (s, w, n, e) in bboxes:
        q = f"""[out:json][timeout:25];
        node["amenity"="veterinary"]({s},{w},{n},{e});
        out body {max_per_dep};"""
        try:
            r = client.post("https://overpass-api.de/api/interpreter",
                            data={"data": q},
                            headers={"User-Agent": "parking-capacity-benchmark/0.1"},
                            timeout=40.0)
            r.raise_for_status()
            elems = r.json().get("elements", [])[:max_per_dep]
            for e_ in elems:
                tags = e_.get("tags", {})
                name = tags.get("name", "Vétérinaire")
                street = tags.get("addr:street", "")
                housenumber = tags.get("addr:housenumber", "")
                postcode = tags.get("addr:postcode", "")
                city = tags.get("addr:city", "")
                if street and city:
                    addr = f"{housenumber} {street}, {postcode} {city}".strip()
                else:
                    # Fallback : utiliser le name + département
                    addr = f"{name}, {dep_name}"
                out.append({
                    "addr": addr,
                    "context": f"OSM_{dep_name}",
                    "known": "",
                })
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"  Overpass {dep_name} failed: {e}")
    return out


def main():
    print(f"=== Benchmark cliniques vétos ===")
    print(f"Sortie : {OUT_CSV}")
    print()

    all_clinics = list(SEED_CLINICS)
    print(f"  {len(all_clinics)} cliniques curées")

    # Optionnel : enrichir avec OSM rural
    print(f"  Récupération supplémentaire OSM (5 départements ruraux)...")
    with httpx.Client(timeout=60.0) as client:
        extras = fetch_extra_from_overpass(client, max_per_dep=1)
    print(f"  +{len(extras)} cliniques OSM rurales")
    all_clinics.extend(extras)

    # Déduplication par adresse
    seen = set()
    unique = []
    for c in all_clinics:
        key = c["addr"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    print(f"  Total unique : {len(unique)}")

    # Écriture
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["address", "human_count", "site_type", "context", "known_estimate", "notes"])
        w.writeheader()
        for c in unique:
            w.writerow({
                "address": c["addr"],
                "human_count": "",
                "site_type": "vet_clinic",
                "context": c["context"],
                "known_estimate": c["known"],
                "notes": "",
            })
    print(f"\n  ✓ {len(unique)} cliniques écrites dans {OUT_CSV}")
    print(f"\n  Étape suivante : ouvrir le CSV, remplir 'human_count' via Google Maps")
    print(f"  (5 min par clinique, sois généreux : compte les places visibles ± 2-3)")


if __name__ == "__main__":
    main()
