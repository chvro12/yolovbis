"""Typologie d'un site → fourchette de capacité parking plausible + facteurs d'occupation.

Combine 2 sources :
1. **APE INSEE** (code NAF) via ``sirene_client.search_sirene()``.
2. **Tags OSM** (amenity / shop / office / building) déjà récupérés par Overpass.

Mapping → ``SiteTypology``  qui contient :
- ``family`` : grande famille (clinic, supermarket, hospital, school, residential, office, …)
- ``expected_capacity_min`` / ``_max`` : fourchette de capacité plausible (places)
- ``expected_occupation_rate`` : taux d'occupation typique en journée (0-1)
- ``confidence`` : qualité du match (none | weak | medium | strong)

Cette information sert ensuite à **calibrer `best_effort_estimate`** :
- borne inférieure = max(observed_vehicle_floor, typology_min)
- borne supérieure = min(plausible_ceiling, typology_max)
- valeur centrale = max(vehicles / occupation_rate, typology_min)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class SiteTypology:
    """Typologie estimée d'un site avec contraintes capacité."""

    family: str = "unknown"
    label: str = ""                      # libellé lisible humain
    expected_capacity_min: Optional[int] = None
    expected_capacity_max: Optional[int] = None
    expected_occupation_rate: float = 0.5   # 0-1, fraction du parking occupée en journée
    confidence: str = "none"                # none | weak | medium | strong
    sources: list[str] = field(default_factory=list)
    raw_ape: Optional[str] = None
    raw_amenity: Optional[str] = None


# ===== Table APE (NAF rev. 2) → typologie =====
# Fourchettes basées sur ratios PLU typiques + observation terrain.
# Format : code_prefix → (family, label, min, max, occupation_rate)

APE_TABLE: list[Tuple[str, str, str, Optional[int], Optional[int], float]] = [
    # === Santé ===
    ("75.00", "vet_clinic",      "Clinique vétérinaire",        3,   20, 0.40),  # 75.00Z
    ("86.10", "hospital_large",  "Hôpital",                    100,  800, 0.75),  # 86.10Z
    ("86.21", "clinic_doctor",   "Cabinet médical",              5,   25, 0.50),  # 86.21Z gen. medicine
    ("86.22", "clinic_doctor",   "Cabinet médical spécialisé",   3,   20, 0.50),
    ("86.23", "clinic_dentist",  "Cabinet dentaire",             3,   15, 0.50),
    ("86.90", "clinic_paramed",  "Activité paramédicale",        2,   15, 0.45),
    ("87",    "ehpad",           "EHPAD / maison de retraite",  20,  100, 0.55),
    # === Commerce de détail ===
    # 47.11 (alimentaire) : tranches resserrées par observation terrain typique
    ("47.11A", "small_grocery",  "Supérette < 120 m²",           5,   20, 0.55),
    ("47.11B", "small_grocery",  "Supérette 120-400 m²",        10,   35, 0.55),
    ("47.11C", "supermarket",    "Supermarché 400-2500 m²",     25,   90, 0.55),
    ("47.11D", "supermarket",    "Supermarché 2500 m² env.",    40,  150, 0.55),
    ("47.11E", "hypermarket",    "Hypermarché 2500-5000 m²",   100,  500, 0.50),
    ("47.11F", "hypermarket",    "Hypermarché >= 5000 m²",     300, 1500, 0.50),
    ("47.19A", "department_store","Grand magasin non alimentaire", 80,  600, 0.45),
    ("47.19B", "department_store","Autres commerces magasin",     30,  200, 0.45),
    ("47.30", "fuel_station",    "Station-service",              5,   30, 0.30),
    ("47.7",  "small_shop",      "Petit commerce détail",        2,   15, 0.30),
    # === Restauration ===
    ("56.10A", "restaurant",     "Restaurant traditionnel",     10,   60, 0.55),
    ("56.10B", "restaurant",     "Cafétéria / libre-service",   15,   80, 0.55),
    ("56.10C", "fast_food",      "Restauration rapide",         15,  100, 0.60),
    ("56.21", "catering",        "Traiteur",                     3,   15, 0.30),
    ("56.30", "bar",             "Bar / café",                   3,   20, 0.40),
    # === Hébergement ===
    ("55.10", "hotel",           "Hôtel",                       20,  300, 0.55),
    ("55.20", "hotel",           "Hébergement touristique",     10,  150, 0.55),
    # === Bureau / administration ===
    ("70",    "office",          "Bureau / siège",              10,  150, 0.70),
    ("84.11", "admin_public",    "Administration publique",     10,  100, 0.65),
    ("84.12", "admin_public",    "Administration sectorielle",  20,  150, 0.65),
    ("64.",   "bank_finance",    "Banque / assurance",           5,   40, 0.55),
    ("65.",   "bank_finance",    "Assurance",                    5,   40, 0.55),
    ("66.",   "bank_finance",    "Auxiliaire finance",           5,   40, 0.55),
    # === Enseignement ===
    ("85.10", "school_primary",  "Maternelle / primaire",        5,   30, 0.20),
    ("85.20", "school_primary",  "Primaire",                     5,   30, 0.20),
    ("85.31", "school_secondary","Collège / lycée",             20,  100, 0.35),
    ("85.32", "school_secondary","Lycée technique",             20,  120, 0.35),
    ("85.4",  "school_higher",   "Enseignement supérieur",      50,  500, 0.50),
    ("85.5",  "training",        "Formation continue",          10,   60, 0.45),
    # === Industrie / logistique ===
    ("10.",   "industry_food",   "Industrie agroalimentaire",   20,  200, 0.60),
    ("28.",   "industry_mach",   "Fabrication machines",        20,  200, 0.65),
    ("29.",   "industry_auto",   "Industrie automobile",        30,  300, 0.65),
    ("52.10", "logistics_storage","Entreposage / stockage",      5,   80, 0.35),
    ("52.29", "logistics_aux",   "Auxiliaire transport",        10,  100, 0.40),
    # === Immobilier ===
    ("68.20A","residential",     "Location logement",            5,   40, 0.50),
    ("68.20B","residential",     "Location logement (immeuble)", 5,   40, 0.50),
    ("68.31", "real_estate",     "Agence immobilière",           2,   10, 0.40),
    # === Sport / loisirs ===
    ("93.11", "sport_facility",  "Installation sportive",       30,  300, 0.45),
    ("93.13", "sport_gym",       "Salle de fitness",            10,   60, 0.40),
    ("93.21", "amusement_park",  "Parc d'attractions",         100, 1500, 0.50),
]


# ===== Table OSM amenity/shop → typologie =====
OSM_AMENITY_TABLE: Dict[str, Tuple[str, str, Optional[int], Optional[int], float]] = {
    # amenity
    "hospital":     ("hospital_large", "Hôpital OSM",               100,  800, 0.75),
    "clinic":       ("clinic_doctor",  "Clinique OSM",                5,   30, 0.50),
    "doctors":      ("clinic_doctor",  "Cabinet médical OSM",         3,   15, 0.50),
    "dentist":      ("clinic_dentist", "Dentiste OSM",                3,   12, 0.50),
    "veterinary":   ("vet_clinic",     "Vétérinaire OSM",             3,   20, 0.40),
    "pharmacy":     ("pharmacy",       "Pharmacie",                   3,   15, 0.45),
    "school":       ("school_primary", "École OSM",                  10,   50, 0.25),
    "kindergarten": ("school_primary", "Maternelle OSM",              5,   20, 0.20),
    "university":   ("school_higher",  "Université OSM",            100,  800, 0.55),
    "restaurant":   ("restaurant",     "Restaurant OSM",             10,   60, 0.55),
    "fast_food":    ("fast_food",      "Fast-food OSM",              15,  100, 0.60),
    "cafe":         ("bar",            "Café OSM",                    3,   20, 0.40),
    "bar":          ("bar",            "Bar OSM",                     3,   20, 0.40),
    "fuel":         ("fuel_station",   "Station-service OSM",         5,   30, 0.30),
    "bank":         ("bank_finance",   "Banque OSM",                  3,   30, 0.50),
    "townhall":     ("admin_public",   "Mairie OSM",                 10,  100, 0.55),
    "post_office":  ("admin_public",   "Poste OSM",                   5,   30, 0.50),
    "library":      ("admin_public",   "Bibliothèque OSM",            5,   40, 0.45),
    "place_of_worship": ("place_of_worship", "Lieu culte OSM",       10,  200, 0.20),
    "cinema":       ("cinema",         "Cinéma OSM",                 50,  500, 0.40),
    "theatre":      ("cinema",         "Théâtre OSM",                30,  300, 0.40),
    # shop
    "supermarket":  ("supermarket",    "Supermarché OSM",           50,  300, 0.55),
    "mall":         ("hypermarket",    "Centre commercial OSM",    200, 1500, 0.50),
    "convenience":  ("small_grocery",  "Supérette OSM",              5,   30, 0.50),
    "department_store": ("department_store", "Grand magasin OSM",   80,  500, 0.45),
}


def _match_ape(ape: str) -> Optional[Tuple[str, str, Optional[int], Optional[int], float]]:
    """Match préfixe APE le plus long."""
    if not ape:
        return None
    # Cherche match exact d'abord, puis préfixe
    candidates = sorted(APE_TABLE, key=lambda r: -len(r[0]))
    for prefix, family, label, lo, hi, occ in candidates:
        if ape.upper().startswith(prefix.upper()):
            return (family, label, lo, hi, occ)
    return None


def _match_osm(amenity_or_shop: str) -> Optional[Tuple[str, str, Optional[int], Optional[int], float]]:
    if not amenity_or_shop:
        return None
    return OSM_AMENITY_TABLE.get(amenity_or_shop.lower())


def classify_site(
    *,
    ape_code: Optional[str] = None,
    osm_amenity: Optional[str] = None,
    osm_shop: Optional[str] = None,
    osm_office: Optional[str] = None,
    osm_building: Optional[str] = None,
) -> SiteTypology:
    """Combine APE + OSM tags → ``SiteTypology``.

    Priorité de confiance :
    - APE + OSM concordants → ``strong``
    - APE seul (officiel) → ``medium``
    - OSM seul → ``medium`` aussi (humain validé)
    - Aucun signal → ``none``
    """
    sources: list[str] = []
    ape_match = _match_ape(ape_code or "")
    osm_match = (
        _match_osm(osm_amenity or "")
        or _match_osm(osm_shop or "")
    )

    if ape_match and osm_match:
        # Si les deux concordent (même family), on combine
        if ape_match[0] == osm_match[0]:
            family, label, lo, hi, occ = ape_match
            sources = [f"ape:{ape_code}", f"osm:{osm_amenity or osm_shop}"]
            return SiteTypology(
                family=family, label=label,
                expected_capacity_min=lo, expected_capacity_max=hi,
                expected_occupation_rate=occ, confidence="strong",
                sources=sources, raw_ape=ape_code, raw_amenity=osm_amenity or osm_shop,
            )
        # Désaccord : on prend l'APE (plus officiel)
        family, label, lo, hi, occ = ape_match
        sources = [f"ape:{ape_code}", f"osm_divergent:{osm_amenity or osm_shop}"]
        return SiteTypology(
            family=family, label=label,
            expected_capacity_min=lo, expected_capacity_max=hi,
            expected_occupation_rate=occ, confidence="medium",
            sources=sources, raw_ape=ape_code, raw_amenity=osm_amenity or osm_shop,
        )

    if ape_match:
        family, label, lo, hi, occ = ape_match
        return SiteTypology(
            family=family, label=label,
            expected_capacity_min=lo, expected_capacity_max=hi,
            expected_occupation_rate=occ, confidence="medium",
            sources=[f"ape:{ape_code}"], raw_ape=ape_code,
        )

    if osm_match:
        family, label, lo, hi, occ = osm_match
        return SiteTypology(
            family=family, label=label,
            expected_capacity_min=lo, expected_capacity_max=hi,
            expected_occupation_rate=occ, confidence="medium",
            sources=[f"osm:{osm_amenity or osm_shop}"], raw_amenity=osm_amenity or osm_shop,
        )

    # Fallbacks office/building (très peu informatif)
    if osm_office:
        return SiteTypology(
            family="office", label=f"Bureau OSM ({osm_office})",
            expected_capacity_min=5, expected_capacity_max=100,
            expected_occupation_rate=0.60, confidence="weak",
            sources=[f"osm_office:{osm_office}"], raw_amenity=osm_office,
        )
    if osm_building in ("residential", "apartments", "house"):
        return SiteTypology(
            family="residential", label=f"Résidentiel OSM ({osm_building})",
            expected_capacity_min=2, expected_capacity_max=40,
            expected_occupation_rate=0.50, confidence="weak",
            sources=[f"osm_building:{osm_building}"], raw_amenity=osm_building,
        )

    return SiteTypology(family="unknown", label="non classifié", confidence="none")


def apply_typology_to_estimate(
    typology: SiteTypology,
    *,
    vehicle_count: int,
    plausible_ceiling: Optional[int],
) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    """Calibre best_effort selon la typologie.

    Retourne ``(estimate, min, max, rationale)``.

    Logique :
    - Si occupation rate connu et vehicles > 0 : estimate = vehicles / occupation_rate
    - Bornes : intersection(typology range, [vehicles, ceiling])
    """
    rationale_parts = [f"typology={typology.family}"]
    if typology.confidence == "none":
        return None, None, None, "no_typology"

    tlo = typology.expected_capacity_min
    thi = typology.expected_capacity_max
    occ = typology.expected_occupation_rate

    # Estimation centrale : par occupation observée
    if vehicle_count > 0 and 0.15 < occ < 0.95:
        est_from_occupation = max(vehicle_count, int(round(vehicle_count / occ)))
        rationale_parts.append(f"vehicles={vehicle_count}/occ={occ:.2f}")
    else:
        est_from_occupation = None

    # Bornes
    lo = max(vehicle_count, tlo) if tlo is not None else vehicle_count
    hi = thi if thi is not None else (plausible_ceiling or vehicle_count + 20)
    if plausible_ceiling is not None and thi is not None:
        # Si la zone bitumée éligible est très petite, on ne dépasse pas le ceiling.
        hi = min(hi, plausible_ceiling) if plausible_ceiling >= lo else hi

    # Valeur centrale
    if est_from_occupation is not None:
        center = est_from_occupation
    else:
        center = (lo + hi) // 2

    center = max(min(center, hi), lo)
    rationale_parts.append(f"range={tlo}-{thi}")
    return int(center), int(lo), int(hi), "+".join(rationale_parts)
