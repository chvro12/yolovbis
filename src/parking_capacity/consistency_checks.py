"""Couche de cohérence : vérifie qu'une prédiction est compatible avec les signaux observés.

Indépendante de la précision du modèle : ne dit pas "la prédiction est fausse de X places",
dit "la prédiction est incohérente avec tel signal" (preuve secondaire qui devrait l'étayer
ou la contredire).

Chaque ``ConsistencyFlag`` porte :

- ``name`` : identifiant stable (utilisable pour filtrer en base / CSV).
- ``severity`` : ``info`` | ``medium`` | ``high``.
- ``reason`` : message lisible expliquant le déclenchement.

Un flag ``high`` signale une contradiction logique forte (le modèle prédit X mais aucun
signal ne soutient X). Un flag ``medium`` signale une incohérence probable. Un flag
``info`` est un avertissement contextuel utile au reviewer, pas une erreur.

Les seuils sont conservateurs : on préfère manquer une incohérence (faux négatif) que
flagger un cas légitime (faux positif). L'objectif est que ``high_count > 0`` soit un
signal fiable de "à reviewer en priorité".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


SEVERITY_RANK = {"info": 0, "medium": 1, "high": 2}


@dataclass
class ConsistencyFlag:
    name: str
    severity: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "severity": self.severity, "reason": self.reason}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Accès tolérant : ``RowResult`` dataclass ou ``dict`` plat."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def run_consistency_checks(res: Any) -> List[ConsistencyFlag]:
    """Évalue les règles de cohérence sur un ``RowResult`` (ou son ``to_flat_dict``)."""
    flags: List[ConsistencyFlag] = []

    estimated = _get(res, "estimated_capacity")
    if estimated is None:
        # Pas de prédiction → la couche est moins applicable, on retourne tôt.
        return flags

    vehicle_count = int(_get(res, "vehicle_count", 0) or 0)
    parking_area_m2 = _get(res, "parking_area_detected_m2") or _get(res, "area_total_m2", 0.0) or 0.0
    try:
        parking_area_m2 = float(parking_area_m2)
    except (TypeError, ValueError):
        parking_area_m2 = 0.0

    ratio_outside = _get(res, "parking_outside_buildings_ratio")
    typology_max = _get(res, "site_typology_max")
    typology_min = _get(res, "site_typology_min")
    typology_conf = _get(res, "site_typology_confidence")
    typology_family = _get(res, "site_typology_family") or "unknown"
    primary_source = _get(res, "primary_source") or ""
    primary_confidence = _get(res, "primary_confidence") or ""
    plausible_ceiling = _get(res, "plausible_capacity_ceiling")
    min_cap = _get(res, "min_capacity")
    max_cap = _get(res, "max_capacity")
    nearby_pub = _get(res, "nearby_public_capacity_estimate")
    slots_total = int(_get(res, "slots_total_count", 0) or 0)
    slot_method = _get(res, "slot_detection_method") or "none"
    n_parkings_parcelle = int(_get(res, "n_parkings_parcelle", 0) or 0)
    capacity_osm_parcelle = int(_get(res, "capacity_osm_parcelle", 0) or 0)

    # --- 1. Surface inventée : prédit > 0 mais ni véhicule ni surface réelle observée ---
    # Cas type : MONTAUBAN (predicted 13 vs vraie 0). Le modèle hallucine du bitume.
    if (
        estimated > 5
        and vehicle_count == 0
        and parking_area_m2 < 30.0
        and n_parkings_parcelle == 0
    ):
        flags.append(
            ConsistencyFlag(
                name="invented_surface",
                severity="high",
                reason=(
                    f"prédiction {estimated} places mais 0 véhicule détecté, "
                    f"{parking_area_m2:.0f} m² de surface parking et aucun parking OSM sur parcelle"
                ),
            )
        )

    # --- 2. Comptage de toits : prédiction grosse mais surface hors-bâti faible ---
    # Cas type : CERISE (predicted 39 vs vraie 20). Le modèle compte un toit comme bitume.
    if (
        estimated > 20
        and ratio_outside is not None
        and ratio_outside < 0.15
    ):
        flags.append(
            ConsistencyFlag(
                name="counting_buildings",
                severity="high",
                reason=(
                    f"prédiction {estimated} places mais seulement "
                    f"{ratio_outside*100:.0f}% de surface garable hors bâtiments — suspicion de toit compté"
                ),
            )
        )

    # --- 3. Typologie : prédiction très supérieure au plafond métier ---
    # SIRENE/OSM nous donne une fourchette par type d'établissement. Au-delà de 2× le max,
    # c'est très probablement une mauvaise parcelle ou un comptage du parking voisin.
    if (
        typology_max is not None
        and typology_conf in ("medium", "strong")
        and estimated > int(typology_max) * 2
    ):
        flags.append(
            ConsistencyFlag(
                name="typology_exceeded",
                severity="medium",
                reason=(
                    f"prédiction {estimated} places ≫ typologie {typology_family} "
                    f"(max attendu : {typology_max})"
                ),
            )
        )

    # --- 4. Typologie : prédiction très inférieure au plancher métier ---
    # Si le métier exige typiquement ≥ N places (clinique vétérinaire, supermarché),
    # une prédiction divisée par 3 est suspecte.
    if (
        typology_min is not None
        and typology_conf in ("medium", "strong")
        and int(typology_min) >= 5
        and estimated >= 0
        and estimated * 3 < int(typology_min)
    ):
        flags.append(
            ConsistencyFlag(
                name="typology_underestimated",
                severity="medium",
                reason=(
                    f"prédiction {estimated} places ≪ typologie {typology_family} "
                    f"(min attendu : {typology_min})"
                ),
            )
        )

    # --- 5. Capacité = 0 mais parking public voisin substantiel ---
    # Info utile au reviewer : le site n'a pas de parking dédié mais les clients ont
    # un grand parking public à côté. Pas une erreur ; un contexte.
    if (
        estimated == 0
        and isinstance(nearby_pub, (int, float))
        and nearby_pub >= 10
    ):
        flags.append(
            ConsistencyFlag(
                name="relies_on_nearby_public",
                severity="info",
                reason=(
                    f"pas de parking privé détecté, mais ~{int(nearby_pub)} places "
                    f"de parking public OSM à proximité"
                ),
            )
        )

    # --- 6. Confiance basse + grosse prédiction ---
    # private_marked_slots en `low` confidence avait MAE 8,7. Quand low + grand nombre,
    # on signale plus fortement que l'incertitude est élevée.
    if (
        primary_confidence in ("low", "weak")
        and estimated >= 15
    ):
        flags.append(
            ConsistencyFlag(
                name="low_confidence_large_value",
                severity="medium",
                reason=(
                    f"prédiction {estimated} places avec confiance `{primary_confidence}` — "
                    f"erreur attendue élevée d'après les segments de validation"
                ),
            )
        )

    # --- 7. Plafond saturé : la prédiction colle au ceiling avec min==max ---
    # Signal que la borne est contraignante : sans le plafond, le scénario voulait prédire
    # plus haut. Soit le plafond est bien calibré (cas attendu), soit il sous-estime un
    # vrai grand site (cas ALLONZIER pré-fix).
    if (
        plausible_ceiling is not None
        and min_cap is not None
        and max_cap is not None
        and estimated >= int(plausible_ceiling)
        and min_cap == max_cap == estimated
    ):
        flags.append(
            ConsistencyFlag(
                name="ceiling_saturated",
                severity="medium",
                reason=(
                    f"prédiction = plafond physique {plausible_ceiling} avec min=max — "
                    f"le scénario voulait probablement prédire au-delà"
                ),
            )
        )

    # --- 8. Source = marked_slots mais slots_total_count = 0 ---
    # Le primary_source affirme "places marquées" mais le détecteur de slots n'en a
    # trouvé aucune. Contradiction interne.
    if (
        primary_source == "private_marked_slots"
        and slots_total == 0
        and slot_method != "none"
        and estimated > 0
    ):
        flags.append(
            ConsistencyFlag(
                name="marked_slots_source_no_detection",
                severity="high",
                reason=(
                    f"source primaire `private_marked_slots` mais 0 place détectée "
                    f"par le détecteur (méthode `{slot_method}`)"
                ),
            )
        )

    # --- 9. Capacité OSM taggée sur parcelle ignorée ---
    # OSM a un tag capacity explicite sur un parking de la parcelle mais on prédit
    # à côté. L'OSM tagué est l'évidence la plus forte ; toute divergence > 50% est suspecte.
    if (
        capacity_osm_parcelle > 0
        and estimated > 0
        and abs(estimated - capacity_osm_parcelle) / max(capacity_osm_parcelle, 1) > 0.5
    ):
        flags.append(
            ConsistencyFlag(
                name="osm_tagged_divergence",
                severity="medium",
                reason=(
                    f"OSM tag capacity sur parcelle = {capacity_osm_parcelle}, "
                    f"prédiction = {estimated} (écart > 50%)"
                ),
            )
        )

    return flags


def summarize_flags(flags: List[ConsistencyFlag]) -> Dict[str, Any]:
    """Agrège la liste de flags en un dict prêt à exposer dans ``RowResult``."""
    by_severity = {"info": 0, "medium": 0, "high": 0}
    for f in flags:
        if f.severity in by_severity:
            by_severity[f.severity] += 1
    max_sev = "none"
    if by_severity["high"] > 0:
        max_sev = "high"
    elif by_severity["medium"] > 0:
        max_sev = "medium"
    elif by_severity["info"] > 0:
        max_sev = "info"
    return {
        "count": len(flags),
        "high_count": by_severity["high"],
        "medium_count": by_severity["medium"],
        "info_count": by_severity["info"],
        "max_severity": max_sev,
        "needs_review": by_severity["high"] > 0,
        "flags": [f.to_dict() for f in flags],
    }
