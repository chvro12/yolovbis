"""Résolution de la capacité théorique de stationnement (``parking_capacity_estimation``).

Hiérarchie produit (aucune étape ne dérive la capacité du seul comptage de véhicules) :

A. OSM ``capacity`` taggé fiable
B. Places marquées visibles (vision spécialisée parking, preuve suffisante)
C. Géométrie de places marquées (rangées / marquages, confiance medium+)
D. Surface bitumée garable non marquée (scénario ``unmarked_surface`` medium+)
E. Stationnement linéaire bord de chaussée (scénario ``roadside_parking`` medium+)
F. Cour / garage / zone privée (``courtyard_parking``)
G. Comptage OSM ``parking_space``
H. ML régression validé
I. Indices faibles (géométrie weak, scénarios weak) puis refus / secours OSM

Les véhicules détectés n’entrent pas dans cette chaîne : ils servent uniquement de
``supporting_evidence`` (confiance, usage) côté pipeline / ``result.json``.
"""

from __future__ import annotations

from typing import Optional, Tuple

from parking_capacity.vision_estimate import VisionEstimate


_STRONG_CONFS = ("medium", "strong", "high")


def pick_primary_capacity(
    *,
    source_priority: str,
    has_osm_capacity: bool,
    cap_p: int,
    cap_b: int,
    tagged_p: int,
    tagged_b: int,
    ban_score: float,
    vision_est: Optional[VisionEstimate],
    vision_primary_places: Optional[int],
    geometry_places: Optional[int],
    geometry_confidence: str,
    ml_int: Optional[int],
    ml_mode_l: str,
    mid_a: int,
    mn_a: int,
    mx_a: int,
    ta: float,
    osm_parking_space_count: int,
    visual_evidence_level: str,
    visual_specialized_effective: bool = False,
    scenario_primary_capacity: Optional[int] = None,
    scenario_primary_source: Optional[str] = None,
    scenario_primary_confidence: str = "none",
) -> Tuple[Optional[int], Optional[str], Optional[str], str]:
    """Retourne ``(primary, source, confidence, provenance)`` pour l’estimation théorique."""
    sp = (source_priority or "hybrid").strip().lower()
    if sp not in ("aerial", "osm", "hybrid"):
        sp = "hybrid"

    v_pl = vision_primary_places if vision_primary_places and vision_primary_places > 0 else None
    m_pl = ml_int if ml_int and ml_int > 0 else None

    g_pl = geometry_places if geometry_places and geometry_places > 0 else None
    g_ok = geometry_confidence in _STRONG_CONFS

    def osm_primary() -> Tuple[Optional[int], Optional[str], Optional[str], str]:
        if cap_p > 0:
            conf = "high" if ban_score >= 0.7 else "medium"
            return cap_p, "osm_parcelle", conf, "priorité_osm:tag_capacity_parcelle"
        if cap_b > 0:
            return cap_b, "osm_buffer", "low", "priorité_osm:tag_capacity_buffer"
        return None, None, None, "osm:pas_de_capacity_taguée"

    def aerial_chain() -> Tuple[Optional[int], Optional[str], Optional[str], str]:
        """Chaîne aérienne : B → C → D → E → F → G (OSM places) → H (ML) → indices faibles."""
        # B — places marquées visibles (vision spécialisée, pas SegFormer générique seul)
        if v_pl and visual_specialized_effective and visual_evidence_level in ("medium", "strong"):
            return (
                int(v_pl),
                "vision_marked_visible",
                "medium",
                "priorité_aérienne:places_marquées_visibles",
            )

        # C — géométrie marquages fiable
        if g_pl and g_ok:
            conf = "high" if geometry_confidence == "strong" else "medium"
            return g_pl, "parking_geometry", conf, "priorité_aérienne:géométrie_places_marquées"

        # D — surface garable non marquée
        if (
            scenario_primary_capacity
            and scenario_primary_source == "scenario_unmarked_surface"
            and scenario_primary_confidence == "medium"
        ):
            return (
                int(scenario_primary_capacity),
                scenario_primary_source,
                "medium",
                "priorité_aérienne:surface_non_marquée",
            )

        # E — linéaire chaussée / bas-côté
        if (
            scenario_primary_capacity
            and scenario_primary_source == "scenario_roadside_parking"
            and scenario_primary_confidence == "medium"
        ):
            return (
                int(scenario_primary_capacity),
                scenario_primary_source,
                "medium",
                "priorité_aérienne:stationnement_linéaire",
            )

        # F — cour / garage / zone privée
        if (
            scenario_primary_capacity
            and scenario_primary_source == "scenario_courtyard_parking"
            and scenario_primary_confidence in ("medium", "weak")
        ):
            return (
                int(scenario_primary_capacity),
                scenario_primary_source,
                "low" if scenario_primary_confidence == "weak" else "medium",
                "priorité_aérienne:cour_ou_zone_privée",
            )

        # G — comptage OSM parking_space (après scénarios surfaciques)
        if osm_parking_space_count > 0:
            return (
                osm_parking_space_count,
                "osm_parking_space_count",
                "medium",
                "priorité_aérienne:comptage_parking_space_osm",
            )

        # H — ML
        if ml_mode_l in ("before_vision", "fallback", "aux") and m_pl:
            return m_pl, "ml_regressor", "low", "priorité_aérienne:ml_régression"

        # Indices faibles : géométrie marquée basse confiance
        if g_pl and geometry_confidence == "weak":
            return (
                g_pl,
                "parking_geometry",
                "low",
                "priorité_aérienne:géométrie_basse_confiance_indice",
            )

        # Surface / linéaire en weak (indice seulement)
        if (
            scenario_primary_capacity
            and scenario_primary_source
            and scenario_primary_confidence == "weak"
            and scenario_primary_source
            in ("scenario_unmarked_surface", "scenario_roadside_parking", "scenario_courtyard_parking")
        ):
            return (
                int(scenario_primary_capacity),
                scenario_primary_source,
                "low",
                f"priorité_aérienne:{scenario_primary_source}_indice_faible",
            )

        # Secours OSM capacity si présent (hors has_osm_capacity déjà traité en hybrid)
        if cap_p > 0:
            return cap_p, "osm_parcelle", "medium", "secours_osm_capacity_après_image"
        if cap_b > 0:
            return cap_b, "osm_buffer", "low", "secours_osm_buffer_après_image"

        return None, None, None, "aucune_source"

    def hybrid_chain() -> Tuple[Optional[int], Optional[str], Optional[str], str]:
        if has_osm_capacity:
            p, s, c, note = osm_primary()
            if p is not None:
                return p, s, c, f"hybrid:osm_fiable→{note}"
        p2, s2, c2, n2 = aerial_chain()
        if p2 is not None:
            return p2, s2, c2, f"hybrid:sans_osm_capacity→{n2}"
        return None, None, None, "hybrid:aucune_capacité_déduite"

    def osm_priority_chain() -> Tuple[Optional[int], Optional[str], Optional[str], str]:
        p, s, c, n = osm_primary()
        if p is not None:
            return p, s, c, f"osm_first:{n}"
        return aerial_chain()

    if sp == "aerial":
        return aerial_chain()
    if sp == "osm":
        return osm_priority_chain()
    return hybrid_chain()
