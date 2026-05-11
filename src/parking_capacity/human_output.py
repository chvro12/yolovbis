"""Sortie lisible humain pour CLI : distingue capacité retenue, indices non retenus, raisons."""

from __future__ import annotations

from parking_capacity.pipeline import RowResult


def _method_fr(source: str | None) -> str:
    if not source:
        return "inconnue"
    m = {
        "osm_parcelle": "OSM (tag capacity sur parcelle)",
        "osm_buffer": "OSM (tag capacity hors parcelle, buffer)",
        "osm_parking_space_count": "OSM (comptage parking:individual / places)",
        "vision_specialized": "orthophoto + modèle vision spécialisé parking",
        "vision_marked_visible": "orthophoto + vision spécialisée (places marquées visibles)",
        "ml_regressor": "orthophoto + modèle ML (régression)",
        "parking_geometry": "orthophoto + géométrie (places marquées)",
        "scenario_unmarked_surface": "orthophoto + zone bitumée non marquée",
        "scenario_roadside_parking": "orthophoto + bord de chaussée",
        "scenario_courtyard_parking": "orthophoto + cour / arrière de bâtiment",
    }
    return m.get(source, source)


def _sources_fr(sources_used: str | None) -> str:
    if not sources_used:
        return "—"
    parts = []
    for p in sources_used.split("|"):
        q = {
            "ban": "géocodage BAN",
            "apicarto": "parcelle cadastrale APICarto",
            "osm_overpass": "parkings OSM (Overpass)",
            "ign_wms_orthophoto": "orthophoto IGN (WMS)",
            "segformer_parking": "SegFormer (zones parking, indice surface)",
            "parking_geometry": "géométrie rangées (lignes / espacements)",
            "ml_regression": "régression ML sur puce",
        }.get(p, p)
        parts.append(q)
    return ", ".join(parts)


def _format_unretained(row: RowResult) -> list[str]:
    """Liste explicite des indices visuels NON retenus + éléments rejetés (toit, etc.)."""
    out: list[str] = []
    if row.surface_only_capacity_hint is not None and row.method_used != "vision_specialized":
        area = f"~{row.surface_only_area_m2:.0f} m²" if row.surface_only_area_m2 else "?"
        out.append(
            f"Surface SegFormer seule : ~{row.surface_only_capacity_hint} places ({area}) — "
            "indice, non retenu car SegFormer générique ne compte pas les places."
        )
    if (
        row.geometric_capacity_estimate is not None
        and row.geometry_confidence == "weak"
        and row.method_used != "parking_geometry"
    ):
        out.append(
            f"Géométrie faible (places marquées) : ~{row.geometric_capacity_estimate} places — "
            "non retenue, marquages ou répétition insuffisants."
        )
    if row.roof_likelihood > 0.10:
        out.append(
            f"Toits détectés couvrant ~{row.roof_likelihood:.0%} de la puce — "
            "rangées dont le centre tombe dans un toit ont été rejetées."
        )
    if row.geometry_debug:
        cf = row.geometry_debug.get("chain_failure")
        if cf:
            out.append(f"Chaîne géométrique stoppée à : {cf}")
        rrs = row.geometry_debug.get("rejection_reasons") or []
        for r in rrs:
            out.append(f"Rangées rejetées pour cause : {r}")
    return out


def _format_components(row: RowResult) -> list[str]:
    """Affiche chaque scénario calculé (places marquées / non marquées / bord chaussée / cour)."""
    out: list[str] = []
    primary_mode = row.parking_visual_mode
    comps = row.visual_capacity_components if isinstance(row.visual_capacity_components, dict) else {}

    def _line(mode: str, label: str) -> str | None:
        v = comps.get(mode) if isinstance(comps, dict) else None
        if not v:
            return None
        retenu = " (retenu)" if mode == primary_mode else ""
        cap = v.get("capacity_estimate")
        mn = v.get("capacity_min")
        mx = v.get("capacity_max")
        conf = v.get("confidence", "?")
        return f"- {label}{retenu} : ~{cap} ({mn}-{mx}) confiance={conf}"

    for mode, label in (
        ("marked_slots", "Places marquées"),
        ("unmarked_surface", "Zone bitumée non marquée"),
        ("roadside_parking", "Stationnement bord chaussée"),
        ("courtyard_parking", "Cour / arrière bâtiment"),
    ):
        line = _line(mode, label)
        if line:
            out.append(line)
    return out


def _manual_check_list(row: RowResult) -> list[str]:
    out: list[str] = []
    if row.method_used == "parking_geometry" and row.geometry_confidence in ("weak", "medium"):
        out.append("Comparer visuellement les rangées tracées (debug_parking_rows.png) à l'orthophoto.")
    if row.method_used in ("osm_parcelle", "osm_buffer") and row.geometric_capacity_estimate:
        out.append("OSM divergent vs géométrie : confirmer le polygone parking sur OSM.")
    if row.surface_only_capacity_hint and row.method_used not in ("vision_specialized", "parking_geometry"):
        out.append("Hint surface seule : si la zone n'est pas un parking, ignorer.")
    if row.refuse_prediction:
        out.append("Prédiction refusée : vérifier l'adresse / image, ajouter un poids spécialisé ou un tag OSM.")
    return out


def format_run_address_pretty(row: RowResult) -> str:
    """Texte multi-lignes pour affichage console : retenu vs non retenus + vérifications."""
    lo = row.min_capacity
    hi = row.max_capacity
    if lo is not None and hi is not None and lo != hi:
        fork = f"{lo}–{hi}"
    elif lo is not None:
        fork = str(lo)
    else:
        fork = "—"
    est = row.estimated_capacity
    est_s = str(est) if est is not None else "— (voir indices ci-dessous)"
    lines = [
        f"Adresse : {row.input_address}",
        f"BAN : {row.ban_label or '—'} (score {row.ban_score})",
        f"Rayon : {row.radius_m_used or '—'} m",
        "",
        "=== Capacité théorique (parking_capacity_estimation) ===",
        f"Capacité publiée (retenue) : {est_s} places",
        f"Fourchette : {fork} places",
        f"Méthode : {_method_fr(row.method_used)}",
        f"Capacité théorique brute (hors refus) : {row.primary_capacity if row.primary_capacity is not None else '—'} places",
        f"Site (calage ratios) : {getattr(row, 'site_type', 'unknown')}",
        f"Mode visuel : {row.parking_visual_mode}",
        f"Confiance : {row.primary_confidence or '—'}",
        f"Provenance : {row.capacity_provenance or '—'}",
        "",
        "=== Preuves secondaires (véhicules, usage) ===",
        f"Présence véhicules : n={row.vehicle_count} ({row.vehicle_detection_method}) — "
        f"indice de confiance / usage, pas la capacité théorique.",
        f"Alignement (indice) : {row.vehicle_alignment_score:.2f} | clusters : {row.parked_vehicle_clusters}",
        f"Comptage observé (métadonnée) : {row.observed_vehicle_floor} | Plafond physique : {row.plausible_capacity_ceiling}",
        f"Parking usability score : {row.parking_usability_score or 0:.0f}/100 "
        f"({row.semantic_confidence})",
        "",
        "=== Surface classifiée ===",
        f"Asphalte : {row.asphalt_likelihood:.0%} | Toit : {row.roof_likelihood:.0%} | "
        f"Chaussée : {row.road_likelihood:.0%} | Végétation : {row.vegetation_likelihood:.0%}",
        f"Bâtiments exclus ~{row.building_area_m2 or 0:.0f} m² | ratio hors bâti : {(row.parking_outside_buildings_ratio or 0):.0%}",
        f"Accès route : score={row.vehicle_access_score or 0:.2f}, connexion={row.road_connection_detected}",
        "",
        "=== Preuve visuelle / géométrie ===",
        f"Niveau : {row.visual_evidence_level or '—'} (image : {row.image_used})",
        f"Rangées détectées : {row.parking_rows_detected} (orientation ~{row.estimated_row_orientation_deg:.0f}°)",
        f"Géométrie marquages : conf={row.geometry_confidence}, "
        f"cap={row.geometric_capacity_estimate} ({row.geometric_capacity_min}-{row.geometric_capacity_max})",
        f"Sources : {_sources_fr(row.sources_used)}",
    ]
    comp_lines = _format_components(row)
    if comp_lines:
        lines.append("")
        lines.append("=== Composants par scénario ===")
        lines.extend(comp_lines)
    unretained = _format_unretained(row)
    if unretained:
        lines.append("")
        lines.append("=== Indices non retenus ===")
        for u in unretained:
            lines.append(f"- {u}")
    manual = _manual_check_list(row)
    if manual:
        lines.append("")
        lines.append("=== À vérifier manuellement ===")
        for m in manual:
            lines.append(f"- {m}")
    w = (row.warnings or "").strip()
    if w:
        lines.append("")
        lines.append("=== Avertissements ===")
        for part in w.split(";"):
            p = part.strip()
            if p:
                lines.append(f"- {p}")
    if row.error:
        lines.append("")
        lines.append(f"Erreur : {row.error}")
    return "\n".join(lines)
