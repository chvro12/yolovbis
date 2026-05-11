"""Classification du type de site pour calibrer les ratios m²/place (capacité théorique)."""

from __future__ import annotations

import re
from typing import Iterable, Optional, Set

SITE_TYPES = (
    "clinic",
    "supermarket",
    "residential",
    "industrial",
    "office",
    "public_parking",
    "unknown",
)


def collect_osm_amenity_tags(classified_rows: Optional[Iterable[object]]) -> Set[str]:
    """Extrait les tags ``amenity`` des éléments OSM classifiés (parkings, etc.)."""
    out: Set[str] = set()
    if not classified_rows:
        return out
    for row in classified_rows:
        el = getattr(row, "element", None)
        if el is None:
            continue
        tags = getattr(el, "tags", None) or {}
        a = tags.get("amenity") or tags.get("healthcare")
        if a:
            out.add(str(a).strip().lower())
    return out


def infer_site_type(
    ban_label: str = "",
    *,
    osm_amenity_tags: Optional[Set[str]] = None,
) -> str:
    """
    Heuristique légère : BAN + tags OSM. Ne pilote pas la capacité seule ; ajuste les ratios.
    """
    tags = osm_amenity_tags or set()
    t = (ban_label or "").lower()
    joined = t + " " + " ".join(tags)

    if any(
        x in joined
        for x in (
            "clinique",
            "clinic",
            "hôpital",
            "hopital",
            "hospital",
            "polyclinique",
            "polyclinic",
            "medical",
            "centre de santé",
        )
    ) or "clinic" in tags or "doctors" in tags or "hospital" in tags:
        return "clinic"

    if any(
        x in joined
        for x in (
            "supermarché",
            "supermarche",
            "hypermarché",
            "hypermarch",
            "carrefour",
            "leclerc",
            "auchan",
            "intermarché",
            "intermarche",
            "lidl",
            "aldi",
        )
    ) or "marketplace" in tags or tags.intersection({"supermarket", "convenience", "mall"}):
        return "supermarket"

    if "parking" in tags or "parking_entrance" in tags or re.search(
        r"\bparking\b", t, re.I
    ) or "parc de stationnement" in t:
        return "public_parking"

    if any(
        x in joined
        for x in (
            "zone industrielle",
            "industrial",
            "z.i.",
            "zi ",
            "parc d'activité",
            "parc d activité",
        )
    ) or "industrial" in tags:
        return "industrial"

    if any(x in joined for x in ("bureaux", "immeuble de bureaux", "office", " siège ", "siege ")):
        return "office"

    if any(
        x in joined
        for x in (
            "rue ",
            "avenue ",
            "boulevard ",
            "impasse ",
            "lotissement",
            "résidence",
            "residence",
            "appartement",
        )
    ) or tags.intersection({"residential", "apartments"}):
        return "residential"

    return "unknown"


def m2_per_space_unmarked_range(site_type: str) -> tuple[float, float]:
    """Fourchette m²/place pour surface non marquée (petit lot / cour / industriel)."""
    st = (site_type or "unknown").lower()
    if st == "industrial":
        return 35.0, 55.0
    if st in ("clinic", "office"):
        return 25.0, 32.0
    if st == "supermarket":
        return 24.0, 32.0
    if st == "public_parking":
        return 25.0, 34.0
    if st == "residential":
        return 28.0, 38.0
    return 26.0, 36.0


def m2_per_space_courtyard_range(site_type: str) -> tuple[float, float]:
    """Cour / arrière-bâtiment : circulation plus lâche."""
    st = (site_type or "unknown").lower()
    if st == "industrial":
        return 35.0, 55.0
    if st in ("clinic", "office", "supermarket"):
        return 30.0, 45.0
    if st == "public_parking":
        return 28.0, 42.0
    return 30.0, 45.0


def m2_per_space_marked_compact_range(site_type: str) -> tuple[float, float]:
    """Places marquées denses (sanity-check géométrie)."""
    st = (site_type or "unknown").lower()
    if st in ("supermarket", "public_parking"):
        return 22.0, 28.0
    if st in ("clinic", "office"):
        return 22.0, 30.0
    return 22.0, 28.0
