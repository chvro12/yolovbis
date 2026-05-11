"""Petites bbox (WGS84) pour tests rapides en France : min_lon,min_lat,max_lon,max_lat."""

from __future__ import annotations

FRANCE_BBOX_PRESETS: dict[str, str] = {
    "paris_small": "2.325,48.842,2.365,48.868",
    "lyon_small": "4.820,45.735,4.865,45.775",
    "nantes_small": "-1.585,47.195,-1.535,47.225",
    "rennes_small": "-1.715,48.095,-1.655,48.125",
    "bordeaux_small": "-0.605,44.825,-0.545,44.865",
}


def resolve_bbox_arg(*, bbox: str | None, preset: str | None) -> str:
    if bbox and bbox.strip():
        return bbox.strip()
    if preset:
        key = preset.strip().lower()
        if key not in FRANCE_BBOX_PRESETS:
            raise ValueError(
                f"preset inconnu {preset!r}. Connus : {', '.join(sorted(FRANCE_BBOX_PRESETS))}"
            )
        return FRANCE_BBOX_PRESETS[key]
    raise ValueError("Indiquez --bbox ou --preset (ex. lyon_small).")
