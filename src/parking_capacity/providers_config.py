"""Configuration centralisée des fournisseurs GIS (YAML + variables d'environnement)."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_PROVIDERS_REL = Path("providers.yaml")
ENV_PROVIDERS_PATH = "PARKING_PROVIDERS_YAML"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def default_providers_dict() -> Dict[str, Any]:
    """Valeurs par défaut alignées sur ``providers.yaml.example``."""
    return {
        "ign": {
            "enabled": True,
            "geoplateforme_wfs_url": "https://data.geopf.fr/wfs/ows",
            "geoplateforme_wms_raster_url": "https://data.geopf.fr/wms-r",
            "use_bdtopo": True,
            "bdtopo_buildings_typename": "BDTOPO_V3:batiment",
            "bdtopo_roads_typename": "BDTOPO_V3:troncon_de_route",
            "bdtopo_zones_typename": "BDTOPO_V3:zone_d_activite_ou_d_interet",
            "wfs_max_features": 500,
        },
        "osm": {
            "enabled": True,
            "overpass_url": "https://overpass-api.de/api/interpreter",
        },
        "microsoft_buildings": {
            "enabled": False,
            "local_path_env": "MICROSOFT_BUILDINGS_PATH",
        },
        "mapillary": {
            "enabled": False,
            "token_env": "MAPILLARY_ACCESS_TOKEN",
            "graph_api_url": "https://graph.mapillary.com",
        },
        "fusion": {
            "access_distance_threshold_m": 40.0,
            "max_plausible_capacity_slots": 39,
        },
    }


@dataclass
class GisProvidersConfig:
    """Configuration résolue (YAML + env)."""

    raw: Dict[str, Any] = field(default_factory=dict)
    ign_enabled: bool = True
    ign_wfs_url: str = "https://data.geopf.fr/wfs/ows"
    ign_use_bdtopo: bool = True
    ign_bdtopo_buildings_typename: str = "BDTOPO_V3:batiment"
    ign_bdtopo_roads_typename: str = "BDTOPO_V3:troncon_de_route"
    ign_wfs_max_features: int = 500
    osm_enabled: bool = True
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    microsoft_enabled: bool = False
    microsoft_buildings_path: Optional[Path] = None
    mapillary_enabled: bool = False
    mapillary_token: Optional[str] = None
    mapillary_graph_url: str = "https://graph.mapillary.com"
    access_distance_threshold_m: float = 40.0
    max_plausible_capacity_slots: int = 39

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GisProvidersConfig:
        ign = d.get("ign") or {}
        osm = d.get("osm") or {}
        ms = d.get("microsoft_buildings") or {}
        mp = d.get("mapillary") or {}
        fusion = d.get("fusion") or {}

        ms_env = str(ms.get("local_path_env") or "MICROSOFT_BUILDINGS_PATH")
        mp_env = str(mp.get("token_env") or "MAPILLARY_ACCESS_TOKEN")

        ms_path = os.environ.get(ms_env, "").strip()
        mp_tok = os.environ.get(mp_env, "").strip()

        return cls(
            raw=d,
            ign_enabled=bool(ign.get("enabled", True)),
            ign_wfs_url=str(ign.get("geoplateforme_wfs_url") or ign.get("wfs_url") or "https://data.geopf.fr/wfs/ows"),
            ign_use_bdtopo=bool(ign.get("use_bdtopo", True)),
            ign_bdtopo_buildings_typename=str(ign.get("bdtopo_buildings_typename") or "BDTOPO_V3:batiment"),
            ign_bdtopo_roads_typename=str(ign.get("bdtopo_roads_typename") or "BDTOPO_V3:troncon_de_route"),
            ign_wfs_max_features=int(ign.get("wfs_max_features") or 500),
            osm_enabled=bool(osm.get("enabled", True)),
            overpass_url=str(osm.get("overpass_url") or "https://overpass-api.de/api/interpreter"),
            microsoft_enabled=bool(ms.get("enabled", False)) and bool(ms_path),
            microsoft_buildings_path=Path(ms_path).expanduser() if ms_path else None,
            mapillary_enabled=bool(mp.get("enabled", False)) and bool(mp_tok),
            mapillary_token=mp_tok or None,
            mapillary_graph_url=str(mp.get("graph_api_url") or "https://graph.mapillary.com"),
            access_distance_threshold_m=float(fusion.get("access_distance_threshold_m", 40.0)),
            max_plausible_capacity_slots=int(fusion.get("max_plausible_capacity_slots", 39)),
        )


def load_providers_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML requis : pip install pyyaml")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return {}
    return data


def load_gis_providers_config(
    *,
    yaml_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> GisProvidersConfig:
    """
    Charge ``providers.yaml`` si présent, sinon valeurs par défaut.

    Ordre : ``yaml_path`` explicite → env ``PARKING_PROVIDERS_YAML`` →
    ``<cwd>/providers.yaml`` → ``<project_root>/providers.yaml``.
    """
    base = default_providers_dict()
    candidates: list[Path] = []
    if yaml_path is not None:
        candidates.append(Path(yaml_path))
    env_p = os.environ.get(ENV_PROVIDERS_PATH, "").strip()
    if env_p:
        candidates.append(Path(env_p).expanduser())
    candidates.append(Path.cwd() / DEFAULT_PROVIDERS_REL)
    if project_root is not None:
        candidates.append(Path(project_root) / DEFAULT_PROVIDERS_REL)

    merged = base
    for p in candidates:
        if p.is_file():
            if yaml is None:
                raise RuntimeError("PyYAML requis pour lire providers.yaml")
            merged = _deep_merge(merged, load_providers_yaml(p))
            break

    return GisProvidersConfig.from_dict(merged)
