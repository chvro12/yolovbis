"""Téléchargement, préparation et registre des jeux satellite (APKLOT, DOTA, xView, SpaceNet)."""

from parking_capacity.datasets_satellite.registry import (
    default_registry,
    load_registry,
    save_registry,
    update_dataset_status,
)

__all__ = [
    "default_registry",
    "load_registry",
    "save_registry",
    "update_dataset_status",
]
