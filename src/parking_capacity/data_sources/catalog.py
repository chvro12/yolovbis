"""Fusion, dédoublonnage et export du catalogue de ressources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import httpx
import pandas as pd

from parking_capacity.data_sources.datagouv import collect_queries
from parking_capacity.data_sources.pan import (
    fetch_all_datasets,
    filter_datasets_by_keywords,
    flatten_pan_resources,
)


def build_merged_catalog(
    *,
    pan_keywords: Sequence[str],
    datagouv_queries: Sequence[str],
    include_pan: bool = True,
    include_datagouv: bool = True,
    datagouv_max_pages: int = 30,
    client: Optional[httpx.Client] = None,
) -> list[dict[str, Any]]:
    """Construit la liste plate des ressources (PAN filtré + data.gouv)."""
    rows: list[dict[str, Any]] = []

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=180.0, follow_redirects=True)
    try:
        if include_pan:
            datasets = fetch_all_datasets(client=client)
            filtered = filter_datasets_by_keywords(datasets, pan_keywords)
            rows.extend(flatten_pan_resources(filtered))

        if include_datagouv:
            qlist = [q for q in datagouv_queries if str(q).strip()]
            if qlist:
                _, dg_rows = collect_queries(
                    list(qlist),
                    max_pages_per_query=datagouv_max_pages,
                    client=client,
                )
                rows.extend(dg_rows)
    finally:
        if own_client and client is not None:
            client.close()

    # Déduplication par URL de ressource (quand présente)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in rows:
        url = r.get("resource_url")
        key = str(url) if url else f"{r.get('source')}:{r.get('resource_id')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def write_catalog(rows: Iterable[dict[str, Any]], path: Path, *, fmt: str = "csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    fmt_l = fmt.lower().strip()
    if fmt_l == "csv":
        df.to_csv(path, index=False)
    elif fmt_l == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    else:
        raise ValueError("fmt doit être csv ou jsonl")


def default_pan_keywords() -> List[str]:
    return [
        "stationnement",
        "parking",
        "parc relais",
        "saemes",
        "indigo",
        "netex",
        "silo",
        "parkings",
    ]


def default_datagouv_queries() -> List[str]:
    return [
        "stationnement hors voirie",
        "lieux de stationnement",
        "parc de stationnement",
        "parking capacité",
        "parkings métropole",
    ]
