"""API data.gouv.fr — recherche de jeux de données (pagination)."""

from __future__ import annotations

from typing import Any, Iterator, List
from urllib.parse import quote_plus

import httpx

DATAGOUV_API = "https://www.data.gouv.fr/api/1/datasets/"


def iter_datasets_search(
    query: str,
    *,
    page_size: int = 100,
    max_pages: int = 50,
    client: httpx.Client | None = None,
) -> Iterator[dict[str, Any]]:
    """Itère sur les jeux data.gouv correspondant à `query` (paginé)."""
    own = client is None
    if own:
        client = httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        page = 1
        while page <= max_pages:
            url = f"{DATAGOUV_API}?q={quote_plus(query)}&page_size={page_size}&page={page}"
            r = client.get(url)
            r.raise_for_status()
            payload = r.json()
            batch = payload.get("data") or []
            if not batch:
                return
            for d in batch:
                yield d
            next_url = payload.get("next_page")
            if not next_url:
                return
            page += 1
    finally:
        if own:
            client.close()


def flatten_datagouv_resources(
    datasets: Iterable[dict[str, Any]],
    *,
    search_query: str,
    source_label: str = "data.gouv.fr",
) -> list[dict[str, Any]]:
    """Une ligne par ressource pour des jeux data.gouv."""
    rows: list[dict[str, Any]] = []
    for d in datasets:
        did = d.get("id")
        title = d.get("title")
        slug = d.get("slug")
        page_url = d.get("page")
        org = d.get("organization") or {}
        org_name = org.get("name") if isinstance(org, dict) else None
        licence = d.get("license")
        for res in d.get("resources") or []:
            if not isinstance(res, dict):
                continue
            rows.append(
                {
                    "source": source_label,
                    "search_query": search_query,
                    "dataset_id": did,
                    "dataset_title": title,
                    "dataset_slug": slug,
                    "dataset_page_url": page_url,
                    "dataset_licence": licence,
                    "dataset_organization": org_name,
                    "resource_id": res.get("id"),
                    "resource_title": res.get("title"),
                    "resource_format": (res.get("format") or "").upper() if res.get("format") else "",
                    "resource_url": res.get("url"),
                    "resource_updated": res.get("last_modified"),
                }
            )
    return rows


def collect_queries(
    queries: List[str],
    *,
    page_size: int = 100,
    max_pages_per_query: int = 30,
    client: httpx.Client | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Exécute plusieurs requêtes texte ; dédoublonne les jeux par id.
    Retourne (liste jeux uniques, lignes plates ressources).
    """
    seen: set[str] = set()
    unique_sets: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    own = client is None
    if own:
        client = httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            for d in iter_datasets_search(
                q,
                page_size=page_size,
                max_pages=max_pages_per_query,
                client=client,
            ):
                did = d.get("id")
                key = str(did) if did is not None else str(d.get("slug"))
                if key in seen:
                    continue
                seen.add(key)
                unique_sets.append(d)
                all_rows.extend(flatten_datagouv_resources([d], search_query=q))
    finally:
        if own:
            client.close()

    return unique_sets, all_rows
