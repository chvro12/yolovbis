"""API transport.data.gouv.fr — liste des jeux de données."""

from __future__ import annotations

from typing import Any, Iterable, List

import httpx

PAN_DATASETS_URL = "https://transport.data.gouv.fr/api/datasets"


def fetch_all_datasets(*, client: httpx.Client | None = None) -> List[dict[str, Any]]:
    """Télécharge la liste complète des jeux publiés sur le PAN (JSON)."""
    own = client is None
    if own:
        client = httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        r = client.get(PAN_DATASETS_URL)
        r.raise_for_status()
        data = r.json()
    finally:
        if own:
            client.close()
    if not isinstance(data, list):
        raise RuntimeError("Réponse PAN inattendue (liste attendue)")
    return data


def _text_blob(d: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in ("title", "slug", "type"):
        v = d.get(k)
        if v:
            parts.append(str(v).lower())
    tags = d.get("tags") or []
    if isinstance(tags, list):
        parts.extend(str(t).lower() for t in tags)
    st = d.get("sub_types") or []
    if isinstance(st, list):
        parts.extend(str(t).lower() for t in st)
    return " ".join(parts)


def filter_datasets_by_keywords(
    datasets: Iterable[dict[str, Any]],
    keywords: Iterable[str],
) -> list[dict[str, Any]]:
    """Filtre les jeux dont titre / slug / tags contiennent l’un des mots-clés."""
    kws = [k.strip().lower() for k in keywords if k and str(k).strip()]
    if not kws:
        return list(datasets)
    out: list[dict[str, Any]] = []
    for d in datasets:
        blob = _text_blob(d)
        if any(k in blob for k in kws):
            out.append(d)
    return out


def flatten_pan_resources(
    datasets: Iterable[dict[str, Any]],
    *,
    source_label: str = "transport.data.gouv.fr",
) -> list[dict[str, Any]]:
    """Une ligne par ressource (fichier) liée à un jeu PAN."""
    rows: list[dict[str, Any]] = []
    for d in datasets:
        did = d.get("id")
        title = d.get("title")
        slug = d.get("slug")
        page_url = d.get("page_url")
        licence = d.get("licence")
        publisher = (d.get("publisher") or {}).get("name") if isinstance(d.get("publisher"), dict) else d.get("publisher")
        for res in d.get("resources") or []:
            if not isinstance(res, dict):
                continue
            rows.append(
                {
                    "source": source_label,
                    "dataset_id": did,
                    "dataset_title": title,
                    "dataset_slug": slug,
                    "dataset_page_url": page_url,
                    "dataset_licence": licence,
                    "dataset_publisher": publisher,
                    "resource_id": res.get("id"),
                    "resource_title": res.get("title"),
                    "resource_format": (res.get("format") or "").upper() if res.get("format") else "",
                    "resource_url": res.get("url"),
                    "resource_updated": res.get("updated"),
                }
            )
    return rows
