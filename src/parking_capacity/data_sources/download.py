"""Téléchargement HTTP d'une ressource (fichier) avec limite de taille."""

from __future__ import annotations

from pathlib import Path

import httpx


def download_url_to_file(
    url: str,
    dest: Path,
    *,
    max_bytes: int = 500 * 1024 * 1024,
    timeout_s: float = 300.0,
    client: httpx.Client | None = None,
) -> int:
    """
    Télécharge `url` vers `dest` en flux. Retourne la taille en octets.
    Lève ValueError si la taille dépasse max_bytes (en-tête Content-Length ou compteur).
    """
    own = client is None
    if own:
        client = httpx.Client(timeout=timeout_s, follow_redirects=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            cl = r.headers.get("content-length")
            if cl is not None:
                cl_int = int(cl)
                if cl_int > max_bytes:
                    raise ValueError(
                        f"Fichier trop volumineux ({cl_int} octets > {max_bytes})"
                    )
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Téléchargement interrompu : dépassement de {max_bytes} octets")
                    f.write(chunk)
    finally:
        if own:
            client.close()
    return total
