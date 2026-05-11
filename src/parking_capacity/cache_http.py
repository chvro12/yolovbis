"""Cache disque minimal pour requêtes HTTP (GET/POST) — WMS, Overpass."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

DEFAULT_TTL_S = 86_400


def _cache_key(method: str, url: str, body: Optional[bytes]) -> str:
    h = hashlib.sha256()
    h.update(method.upper().encode())
    h.update(b"\n")
    h.update(url.encode())
    if body:
        h.update(b"\n")
        h.update(body)
    return h.hexdigest()


def _entry_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def load_cached_response(
    cache_dir: Optional[Path],
    *,
    method: str,
    url: str,
    body: Optional[bytes],
    ttl_s: float = DEFAULT_TTL_S,
) -> Optional[bytes]:
    if cache_dir is None:
        return None
    path = _entry_path(cache_dir, _cache_key(method, url, body))
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(meta.get("ts", 0)) > ttl_s:
            return None
        b = meta.get("b64")
        if not b:
            return None
        import base64

        return base64.b64decode(str(b))
    except Exception:
        return None


def save_cached_response(
    cache_dir: Optional[Path],
    *,
    method: str,
    url: str,
    body: Optional[bytes],
    content: bytes,
) -> None:
    if cache_dir is None:
        return
    import base64

    path = _entry_path(cache_dir, _cache_key(method, url, body))
    payload = {"ts": time.time(), "b64": base64.b64encode(content).decode("ascii")}
    path.write_text(json.dumps(payload), encoding="utf-8")


def post_with_cache(
    client: httpx.Client,
    url: str,
    content: str,
    *,
    cache_dir: Optional[Path] = None,
    ttl_s: float = DEFAULT_TTL_S,
    max_retries: int = 3,
) -> httpx.Response:
    body = content.encode("utf-8")
    cached = load_cached_response(cache_dir, method="POST", url=url, body=body, ttl_s=ttl_s)
    if cached is not None:
        return httpx.Response(200, content=cached)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = client.post(
                url,
                content=content,
                headers={
                    "Content-Type": "text/plain;charset=UTF-8",
                    # Overpass exige un User-Agent identifiable (sinon 406).
                    "User-Agent": "parking-capacity/0.1 (+https://github.com/)",
                },
            )
            r.raise_for_status()
            save_cached_response(cache_dir, method="POST", url=url, body=body, content=r.content)
            return r
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(0.8 * (attempt + 1))
    raise last_exc  # type: ignore[misc]
