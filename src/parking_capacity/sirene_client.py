"""Client API recherche-entreprises (data.gouv.fr) pour récupérer le code APE d'une adresse.

API publique gratuite, sans clé : https://recherche-entreprises.api.gouv.fr/

Retourne l'établissement le mieux matché à l'adresse fournie, avec code APE (NAF) qu'on
mappe ensuite en typologie de site (commerce / clinique / hôpital / résidentiel / …) via
``site_typology.py``.

Cache disque optionnel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://recherche-entreprises.api.gouv.fr/search"
TIMEOUT_S = 10.0


@dataclass
class SireneEstablishment:
    """Un établissement SIRENE matché à l'adresse."""

    name: str
    siret: Optional[str]
    ape_code: Optional[str]          # ex. "75.00Z" pour vétérinaire
    address: Optional[str]
    address_match_score: float = 0.0  # 0-1, proxy de qualité du matching


@dataclass
class SireneResult:
    """Résultat de la recherche : 0+ établissements + métadonnées."""

    establishments: List[SireneEstablishment]
    primary: Optional[SireneEstablishment]  # le meilleur match
    raw_count: int = 0
    cached: bool = False
    query: str = ""


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]


def _normalize_address(addr: str) -> str:
    """Normalise pour matching SIRENE : sans accents/A/B/bis/ter, minuscules."""
    s = addr.upper()
    # supprime suffixes courants
    s = re.sub(r"\b(BIS|TER|QUATER|A|B|C)\b", " ", s)
    # remplace caractères spéciaux par espaces
    s = re.sub(r"[^\w]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _address_match_score(query: str, candidate: str) -> float:
    """Score Jaccard simple sur les tokens normalisés."""
    if not query or not candidate:
        return 0.0
    qt = set(_normalize_address(query).split())
    ct = set(_normalize_address(candidate).split())
    if not qt or not ct:
        return 0.0
    inter = qt & ct
    union = qt | ct
    return len(inter) / max(len(union), 1)


def search_sirene(
    address: str,
    *,
    client: Optional[httpx.Client] = None,
    cache_dir: Optional[Path] = None,
    per_page: int = 5,
) -> SireneResult:
    """Recherche les établissements actifs à cette adresse via recherche-entreprises.

    Stratégie de fallback :
    1. Essai direct avec l'adresse complète.
    2. Si 0 résultat, essai sans le ``2A``/``2B`` suffix du numéro (cas Bouzonville).
    """
    # Cache
    if cache_dir is not None:
        cdir = Path(cache_dir) / "sirene"
        cdir.mkdir(parents=True, exist_ok=True)
        cpath = cdir / f"{_cache_key(address)}.json"
        if cpath.is_file():
            try:
                data = json.loads(cpath.read_text())
                return _parse_response(data, address, cached=True)
            except Exception:  # noqa: BLE001
                pass

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=TIMEOUT_S)

    def _do_search(q: str) -> dict:
        try:
            r = client.get(API_URL, params={"q": q, "per_page": per_page})
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            logger.info("sirene_client: %s", e)
            return {"total_results": 0, "results": []}

    try:
        data = _do_search(address)
        if data.get("total_results", 0) == 0:
            # Retry : supprime suffixe lettre du numéro (2A → 2)
            simplified = re.sub(r"^(\d+)[A-Za-z]\b", r"\1", address.strip())
            if simplified != address:
                data = _do_search(simplified)
    finally:
        if own_client:
            client.close()

    # Cache (même en cas de 0 résultats : évite de re-spammer l'API)
    if cache_dir is not None:
        try:
            cpath.write_text(json.dumps(data, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass

    return _parse_response(data, address, cached=False)


# Pertinence parking : un APE "site physique avec parking" est plus pertinent qu'un APE
# "holding immobilier / activité dématérialisée". On utilise ce score pour départager les
# établissements à la même adresse.
_APE_PARKING_RELEVANCE: list[tuple[str, float]] = [
    # Très pertinent : sites physiques avec parking attaché
    ("47.11", 1.00),  # commerce alimentaire (super/hyper)
    ("47.19", 0.95),  # grand magasin
    ("75.00", 0.95),  # vétérinaire
    ("86.10", 0.95),  # hôpital
    ("86.21", 0.85),  # cabinet médical
    ("86.22", 0.85),
    ("86.23", 0.85),
    ("86.90", 0.85),
    ("85.10", 0.80),  # école
    ("85.20", 0.80),
    ("85.31", 0.80),
    ("85.32", 0.80),
    ("85.4",  0.80),
    ("87",    0.90),  # EHPAD
    ("55.10", 0.85),  # hôtel
    ("56.10", 0.80),  # restaurant
    ("84.11", 0.75),  # admin publique
    ("84.12", 0.75),
    ("47.30", 0.85),  # station-service
    ("47.7",  0.55),  # petit commerce
    ("93.",   0.75),  # sport/loisirs
    # Moyennement pertinent
    ("64.",   0.40),  # banque
    ("65.",   0.40),  # assurance
    ("66.",   0.40),
    ("70.",   0.50),  # bureau
    # Peu pertinent : pas de "site" attaché ou holding pur
    ("68.20", 0.20),  # location logement (rarement le bon "site")
    ("68.31", 0.30),  # agence immobilière
    ("68.32", 0.20),  # gestion immobilière
    ("82.",   0.30),  # services administratifs
    ("63.",   0.20),  # services info dématérialisés
]


def _parking_relevance(ape: Optional[str]) -> float:
    """Score 0-1 de pertinence parking pour un code APE."""
    if not ape:
        return 0.10
    ape_u = ape.upper()
    # Match préfixe le plus long
    candidates = sorted(_APE_PARKING_RELEVANCE, key=lambda r: -len(r[0]))
    for prefix, score in candidates:
        if ape_u.startswith(prefix):
            return score
    return 0.50  # default neutre


def _parse_response(data: dict, query: str, *, cached: bool) -> SireneResult:
    """Parse la réponse JSON et choisit le meilleur match.

    Score combiné = address_match × 0.6 + parking_relevance × 0.4.
    Ça privilégie un vétérinaire à la bonne adresse (75.00Z, relevance 0.95) sur un holding
    immobilier à la même adresse (68.20B, relevance 0.20).
    """
    raw = data.get("results", [])
    establishments: List[SireneEstablishment] = []
    for r in raw:
        ape = r.get("activite_principale") or r.get("siege", {}).get("activite_principale")
        siret = r.get("siret") or r.get("siege", {}).get("siret")
        addr = r.get("siege", {}).get("adresse")
        name = r.get("nom_complet") or r.get("nom_raison_sociale") or ""
        addr_score = _address_match_score(query, addr or "")
        relevance = _parking_relevance(ape)
        combined = 0.6 * addr_score + 0.4 * relevance
        establishments.append(SireneEstablishment(
            name=str(name),
            siret=str(siret) if siret else None,
            ape_code=str(ape) if ape else None,
            address=str(addr) if addr else None,
            address_match_score=round(combined, 3),
        ))
    establishments.sort(key=lambda e: -e.address_match_score)
    primary = establishments[0] if establishments else None
    return SireneResult(
        establishments=establishments,
        primary=primary,
        raw_count=int(data.get("total_results", 0)),
        cached=cached,
        query=query,
    )
