"""Requête Overpass parking_space : tags requis pour le comptage."""

import httpx

from parking_capacity.overpass import build_overpass_parking_space_count, query_parking_space_count


def test_parking_space_query_uses_out_tags_not_skel():
    q = build_overpass_parking_space_count(45.730513, 4.8587, 50)
    assert "out tags" in q
    assert "out skel" not in q
    assert 'relation["amenity"="parking_space"]' in q


def test_parking_space_count_skelly_and_tagged(monkeypatch):
    def fake_post(
        _base_url: str,
        _query: str,
        _client: httpx.Client,
        *,
        cache_dir=None,
        max_retries: int = 3,
    ) -> dict:
        return {
            "elements": [
                {"type": "node", "id": 1},
                {"type": "way", "id": 2, "tags": {"amenity": "parking_space"}},
            ]
        }

    monkeypatch.setattr("parking_capacity.overpass._post_overpass", fake_post)
    c = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    try:
        n, _ = query_parking_space_count(45.0, 4.0, radius_m=10, client=c, delay_s=0, cache_dir=None)
    finally:
        c.close()
    assert n == 2
