"""Pipeline avec clients HTTP mockés."""

import httpx

from parking_capacity.pipeline import process_address


def _mock_client_factory(ban_json, parcel_json, overpass_json):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api-adresse.data.gouv.fr" in url:
            return httpx.Response(200, json=ban_json)
        if "apicarto.ign.fr" in url and "parcelle" in url:
            return httpx.Response(200, json=parcel_json)
        if "overpass-api.de" in url or "interpreter" in url:
            return httpx.Response(200, json=overpass_json)
        return httpx.Response(404, text="not mocked")

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, timeout=30.0)


def test_pipeline_osm_capacity_no_vision():
    ban_json = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [2.3, 48.85]},
                "properties": {"label": "Test", "score": 0.9},
            }
        ]
    }
    parcel_json = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2.29, 48.84], [2.31, 48.84], [2.31, 48.86], [2.29, 48.86], [2.29, 48.84]]],
                },
                "properties": {"id": "0000000000000000"},
            }
        ],
    }
    overpass_json = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"amenity": "parking", "capacity": "42"},
                "geometry": [
                    {"lat": 48.849, "lon": 2.300},
                    {"lat": 48.849, "lon": 2.301},
                    {"lat": 48.851, "lon": 2.301},
                    {"lat": 48.851, "lon": 2.300},
                    {"lat": 48.849, "lon": 2.300},
                ],
            }
        ]
    }

    client = _mock_client_factory(ban_json, parcel_json, overpass_json)
    r = process_address(
        "1 rue fictive Paris",
        client=client,
        overpass_delay_s=0.0,
        use_vision=False,
        min_intersection_m2=1.0,
    )
    assert r.error is None
    assert r.primary_source == "osm_parcelle"
    assert r.primary_capacity == 42
