# Fournisseurs GIS (IGN, OSM, Microsoft, Mapillary)

Ce document décrit les **sources géographiques réelles** branchées dans le dépôt, la configuration et les limites. **ArcGIS / Esri n’est pas intégré** (clés souvent liées à une offre payante ou à des quotas).

**France métropolitaine** : les sources **principales** exploitées par défaut sont **IGN BD TOPO (WFS) + orthophoto IGN** et **OSM / Overpass**. **Mapillary** et **Microsoft Building Footprints** restent **optionnels** (token ou fichier local).

## Fichiers de configuration

- **`providers.yaml.example`** : modèle à copier vers `providers.yaml` (ou chemin passé à `--providers-yaml`, ou variable d’environnement `PARKING_PROVIDERS_YAML`).
- **Secrets** : à mettre dans `.env` (ou l’environnement du processus). Le code lit `os.environ` ; la CLI charge `.env` au démarrage si `python-dotenv` est installé.

## 1. IGN Géoplateforme / BD TOPO

- **Service** : WFS public `https://data.geopf.fr/wfs/ows` (GetCapabilities officiel).
- **Implémentation** : `src/parking_capacity/ign_geoplateforme.py` — `GetFeature` en `outputFormat=application/json`, bbox `EPSG:4326`, cache disque sous `<cache_dir>/ign_wfs/*.geojson`.
- **Couches utilisées par défaut** (noms WFS BD TOPO v3) :
  - `BDTOPO_V3:batiment` — emprises bâtiments ;
  - `BDTOPO_V3:troncon_de_route` — axes routiers ;
  - (optionnel YAML) `BDTOPO_V3:zone_d_activite_ou_d_interet` — zones d’activité.
- **BBOX WFS** : pour ce service IGN en `EPSG:4326`, le paramètre attendu est **minLat, minLon, maxLat, maxLon** (sud–ouest–nord–est), pas lon puis lat sur chaque coin.
- **WMS orthophoto** : inchangé (`imagery_wms.py`, `data.geopf.fr/wms-r`, couche BD ORTHO).
- **WMTS / téléchargement API** : non implémentés dans ce client minimal ; les flux vecteur passent par **WFS** + cache local. Pour des volumes massifs, privilégier les **téléchargements** officiels (paquets départementaux) puis import local — même logique que Microsoft Footprints.

## 2. API Carto IGN (APICarto)

- **Parcelles** : `https://apicarto.ign.fr/api/cadastre/parcelle` (déjà utilisé dans `cadastre.py`).
- **Autres modules** : urbanisme, risques, etc. — chacun a son schéma, ses conditions d’usage et des quotas ; consulter [la doc API Carto](https://apicarto.ign.fr/api/doc) avant d’étendre le client.

## 3. OpenStreetMap / Overpass

- **Parkings classiques** : `overpass.py` (`amenity=parking`).
- **Transport / voirie / accès** : `osm_transport.py` — union de requêtes `around:` pour `highway=*` (dont `service=parking_aisle`, `service=driveway`), `amenity=parking_entrance`, `parking=surface|street_side`, `building`, etc.
- **URL** : configurable (`overpass_url` dans YAML, défaut `https://overpass-api.de/api/interpreter`). Respecter la charge publique (délais, cache HTTP déjà utilisé).

## 4. Microsoft Building Footprints (optionnel, local)

- Les jeux **Building Footprints** sont surtout des **téléchargements** (GeoJSON / fichiers régionaux), pas une API live unique.
- **Variable** : `MICROSOFT_BUILDINGS_PATH` — chemin vers un fichier `.geojson` ou un répertoire contenant un `.geojson` (premier trouvé).
- **Code** : `microsoft_buildings.py` — intersection bbox → raster sur puce. Les très gros fichiers (> ~120 Mo) sont refusés en mémoire : découper un extrait régional.

## 5. Mapillary (optionnel)

- **Variable** : `MAPILLARY_ACCESS_TOKEN`.
- **Code** : `mapillary_provider.py` — requête d’images dans une bbox (usage : vérification visuelle rue / accès).
- **Sans token** : pas d’appel.

## 6. Fusion réelle dans le pipeline (`gis_fusion.py`)

Le module **`src/parking_capacity/gis_fusion.py`** construit les masques alignés sur la puce orthophoto et les métriques d’accès (distance au réseau routier en mètres, score réseau, cohérence multi-sources). La commande `check-gis-providers` valide la connectivité ; **`process_address` / `diagnose-address` consomment les mêmes primitives** via `fetch_chip_gis_augmentation` (alias de `build_gis_fusion` dans `gis_context.py`).

### Ordre de priorité — masque **bâtiments**

1. **BD TOPO** `BDTOPO_V3:batiment` (WFS) ;
2. sinon polygones **OSM** `building=*` (Overpass transport) ;
3. sinon **Microsoft Building Footprints** local (`MICROSOFT_BUILDINGS_PATH` + `microsoft_buildings.enabled` dans YAML) ;
4. sinon **heuristique image** (`semantic_layer`).

Champ associé : `building_mask_source` = `bdtopo` | `osm` | `microsoft` | `heuristic`.

### Ordre de priorité — **routes** (chaussée)

1. **OSM** highways / services (Overpass) ;
2. **BD TOPO** `troncon_de_route` (WFS) ;
3. fusion par **OU logique** avec la chaussée détectée sur l’orthophoto (`surface_classification`).

Champ associé : `road_source` = `bdtopo` | `osm` | `bdtopo+osm` | `image_heuristic` | `none`.

### Accès véhicule (France)

- `road_connection_detected` est **vrai** si la bbox puce **touche** le réseau BD TOPO/OSM en projection métrique **ou** si la distance (centre de puce et point adresse) au réseau fusionné est **inférieure au seuil** `fusion.access_distance_threshold_m` (défaut 40 m dans `providers.yaml.example`).
- `access_distance_m` et `road_network_score` sont dans le bloc JSON **`access`** de `result.json`.

### Surfaces « fusion » (GIS + vision)

- `usable_parking_area_m2` : bitume éligible **hors** emprise bâtiments officielle ;
- `final_parking_candidate_area_m2` : zone utile intersectée avec la chaussée (image et/ou routes GIS dilatées) ;
- `semantic_consistency_score` : score 0–1 (BD TOPO, OSM, connexion route, optionnel Mapillary).

### Plafond capacité produit

- `fusion.max_plausible_capacity_slots` (défaut **39**, donc capacité plausible **&lt; 40** places) borne `plausible_capacity_ceiling` après calcul surface / 12 m².

### PNG diagnostic (`diagnose-address`)

Si la puce est réelle et `chip_geo` est présent dans `raw_debug`, le diagnostic régénère les couches et peut écrire :  
`debug_bdtopo_buildings.png`, `debug_bdtopo_roads.png`, `debug_osm_roads.png`, `debug_gis_fusion.png`, `debug_final_parking_candidate.png`.

## 7. Commandes

```bash
# Vérification des fournisseurs (écrit providers_report.md, providers_raw.json, debug_gis_layers.png)
parking-capacity check-gis-providers --lat 49.729962 --lon 1.435073 --radius-m 80 --out ./gis_report

# Diagnostic adresse avec même rayon et config YAML
parking-capacity diagnose-address "2 Bd Industriel, 76270 Neufchâtel-en-Bray" --radius-m 80 --out ./diag_neufchatel --source-priority hybrid --cache-dir data/.cache_http
```

## 8. Gratuit / payant / optionnel

| Source | Coût typique | Obligation |
|--------|----------------|------------|
| IGN WFS / WMS / APICarto | Public gratuit (respect CGU / quotas) | Non (défaut activé) |
| Overpass | Public gratuit (usage raisonnable) | Non |
| Microsoft Footprints | Données ouvertes, hébergement local | Optionnel |
| Mapillary | Compte + token | Optionnel |

## 9. Variables d’environnement utiles

| Variable | Rôle |
|----------|------|
| `PARKING_PROVIDERS_YAML` | Chemin vers `providers.yaml` |
| `fusion.*` (dans YAML) | `access_distance_threshold_m`, `max_plausible_capacity_slots` |
| `MICROSOFT_BUILDINGS_PATH` | Fichier ou dossier GeoJSON empreintes |
| `MAPILLARY_ACCESS_TOKEN` | Jeton API Mapillary |
