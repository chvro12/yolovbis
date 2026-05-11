# Sources de données et jeux publics (parking / vision)

## France — services utilisés par ce dépôt

- **BAN** — géocodage : https://adresse.data.gouv.fr/
- **APICarto** — parcelles : https://apicarto.ign.fr/
- **Overpass** — OSM : https://wiki.openstreetmap.org/wiki/Overpass_API  
  Tags utiles : `amenity=parking`, `amenity=parking_space`, `capacity=*`, `capacity:disabled=*`, `parking=*`.
- **IGN Géoplateforme WMS** — orthophoto BD ORTHO : `https://data.geopf.fr/wms-r` (couche `ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO`).

## Jeux « caméra / surveillance » (occupation, pas capacité ortho)

- **PKLot** (~12k vues fixes, occupation / météo) — utile pour **pré-entraînement** texture / slot detection, **pas** pour prédire la capacité déclarative d’un parking vu du ciel.  
  Références : page UFPR https://web.inf.ufpr.br/vri/databases/parking-lot-database/ , jeu Hugging Face https://huggingface.co/datasets/ramzz22/PKLot  
- **CNRPark / CNRPark-EXT** — patches caméra, occupation ; même limite pour le besoin métier « capacité depuis orthophoto ».

## Jeux plus proches du satellite / segmentation aire

- **APKLOT** et travaux similaires (segmentation de blocs parking sur images aériennes / satellite) — plus alignés avec une chaîne orthophoto + masque + estimation de places. Vérifier **licence** et **résolution / domaine** avant fusion avec des labels OSM France.

## Hugging Face / Roboflow / Kaggle

- Nombreux jeux « parking space detection » ; distinguer **occupation**, **détection de marquage**, et **capacité légale / déclarative**. Toujours vérifier la **licence** (usage commercial, redistribution).

## Signal souvent le plus fiable en open data

Lorsqu’il est présent et cohérent avec la géométrie, le tag OSM **`capacity=*`** reste souvent le meilleur signal **gratuit** pour une aire donnée — mais il est **incomplet** en France : d’où l’intérêt de l’orthophoto + vision + ML en secours.
