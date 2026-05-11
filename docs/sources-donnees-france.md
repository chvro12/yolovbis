# Sources de données France — capacité / emprises parking (entraînement & évaluation)

Document de travail pour constituer des **paires (orthophoto ↔ capacité ou masque)** et évaluer des modèles. Vérifier systématiquement **licences**, **CGU** et **mise à jour** sur les pages officielles avant tout téléchargement massif.

## 1. Capacité « tabulaire » (labels forts ou semi-forts)

| Source | Contenu utile | Intérêt ML |
|--------|----------------|------------|
| [transport.data.gouv.fr — recherche « stationnement »](https://transport.data.gouv.fr/datasets?q=stationnement) | Nombreux jeux **parcs de stationnement** (CSV, GeoJSON, NeTEx) par métropole / exploitant | Joindre **géométrie + nb de places** (ou champs du schéma national) pour superviser ou valider |
| [Schéma national « stationnement » (etalab)](https://schema.data.gouv.fr/etalab/schema-stationnement) | Champs type **nb_places**, PMR, vélo, hauteur max, etc. | Normaliser les CSV locaux avant fusion |
| [Doc stationnement hors voirie (PAN)](https://doc.transport.data.gouv.fr/type-donnees/lieux-de-stationnement/stationnement-hors-voirie) | Cadre juridique / bonnes pratiques de publication | Comprendre la couverture attendue |
| [Parkings Saemes (IDF)](https://transport.data.gouv.fr/datasets?q=saemes+parkings) (résultat recherche PAN) | Référentiel / capacités côté exploitant IDF | Labels **France dense** si géométrie ou adresse exploitable |
| [Parkings Indigo (NeTEx / json)](https://transport.data.gouv.fr/datasets?q=indigo+parkings) | Offre nationale exploitant | Même usage : jointure spatiale ou par identifiant |
| Jeux métropole (ex. [Saint-Étienne Métropole — parkings](https://transport.data.gouv.fr/datasets?q=saint-etienne+parkings), [La Rochelle](https://transport.data.gouv.fr/datasets?q=la+rochelle+parkings), [Bordeaux aéroport / métropole](https://transport.data.gouv.fr/datasets?q=bordeaux+parking)) | Souvent **capacité + localisation** | Bons blocs pour **fine-tuning régional** |
| [Base nationale des lieux de stationnement hors voirie (BNLS)](https://transport.data.gouv.fr/datasets/base-nationale-des-lieux-de-stationnement) | Fichier consolidé (historique) | **Obsolète / non maintenue** — utile seulement comme **point de départ** ou croisement, pas comme vérité à jour |
| [Parkings et stationnements (data.gouv, agrégat OSM)](https://www.data.gouv.fr/datasets/parkings-et-stationnements) | Parkings OSM avec champ type `capacity` | **Couverture nationale** mais qualité variable ; utile pour **pseudo-labels** ou compléter les trous |

**Idée de pipeline** : moissonner l’API liste des datasets PAN + `data.gouv.fr` API, filtrer les ressources géo + colonnes capacité, harmoniser avec le schéma stationnement, puis **jointure spatiale** (buffer / intersection) avec parcelle ou polygone parking.

## 2. Emprises / masques (segmentation, pas toujours la capacité)

| Source | Contenu | Intérêt ML |
|--------|---------|------------|
| [Emprises des parkings > 1500 m² — data.iledefrance.fr](https://data.iledefrance.fr/explore/dataset/emprises-parkings-plus-de-1500-m) | Polygones issus d’**IA sur orthophoto IGN** (méthode documentée côté producteur) | **Masques « parking »** sur la France la plus dense ; pas forcément **nombre de places** — combiner avec capacité tabulaire ou comptage |
| [Stationnement sur voie publique — emprises (IDF)](https://data.iledefrance.fr/explore/dataset/stationnement-sur-voie-publique-emprises) | Emprises voirie | Autre tâche (voirie), utile si vous élargissez le périmètre |

## 3. Imagerie (entrée modèle)

| Source | Accès | Notes |
|--------|--------|------|
| [BD ORTHO — Géoservices IGN](https://geoservices.ign.fr/bdortho) | [Téléchargement par zone / département](https://geoservices.ign.fr/telechargement-api/BDORTHO), WMS/WMTS ([Géoplateforme `data.geopf.fr`](https://cartes.gouv.fr/aide/fr/guides-utilisateur/utiliser-les-services-de-la-geoplateforme/diffusion/wms-raster/)) | Résolution courante **~20 cm** ; **licence et conditions** à respecter ; volumes importants pour entraînement national |
| Orthophotos **open data locales** (métropoles, `data.gouv.fr`) | Souvent GeoTIFF / WMS | Parfois **plus fin** que la BD ORTHO sur une zone cible |

## 4. Référentiels géographiques (jointure)

| Source | Usage |
|--------|--------|
| [API Adresse — BAN](https://adresse.data.gouv.fr/outils/api-doc) | Adresse → coordonnées |
| [APICarto — parcelle](https://apicarto.ign.fr/api/doc/cadastre) | Point → parcelle(s) |
| [OpenStreetMap France (extrait Geofabrik)](https://download.geofabrik.de/europe/france.html) | `amenity=parking`, `capacity`, géométries complètes hors flux Overpass | Entraînement / stats **hors ligne** ; licence **ODbL** |

## 5. Jeux internationaux (pré-entraînement / transfert)

| Jeu | Rôle |
|-----|------|
| [APKLOT](https://github.com/langheran/APKLOT) | Segmentation de blocs parking aériens |
| [DLR SkyScapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes) | Segmentation sémantique aérienne HD (dont classes parking) |
| [SegFormer parking (HF)](https://huggingface.co/UTEL-UIUC/SegFormer-large-parking) | Poids déjà utilisés dans ce dépôt | **Point de départ** ; à affiner sur patchs BD ORTHO + masques français |

## 6. Outils intégrés au projet (`parking-capacity`)

Après installation (`pip install -e .`) :

```bash
# Catalogue des ressources (PAN filtré + recherche data.gouv), export CSV ou JSONL
parking-capacity catalog -o data/catalog_stationnement.csv

# Même chose sans interroger data.gouv (plus rapide)
parking-capacity catalog -o data/catalog_pan_seul.csv --no-datagouv

# Télécharger une URL issue du catalogue (limite de taille par défaut 500 Mo)
parking-capacity fetch-resource "https://..." -o data/ma_ressource.zip

# Extraire capacité + coordonnées (heuristique) depuis les ressources du catalogue
parking-capacity harvest-labels -c data/catalog_stationnement.csv -o data/harvested_labels.csv --max-files 50

# Puces orthophoto (BD ORTHO WMS) pour entraînement ML — une image par ligne avec lon/lat
parking-capacity build-chips -i data/harvested_labels.csv -d data/chip_dataset --max-rows 200

# Entraînement + évaluation (régression capacité)
parking-capacity train-model -d data/ml_run --synthetic-n 500 --epochs 20 --lr 0.01
parking-capacity eval-model -c data/ml_run/model.pt --chip-dir data/ml_run/chips
```

`harvest-labels` ne couvre pas encore NeTEx / XML complexes : privilégier CSV / GeoJSON dans le catalogue ou après `fetch-resource`.

Les colonnes principales du catalogue : `source`, `dataset_page_url`, `resource_format`, `resource_url`, etc.  
Respectez les **CGU** et **licences** avant tout téléchargement massif.

## 7. Ce qui manque encore pour un « bon score » capacité

1. **Table de vérité** : table unique `{id, centroid ou polygone, capacité_int, source, date}` issue du croisement PAN + métropoles + OSM, avec **dédoublonnage** et **contrôle qualité** échantillon.  
2. **Découpage d’images** : tuiles alignées Lambert 93 ou Web Mercator autour de chaque polygone / adresse, même résolution que l’inférence.  
3. **Tâche claire** : régression (nombre de places) vs segmentation d’emplacements (masques fins) — la deuxième demande plus d’annotation.  
4. **Évaluation** : jeu **hold-out** géographique (ex. métropoles entières réservées au test) pour éviter le sur-apprentissage local.

---

*Dernière mise à jour du fichier : exploration automatisée (recherche web + PAN). Les URLs et métadonnées peuvent changer : vérifier sur le site producteur.*
