# Audit d’architecture (dépôt `parking-capacity`)

## 1. Vue d’ensemble

| Zone | Fichiers principaux |
|------|----------------------|
| CLI | `src/parking_capacity/cli.py` (`run`, `run-address`, `harvest-real-dataset`, `catalog`, `harvest-labels`, `build-chips`, `train-model`, `eval-model`, `ui`) |
| Pipeline adresse | `src/parking_capacity/pipeline.py` |
| Baseline | `src/parking_capacity/baseline.py` |
| Géocodage | `src/parking_capacity/geocode.py` |
| Cadastre | `src/parking_capacity/cadastre.py` |
| Géométrie | `src/parking_capacity/geometry.py` |
| OSM / Overpass | `src/parking_capacity/overpass.py`, `osm_aggregate.py` |
| Moisson dataset réel | `src/parking_capacity/harvest_real_dataset.py` |
| WMS / puces | `src/parking_capacity/imagery_wms.py`, `chip_dataset.py` |
| Vision | `src/parking_capacity/vision_estimate.py` |
| ML | `src/parking_capacity/ml/{dataset,models,train,infer,metrics,geo_split,regression_metrics}.py` |
| Données ouvertes PAN | `src/parking_capacity/data_sources/*` |
| UI | `src/parking_capacity/ui.py` |
| Cache HTTP | `src/parking_capacity/cache_http.py` |
| Tests | `tests/test_*.py` |

## 2. Exécutabilité

- **Sans réseau** : `pytest` (mocks BAN/APICarto/Overpass) — OK.
- **Avec réseau** : `run`, `run-address`, `harvest-real-dataset`, `catalog`, `build-chips` dépendent des CGU et disponibilité des services.
- **Vision / ML lourd** : `pip install -e ".[train]"` pour `torchvision` (ResNet / EfficientNet) ; SegFormer télécharge des poids HF au premier usage.

## 3. Manques / risques (non bloquants pour coder, bloquants pour « vérité terrain »)

| Manque | Impact |
|--------|--------|
| Pas de `model.pt` public validé France | Le ML reste à entraîner et évaluer localement. |
| Labels OSM `capacity` bruyants | Faux positifs / valeurs obsolètes ; filtres dans `harvest-real-dataset`. |
| Pas de split administratif fin (IRIS) | Split `geo` = grille lon/lat arrondie (approximation). |
| Pas de cache WMS fichier dans `imagery_wms` | Cache Overpass POST via `--cache-dir` uniquement. |
| Occupation vs capacité | Jeux type PKLot/CNRPark = occupation ; ce dépôt vise **capacité** ou **places visibles** en orthophoto. |

## 4. Reproductibilité

- `torch`, `numpy`, `pandas` : fixer `--seed` sur `train-model`.
- Overpass / WMS : résultats peuvent varier dans le temps (contributeurs OSM).
