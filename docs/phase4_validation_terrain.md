# Phase 4 — Validation terrain

## Objectif

Mesurer l'erreur réelle du système contre des **comptages humains**, identifier les
typologies où il marche / échoue, et calibrer les seuils de confiance.

## Format d'entrée

Un CSV avec au minimum :

```csv
address,human_count,site_type,notes
"2 Bd Industriel, 76270 Neufchâtel-en-Bray",20,clinique_vetrinaire,parking façade
"50 avenue de France, 75013 Paris",,résidentiel,à compter
"Centre commercial Beaugrenelle, 75015 Paris",350,centre_commercial,
```

- `address` (obligatoire) : adresse à analyser.
- `human_count` (obligatoire pour métriques) : capacité comptée par un humain.
- `site_type` (optionnel) : typologie pour segmenter les résultats.
- `notes` (optionnel) : commentaires libres.

## Commande

```bash
parking-capacity validate-benchmark \
  --input benchmark.csv \
  --out validation_2026_05_15 \
  --providers-yaml providers.yaml \
  --auto-yolo \
  --radius-m 80 \
  --source-priority hybrid
```

Le pipeline tourne sur chaque adresse, accumule les prédictions dans `predictions.csv`, puis
calcule les métriques dans :

- `validation_summary.json` — agrégats globaux.
- `validation_segments.json` — métriques par typologie / confiance / mode visuel / source.
- `validation_failing_cases.json` — top 10 cas en échec (|err| ≥ 15).
- `per_address.csv` — détail ligne par ligne.
- `validation_report.md` — rapport markdown lisible.

## Métriques calculées

| Métrique | Définition | Pourquoi |
|---|---|---|
| **MAE** | Mean Absolute Error = moyenne `|estimé − humain|` | Erreur typique en places |
| **MAPE** | Mean Absolute % Error = moyenne `|estimé − humain| / max(humain, 1)` | Erreur relative |
| **RMSE** | √(moyenne (estimé − humain)²) | Pénalise gros écarts |
| **R²** | 1 − SS_res / SS_tot | Qualité de la régression |
| **Refusal rate** | fraction d'adresses où système refuse (None) | Couverture |
| **Accuracy ±N** | fraction de cas avec `|erreur| ≤ N` | Précision opérationnelle |

**Intervalles de confiance bootstrap 95 %** sur MAE et MAPE (1000 resamples).

## Segmentation automatique

Le rapport segmente par défaut sur les colonnes (si présentes) :

- `site_type` — typologie d'usage (commerce / hôpital / résidentiel / etc.)
- `parking_visual_mode` — `marked_slots` / `unmarked_surface` / `roadside_parking` / `courtyard_parking`
- `semantic_confidence` — `none` / `weak` / `medium` / `strong`
- `primary_source` — `osm_parcelle` / `parking_geometry` / `scenario_*` / etc.
- `primary_confidence` — confiance du primary

Ça permet de voir directement où le système marche : par exemple si MAE est de 2 places
sur `osm_parcelle` (logique, OSM est officiel) mais 25 places sur `scenario_unmarked_surface`,
on sait qu'il faut prioritairement améliorer le scénario unmarked.

## Workflow recommandé

1. **Constituer un échantillon de 30-100 adresses** :
   - 1/3 typologies « faciles » (parkings OSM avec `capacity=*`).
   - 1/3 typologies moyennes (petits commerces, cliniques).
   - 1/3 typologies difficiles (cours d'entreprise, hôpitaux, résidentiel).

2. **Compter humainement** via Google Maps / Street View (5-10 min par adresse).

3. **Lancer la validation** :
   ```bash
   parking-capacity validate-benchmark -i benchmark.csv -o val_v1
   ```

4. **Lire `validation_report.md`** :
   - MAE global → l'erreur typique attendue.
   - Segment `semantic_confidence=strong` → vérifier que ce sous-ensemble a un MAE < 5 places
     (sinon la confidence est mal calibrée).
   - Segment `site_type=commerces` → comparer aux autres typologies.
   - Top cas en échec → identifier les patterns systématiques.

5. **Itérer** :
   - Si la confiance `strong` a un MAE trop élevé → resserrer les seuils dans
     `semantic_layer.py`.
   - Si une typologie échoue → typage adapté dans `parking_scenarios.py` (priors ou facteurs
     d'utilisation différents).
   - Si refus trop fréquent → relâcher les conditions de refus dans `pipeline.py`.

## Reproductibilité

Toutes les sorties JSON contiennent les versions / paramètres utilisés. Lancer la même
validation sur le même CSV produit les mêmes métriques (à hash de cache près).

## Exemple de rapport markdown produit

```markdown
# Rapport de validation terrain

## Vue d'ensemble

- Adresses avec comptage humain : 47
- Prédictions retournées : 31
- Refus système : 16 (34.0 %)
- MAE : 6.84 places (IC 95 % : 5.20–8.50)
- MAPE : 24.3 % (IC 95 % : 18.7–30.1 %)
- RMSE : 11.27 places
- R² : 0.612
- Accuracy ±3 : 38 %
- Accuracy ±5 : 51 %
- Accuracy ±10 : 74 %

## Par segment

### `semantic_confidence`

| Valeur | n / total | refus | MAE | MAPE | R² | ±5 |
|---|---|---|---|---|---|---|
| `strong`   | 12/14 | 14% | 3.21 | 11.2% | 0.83 | 75% |
| `medium`   | 14/18 | 22% | 7.45 | 25.4% | 0.58 | 50% |
| `weak`     | 5/15  | 67% | 12.40| 38.1% | 0.21 | 20% |
```
