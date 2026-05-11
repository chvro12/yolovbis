# Inférence par adresse

## CSV batch

```bash
parking-capacity run -i adresses.csv -o resultats.csv \
  --radius-m 50 --cache-dir data/.cache_http \
  --ml-checkpoint data/runs/parking_resnet18/model.pt --ml-mode fallback
```

Colonnes clés de sortie (en plus des champs historiques) :  
`estimated_capacity`, `min_capacity`, `max_capacity`, `method_used`, `sources_used`,  
`nearby_osm_parkings_count`, `osm_parking_space_count`, `area_total_m2`,  
`baseline_estimate`, `baseline_method`, `ml_vs_baseline_note`, `chip_path` (si sauvegarde demandée côté API — le CLI CSV standard ne sauvegarde pas l’image ; l’UI enregistre un temporaire pour aperçu).

## Une seule adresse (stdout JSON)

```bash
parking-capacity run-address "38 rue du Moulin à Vent, Paris" --radius-m 50 \
  --ml-checkpoint data/runs/parking_resnet18/model.pt
```

## Logique métier (résumé)

1. Géocodage BAN.  
2. Overpass dans `--radius-m` : parkings `amenity=parking` + comptage `amenity=parking_space`.  
3. Si somme `capacity` OSM > 0 : **priorité** à ce signal.  
4. Sinon, si **priorité orthophoto** active (`aerial_first`, défaut) : orthophoto centrée (emprise ≥ rayon), **SegFormer**, puis **ML** selon `--ml-mode`, puis **surface / ratios m²·place⁻¹**, puis **comptage parking_space**.  
5. Une **baseline** par priorité OSM → surface → vision → ML est calculée pour comparaison honnête avec le ML.

## UI

```bash
pip install -e ".[ui]"
parking-capacity ui
```

Rayon, orthophoto aperçu, modes ML, avertissements.
