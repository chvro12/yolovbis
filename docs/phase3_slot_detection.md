# Phase 3 — Détection places marquées (pleines + vides)

## Objectif

Compter les **places marquées** sur le parking, **distinguer pleines et vides**, ne pas
dépendre uniquement des voitures observées (qui ne donnent que le « plancher »).

## Trois pistes d'implémentation

`parking_slot_detection.py` essaie en cascade :

### 1. YOLO fine-tuné aérien (`--slot-yolo-weights`)

Poids personnalisés ayant les classes `parking_space_empty`, `parking_space_filled`,
`parking_slot`, ou similaires. SAHI activé automatiquement si disponible.

```bash
parking-capacity diagnose-address "..." \
  --slot-yolo-weights /path/to/yolov8_parking_slots.pt
```

### 2. Roboflow Universe (clé API + identifiant modèle)

Roboflow Universe expose des centaines de modèles communautaires de détection de places.
Inférence via leur API HTTP, sans téléchargement local.

```bash
parking-capacity diagnose-address "..." \
  --roboflow-api-key $ROBOFLOW_API_KEY \
  --roboflow-model-id "workspace/aerial-parking-slots/3"
```

Comment trouver un modèle :
1. Aller sur https://universe.roboflow.com
2. Chercher « aerial parking », « parking spot detection », « parking slot »
3. Filtrer par classes (`empty` / `filled`) et résolution proche de BD ORTHO (0.2 m/px)
4. Récupérer l'identifiant `workspace/project/version`

### 3. Heuristique véhicules + rangées (**fonctionne sans poids**)

Cette piste **fonctionne dès maintenant** avec ce qui est déjà calculé par le pipeline :

- Récupère les **rangées géométriques** détectées par `parking_geometry.py` (longueur + orientation).
- Récupère les **véhicules** détectés par `vehicle_detection.py` (Phase 2).
- Pour chaque rangée :
  - `places_total = longueur_m / slot_width_typ` (slot_width = 2.5 m).
  - `places_pleines = nb_véhicules_dans_la_rangée` (assignés par proximité à l'axe).
  - `places_vides = places_total − places_pleines`.
- Applique le **plafond physique** de la couche sémantique.
- Filtre par **polygone parcelle** si fourni.

Limites :
- Dépend de la qualité de la détection rangées (peut être bruitée sur orthophoto réelle).
- Sans Phase 2 fine-tunée, le compte de véhicules est approximatif → places pleines aussi.
- Champs synthétiques dans `result.json` ; PNG de debug à venir.

## Sorties

Dans `result.json` :

```jsonc
{
  "slots_total_count": 24,
  "slots_filled_count": 10,
  "slots_empty_count": 14,
  "slot_detection_method": "heuristic_vehicles_rows",
  "slot_evidence": {
    "total": 24,
    "filled": 10,
    "empty": 14,
    "method": "heuristic_vehicles_rows",
    "notes": []
  }
}
```

Dans la sortie texte humaine :

```
Places marquées (Phase 3) : total=24 (pleines=10, vides=14) — méthode : heuristic_vehicles_rows
```

## Prochaine étape

Brancher un vrai modèle YOLO entraîné sur **PKLot** ou **Aerial Parking Spaces** (Roboflow Universe).
Cette intégration est déjà en place dans le code via les flags `--slot-yolo-weights` ou
`--roboflow-api-key/--roboflow-model-id`.
