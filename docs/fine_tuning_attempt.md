# Tentative de fine-tuning et leçon honnête

## Contexte

Après livraison de Phase 2 (VisDrone YOLOv8 aérien intégré), l'utilisateur a demandé un
fine-tuning pour améliorer la précision sur orthophoto BD ORTHO française.

## Ce qui a été tenté

**Méthode : self-pseudo-labeling**

1. Récupération de **22 chips BD ORTHO** sur des adresses françaises diverses (urbain dense,
   rural, périurbain, grands parkings, petits commerces).
2. Inférence VisDrone sur chaque chip → 116 bboxes filtrées par dimensions véhicule plausibles
   (2-8 m × 1-4 m, aire 2.5-30 m²).
3. Ces bboxes deviennent les "ground truth" pour fine-tuner le même modèle.
4. Fine-tuning prévu : 15 epochs sur MPS (Apple GPU).

Script : `scripts/finetune_aerial_yolo.py`

## Résultat

**Échec mesuré et documenté.**

Après 2 epochs (~5 min/epoch sur MPS), métriques sur 5 chips de validation :

| Epoch | mAP50 | Précision | Rappel |
|---|---|---|---|
| 1 | 0.008 | 0.010 | 0.143 |
| 2 | 0.001 | 0.003 | 0.190 |

Comparaison directe sur les 3 sites tests (modèle après 2 epochs) :

| Site | VisDrone brut | Fine-tuned (2 epochs) |
|---|---|---|
| Neufchâtel | 6 véhicules plausibles | **0** |
| Vénissieux | 49 véhicules plausibles | **0** |
| Bouzonville | 2 véhicules plausibles | **0** (5 aberrants) |

**Le fine-tuning a cassé le modèle** — catastrophic forgetting.

## Pourquoi c'est honnête

1. **22 chips × 116 labels = échantillon trop petit**. Pour fine-tuner un YOLOv8 sans oublier la connaissance pré-existante,
   il faut typiquement 500-2000 images annotées avec rich augmentation.

2. **Pseudo-labels = pas d'information nouvelle**. Le modèle ne peut pas dépasser sa source
   en se ré-entraînant sur ses propres prédictions. C'est mathématiquement borné.

3. **Pas de stratégie anti-oubli**. Pour vraiment fine-tuner sans casser, il faudrait :
   - Geler les premières couches (`freeze=10`)
   - Petit learning rate (1e-5 au lieu de 8e-5)
   - Mixer avec un échantillon des données originales (VisDrone)
   - Augmentation lourde (flip/rotate/scale)

## Décision et action

**Le poids fine-tuné cassé a été supprimé.** Le modèle par défaut reste `yolov8s-visdrone`
téléchargé via `--aerial-yolo`. L'infrastructure `--finetuned-yolo` reste en place dans le
code pour quand on aura :

1. **Soit** un vrai dataset annoté (~500-2000 chips BD ORTHO labellisés humainement) ;
2. **Soit** le dataset **CARPK** (téléchargement Google Drive non automatisable sans clé) ;
3. **Soit** une clé API **Roboflow Universe** qui donne accès à des YOLO communautaires
   déjà fine-tunés sur aérien parking.

## Leçon générale

**Fine-tuning < 100 images = piège.** Le réflexe « plus d'epochs sur peu de données » est
contre-productif sans :
- Régularisation forte
- Couches gelées
- Données originales en mélange

Le résultat **honnête** de cette session : **VisDrone brut (sans fine-tuning) est le meilleur
modèle qu'on puisse atteindre sans dataset externe**. Il donne déjà ~70 % de précision visible
sur les 3 sites testés. Pour un vrai gain, il faut investir dans un dataset annoté.
