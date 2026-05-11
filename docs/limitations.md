# Limitations honnêtes

## Données

- **OSM** : incomplet ; `capacity` peut être faux ou périmé.  
- **Orthophoto** : parkings souterrains, toitures, ombres, arbres masquent la réalité.  
- **SegFormer** : modèle générique ; ce n’est pas une certification urbanistique.  
- **ML régression** : sans volume et labels propres, le modèle **ne remplace pas** la visite terrain.

## Occupation vs capacité

Les jeux **PKLot**, **CNRPark**, etc. portent surtout sur **l’occupation** vue caméra fixe, pas sur la **capacité réglementaire** ou la **capacité visible depuis le ciel** à grande échelle. Ils peuvent aider au **pré-entraînement** de backbone, pas comme unique preuve de qualité pour la France orthophoto.

## Légal / quotas

Respecter les CGU **BAN**, **IGN / Géoplateforme**, **Overpass**. Utiliser `--overpass-delay` et `--cache-dir` pour limiter la charge.

## Ce que l’outil ne fait pas

- Pas d’intégration NeTEx / XML PAN en natif dans ce dépôt (hors heuristiques `harvest-labels`).  
- Pas de modèle satellite tiers embarqué : l’utilisateur doit fournir checkpoints et jeux compatibles licence.
