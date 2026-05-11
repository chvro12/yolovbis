# Rapport fournisseurs GIS

- Point : `49.729962`, `1.435073` — rayon **80 m**

## Résultats

| Contrôle | Résultat |
|----------|----------|
| IGN WFS joignable | oui |
| BD TOPO bâtiments (échantillon bbox) | oui |
| BD TOPO routes (échantillon bbox) | oui |
| OSM highways (Overpass transport) | oui (7 ways) |
| ArcGIS clé présente | non |
| ArcGIS recherche couches | non |
| Microsoft buildings path | non |
| Mapillary token | non |
| Mapillary ping | non |

### Exemples de noms de voirie OSM

- Avenue des Canadiens
- Grande Rue Saint-Pierre
- Boulevard Industriel
- Rue des Abreuvoirs

## Avertissements

- ARCGIS_API_KEY non configurée — bascule IGN/OSM uniquement.
- MICROSOFT_BUILDINGS_PATH non défini — pas d'empreintes Microsoft locales.
- MAPILLARY_ACCESS_TOKEN non défini — pas d'images Mapillary.