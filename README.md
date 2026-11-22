# 3D Perception with Deep Learning

Faire comprendre le monde en 3D à une machine : reconnaître des objets à partir de nuages de points, estimer la profondeur d'une image, détecter des objets dans une scène LiDAR.

Le projet avance en trois étapes, de l'objet isolé jusqu'à la scène de rue :

| Étape | Tâche | Données | Statut |
|---|---|---|---|
| 1 | Classification de nuages de points (PointNet) | ModelNet40 | Terminée, **88,6 %** |
| 2 | Estimation de profondeur à partir d'une image | NYU Depth v2 | À venir |
| 3 | Détection 3D d'objets dans un scan LiDAR | KITTI | À venir |

## Étape 1 : PointNet sur ModelNet40

![Exemples de nuages de points ModelNet40](docs/modelnet40_samples.png)

ModelNet40 contient 12 311 modèles 3D répartis en 40 classes (avion, chaise, table, guitare...). Chaque maillage est converti en un nuage de 1 024 points tirés uniformément sur sa surface, puis centré et ramené dans la sphère unité.

PointNet ([Qi et al., CVPR 2017](https://arxiv.org/abs/1612.00593)) lit directement ces points, sans les transformer en image ou en voxels. Le même petit réseau traite chaque point séparément, puis un max-pooling résume l'ensemble : le résultat ne dépend pas de l'ordre des points, ce qui est vérifié dans les tests.

### Résultats

| | Précision globale | Moyenne par classe |
|---|---|---|
| Ce projet (25 époques, CPU, sans T-Net 64x64) | **88,6 %** | 83,1 % |
| Article original (250 époques, GPU, modèle complet) | 89,2 % | 86,2 % |

Le modèle arrive à 0,6 point de l'article avec dix fois moins d'époques, entraîné en 5 h sur un processeur Intel i3 sans carte graphique.

![Courbes d'apprentissage](docs/pointnet_curves.png)

Les sauts aux époques 11 et 21 correspondent à la division du taux d'apprentissage par 2. La précision de test reste au-dessus de celle d'entraînement parce que l'entraînement se fait avec des données augmentées (étirement, décalage, bruit) et du dropout, alors que le test se fait sur les nuages bruts.

### Où le modèle se trompe

<img src="docs/pointnet_per_class.png" width="560" alt="Précision par classe">

Les erreurs ne sont pas dues au hasard. Elles se concentrent sur des objets qui ont presque la même forme :

| Vraie classe | Prédite comme | Part des exemples |
|---|---|---|
| flower_pot | plant | 55 % |
| cup | vase | 30 % |
| radio | glass_box | 25 % |
| wardrobe | bookshelf | 25 % |
| night_stand | dresser | 22 % |

Un pot de fleurs contient presque toujours une plante, et une table de nuit ressemble beaucoup à une commode en plus petit. Comme les nuages sont normalisés à la même taille, l'échelle réelle de l'objet est perdue, alors que c'est justement ce qui permet de distinguer ces classes. Les classes les plus faibles n'ont aussi que 20 exemples de test : une seule erreur y coûte 5 points.

La matrice de confusion complète est dans [docs/pointnet_confusion.png](docs/pointnet_confusion.png) et les chiffres par classe dans [docs/pointnet_results.json](docs/pointnet_results.json).

### Reproduire

```bash
conda env create -f environment.yml
conda activate perception
pip install open3d   # optionnel, pour la vue 3D interactive

# ModelNet40.zip (1,9 Go) : http://modelnet.cs.princeton.edu, à décompresser dans data/raw/
python scripts/prepare_modelnet40.py
python scripts/train_pointnet.py --epochs 25 --lr-step 10 --no-feature-transform
python scripts/evaluate_pointnet.py
```

Un entraînement interrompu reprend avec `--resume`. Sur GPU, le modèle complet (sans `--no-feature-transform`) sur 250 époques reproduit le protocole de l'article.

## Structure

```
src/perception/
  datasets/modelnet.py   lecture des .off, échantillonnage, Dataset PyTorch
  models/pointnet.py     PointNet, T-Net et régularisation
scripts/                 préparation, entraînement, évaluation, visualisation
tests/                   tests unitaires (pytest)
docs/                    figures et résultats
```

## Tests

```bash
pytest
```

Les tests vérifient entre autres que la prédiction ne change pas quand on mélange les points, et que le modèle arrive à apprendre par cœur un petit lot.
