# Notebooks de brouillon (`drafts/`)

Utilisez ce dossier pour des **expérimentations non finalisées** (hyperparamètres, comparaisons rapides). Le livrable « propre » reste `cnn.ipynb` / un futur `final_notebook.ipynb` + les scripts `train_and_save.py` et `scripts/`.

## Suggestions de notebooks à créer

| Fichier suggéré | Contenu |
|-----------------|--------|
| `draft_augmentation.ipynb` | Facteurs ×5 / ×10 / ×15 : courbes train vs val, écart de généralisation. |
| `draft_lr_dropout.ipynb` | Grille learning rate × dropout / weight decay. |
| `draft_resnet_vs_cnn.ipynb` | Courbes côte à côte, temps par epoch, matrice de confusion. |
| `draft_gradcam_samples.ipynb` | Quelques images + cartes Grad-CAM (`src/gradcam.py`). |

## Scripts automatisés (alternative aux notebooks)

- `python scripts/sweep_augment.py --model cnn --factors 5 10 15 --epochs 8`  
  → écrit `output/augment_sweep_cnn.json`
- `python scripts/compare_models.py --epochs 8 --augment 3`  
  → enchaîne MLP, CNN, SmallResNet, ResNet18 TL et produit `output/comparison_table.md`
