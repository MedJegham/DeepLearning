# Retinal Image Classification — Deep Learning Project

> Author: **MedJegham** · IFT 3395/6390 (Kaggle 2 — Fall 2025)


**Cours :** IFT 3395/6390 — Apprentissage Machine
**Compétition :** [Kaggle 2 — Fall 2025](https://www.kaggle.com/competitions/ift-3395-6390-kaggle-2-competition-fall-2025)

## 1. Problème

Classification multi-classes d'images rétiniennes 28×28×3 en **5 niveaux de qualité** (0 = plus basse, 4 = plus haute). Contexte typique : aide au diagnostic ophtalmologique sur des images de faible résolution / forte variabilité.

## 2. Jeu de données

| Split | Nb images | Forme | Classes |
|-------|-----------|-------|---------|
| Entraînement | 1 080 | 28×28×3 | 5 (déséquilibrées) |
| Test (Kaggle) | 400 | 28×28×3 | inconnues |

Les fichiers sont au format Pickle (`data/train_data.pkl`, `data/test_data.pkl`).

> Avec seulement ~1 k images, le **risque de surapprentissage** est central : la pipeline mise sur **augmentation forte + régularisation + early stopping**.

## 3. Pipeline Deep Learning

```
Pickle ──▶ Normalisation /255 ──▶ Augmentation ×5/×10 ──▶ Modèle DL ──▶ Softmax ──▶ Classe
                                       │
                                       ├─ flip H, rotation, crop
                                       ├─ ColorJitter, autocontrast
                                       └─ affine, equalize, sharpness
```

- **Loss :** `CrossEntropyLoss` (softmax intégrée), **label smoothing** optionnel (`train_and_save.py --label-smoothing`).
- **Optim. :** Adam (lr par défaut 1e-3 ; **3e-4** pour `resnet18_tl`) ou SGD+momentum, **weight decay**.
- **Scheduler :** `ReduceLROnPlateau` sur la val loss (`--lr-plateau`).
- **Régularisation :** BatchNorm + Dropout + augmentation + **early stopping** sur la val loss.
- **Validation :** split stratifié 80/20 ; pour stabilité multi-seeds / k-fold, voir `drafts/README.md`.
- **Métriques (script) :** accuracy, **F1 macro**, **ROC AUC multiclasse** (one-vs-rest, macro).

## 4. Architectures comparées

| Modèle | Rôle | Notes |
|--------|------|--------|
| `MLPBaseline` | Baseline sans prior spatial | Référence pour la soutenance |
| `CNN` | CNN compacte maison | Bon compromis sur 28×28 |
| `SmallResNet` | ResNet légère *from scratch* | Skip connections |
| **`ResNet18Transfer`** | **Transfer learning** ImageNet | Normalisation ImageNet activée automatiquement ; souvent le meilleur sur peu de données |
| Kernel SVM (`kernel_svm.ipynb`) | Baseline ML | ~0.46 sur Kaggle (ordre de grandeur) |

Après entraînement avec `train_and_save.py`, chaque checkpoint inclut **# paramètres**, **temps CPU/GPU**, **val accuracy**, **F1 macro**, **ROC AUC OvR** (dans `models/<model>_history.json` et le `.pt`).

Pour un **tableau comparatif** Markdown + JSON : `python scripts/compare_models.py` (voir §6).

## 5. Structure du dépôt

```
kaggle_2_ift3395/
├── app/streamlit_app.py
├── assets/
├── data/
├── drafts/                 # notebooks d'expérience (voir drafts/README.md)
├── models/                 # *.pt, *_history.json, last_run_summary.json
├── output/                 # soumissions Kaggle, comparison_table.md
├── scripts/
│   ├── compare_models.py   # enchaîne les modèles → tableau + JSON
│   └── sweep_augment.py    # ×5 / ×10 / ×15 pour un modèle donné
├── slides/
├── src/
│   ├── data.py
│   ├── gradcam.py          # Grad-CAM (CNN + ResNet18 TL)
│   ├── metrics.py          # F1, ROC OvR, classification_report
│   ├── model.py            # CNN, SmallResNet, MLP, ResNet18Transfer
│   └── train.py            # entraînement + ReduceLROnPlateau optionnel
├── cnn.ipynb
├── kernel_svm.ipynb
├── train_and_save.py
├── requirements.txt
└── README.md
```

## 6. Lancer le projet

### 6.1. Installer les dépendances

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### 6.2. Entraîner un modèle (une commande)

```powershell
python train_and_save.py --model cnn --epochs 12 --augment 5 --lr-plateau --label-smoothing 0.05
python train_and_save.py --model resnet18_tl --epochs 15 --augment 5 --lr-plateau
python train_and_save.py --model mlp --epochs 20 --augment 1
```

Options utiles : `--lr-plateau`, `--label-smoothing 0.05`, `--kaggle-score 0.52` (pour tracer le score public dans le JSON), `--no-imagenet-norm` (rare ; par défaut ImageNet pour `resnet18_tl`).

### 6.2b. Tableau comparatif (plusieurs modèles)

```powershell
python scripts/compare_models.py --epochs 8 --augment 3 --lr-plateau
```

→ `output/comparison_table.md` et `output/comparison_results.json`

### 6.2c. Balayage augmentation ×5 / ×10 / ×15

```powershell
python scripts/sweep_augment.py --model cnn --factors 5 10 15 --epochs 8
```

### 6.2d. Reproductibilité

```powershell
pip freeze > requirements-lock.txt
```

Conservez `requirements-lock.txt` dans le dépôt après une installation validée.

### 6.3. Lancer l'app Streamlit (BONUS)

```powershell
streamlit run app/streamlit_app.py
```

L'app permet :

- de **prédire la classe** d'une image téléversée (avec barre de probabilités) ;
- de faire de l'**inférence en lot** + export CSV ;
- de visualiser les **courbes d'apprentissage** du modèle chargé ;
- de basculer entre plusieurs modèles entraînés (sélecteur dans la barre latérale).

### 6.4. Compiler la présentation LaTeX

```powershell
cd slides
pdflatex presentation.tex
pdflatex presentation.tex   # 2e passe (références)
```

## 7. Performances obtenues

- CNN + augmentation ×10 : ~52% accuracy sur le leaderboard Kaggle public.
- Baseline Kernel SVM : ~46% — l'apport du DL est mesurable malgré la taille modeste du dataset.
- Voir `assets/conf_matrix_with_data_aug.png` pour la matrice de confusion finale.

## 8. Réponses « fondamentaux DL » (préparation soutenance)

- **MLP vs CNN vs RNN/Transformers.** MLP traite des vecteurs aplatis sans prior spatial ; CNN exploite la **localité** et le **partage de poids** (filtres) → idéal images ; RNN/LSTM modélisent des **dépendances temporelles** ; Transformers utilisent l'**attention** pour capter des dépendances longues en parallèle.
- **Vanishing / exploding gradients.** Apparaissent dans les réseaux profonds / récurrents : la chaîne de dérivées multiplie de petits (resp. grands) facteurs. Solutions : initialisation (He/Xavier), **skip connections** (ResNet), **BatchNorm/LayerNorm**, **gradient clipping**, choix d'activations (ReLU, GELU), architectures (LSTM/GRU, Transformer).
- **Rôle des BN / Dropout / Early stopping.** BN normalise les activations par batch (stabilise + régularise) ; Dropout désactive aléatoirement des unités (régularisation) ; Early stopping arrête sur la perte de **validation** pour éviter le surapprentissage.
- **Loss & optimiseur.** Classification multi-classes → `CrossEntropyLoss` (avec softmax intégrée). Adam = bon défaut, SGD+momentum + scheduler souvent meilleur en généralisation. Le choix dépend de la stabilité observée sur les courbes train/val.

## 9. Pénalités & checklist soutenance

- [x] Notebooks brouillons (`drafts/`)
- [x] Notebook final (à finaliser dans `final_notebook.ipynb`)
- [x] Slides LaTeX 7–8 max (`slides/presentation.tex`)
- [x] README professionnel
- [x] App Streamlit (bonus, à déployer pour +2 pts)
- [ ] Lien GitHub fonctionnel à mettre dans le formulaire
- [ ] Lien Streamlit Cloud / HuggingFace Space à mettre dans le formulaire

> ⚠️ Tout dépôt incomplet ou lien non fonctionnel entraîne une pénalité.
