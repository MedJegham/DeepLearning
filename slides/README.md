# Présentation soutenance

Slides **Beamer** (thème **Boadilla** + couleurs personnalisées) — **8 diapositives** — pour une soutenance d’environ **2 minutes** de parole.

## Compilation

```bash
cd slides
pdflatex presentation.tex
pdflatex presentation.tex
```

Produit `presentation.pdf`. Les figures sont dans `../assets/` (compiler depuis `slides/`).

**Paquets** : `microtype`, `tikz` (bibliothèques `positioning`, `arrows.meta`, `calc`). **FontAwesome** n’est plus requis (meilleure portabilité MiKTeX / Overleaf).

## Avant la soutenance

- Remplace **« Votre nom »** et l’institut si besoin (`\author`, `\institute`).
- Complète la ligne **ResNet18 TL** du tableau avec les chiffres issus de `output/comparison_table.md` ou de ton entraînement.
- Mets à jour la phrase sur la **ROC** si tu as une valeur exacte depuis `sklearn`.

## Plan des slides

1. **Titre** — résumé en une ligne (pipeline, métriques, Streamlit).
2. **Problème & données** — tâche, volumes, déséquilibre, métriques.
3. **Pipeline** — schéma TikZ + bullets (augmentation, régularisation, optimiseur).
4. **Architecture CNN** — blocs + schéma + référence Nature.
5. **Comparaison** — tableau pro + interprétation.
6. **Courbes** — surapprentissage vs. généralisation.
7. **Matrices de confusion** — sans / avec augmentation.
8. **Conclusion** — livrables, limites, commande Streamlit, Q&R.
