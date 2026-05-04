# Rapport de stage M2

Structure active pour travailler d'abord sur le plan :

- `main.tex` : point d'entrée principal du rapport.
- `Rapport_Stage_M2.tex` : wrapper de compatibilité qui pointe vers `main.tex`.
- `plan_detaille.tex` : plan détaillé à compléter avant la rédaction.
- `corps_rapport.tex` : fichier de transition, gardé pour la future version rédigée.
- `bibliographie/references.bib` : bibliographie BibTeX.
- `Images/page_de_garde/` : logo de l'université et autres visuels de couverture.

La configuration LaTeX et les champs de page de garde sont directement dans `main.tex` pour éviter une arborescence trop dispersée pendant la phase de planification.

Compilation conseillée avec `biblatex` et `biber` :

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Ou, si `latexmk` est disponible :

```bash
latexmk -pdf main.tex
```

La page de garde est robuste : si `Images/page_de_garde/logo_bordeaux.png` n'existe pas encore, un cadre de remplacement est affiché à la place du logo.
