# KaLibre

KaLibre est un petit outil de mesure acoustique et d'analyse audio (sweep ESS, estimation de délai, extraction d'IR, détection de polarité, propositions d'EQ).

Fonctionnalités principales
- Génération d'un sweep ESS (balayage exponentiel)
- Capture loopback + micro et estimation de la fonction de transfert
- Extraction de la réponse impulsionnelle (IR) et estimation du délai acoustique
- Détection d'inversion de polarité entre deux mesures
- Profils EQ de référence et suggestions d'égalisation basiques

Aperçu du fonctionnement
Le pipeline général est :
1. Génération d'un signal de test (sweep ESS) ou capture d'une source.
2. Enregistrement duplex : `loopback` (référence électrique) et `mic` (capté).
3. Calcul spectre / fonction de transfert : $H(f)=M(f) / L(f)$ et déconvolution pour obtenir l'IR.
4. Estimation du délai par corrélation normalisée (améliorable par GCC-PHAT).
5. Alignement des IR et analyse de polarité par corrélation et comparaison +B vs −B.

Fichiers importants
- `main.py` : point d'entrée de l'application (lance l'interface).
- `requirements.txt` : dépendances Python nécessaires.
- `kalibre/core/sweep_analysis.py` : génération du sweep et analyse de la référence acoustique.
- `kalibre/core/signals.py` : génération de signaux, estimation du délai, utilitaires.
- `kalibre/core/polarity_analysis.py` : logique de comparaison de polarité.

Dépendances
Installe les dépendances listées :

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Lancer l'application

```powershell
python main.py
```

Construire un exécutable (Windows)
Tu peux créer un `.exe` autonome avec PyInstaller :

```powershell
python -m pip install pyinstaller
python -m PyInstaller --onefile --windowed main.py
```

L'exécutable sera dans `dist\main.exe`. Ne commite pas `dist/` dans le repo (utilise les Releases GitHub pour distribuer les binaires).

Conseils pour mesures basses fréquences (50 Hz)
- Utilise des sweeps plus longs et une fenêtre IR plus large (la valeur par défaut `ir_window_ms=7` est courte pour 50 Hz).
- Moyenne (Welch) et utilisation de la cohérence améliorent la robustesse en LF.

Améliorations mathématiques possibles
- Remplacer la corrélation brute par GCC-PHAT pour estimer le délai (plus robuste au bruit).
- Moyennage de Welch pour stabiliser l'estimation de la fonction de transfert H(f).
- Régularisation (Tikhonov) lors de la déconvolution pour réduire le bruit dans l'IR.

Bonnes pratiques Git
- N'ajoute pas `dist/`, `build/`, `*.spec`, ni `*.exe` au repo.
- Ajoute un `.gitignore` (exemple donné dans `.gitignore` du projet).

Licence et contributions
Ajoute un fichier `LICENSE` si tu veux expliciter la licence. Contributions bienvenues via pull requests.

---

Si tu veux, je peux aussi :
- ajouter un `README` en anglais, ou
- committer ces fichiers et pousser sur GitHub pour toi (si tu veux que je crée le commit localement).