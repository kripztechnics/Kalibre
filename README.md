# KaLibre

Salut à toi caleur de sonorisation. 

KaLibre est un petit outil de mesure acoustique et d'analyse audio (Mise en phase de système automatique, universel et fait pour des utilisation de sono DIY).

J'ai créé ce programme pour toutes ces personnes qui se lancent dans la sonorisation et le partage de musique via amplification,
qui n'ont pas de conaissances particulières dans l'utilisation de certain logiciel type Smaart ou REW. 

Je mets mon discords au cas ou il y a des demandes sur l'utilisation ou certain problème rencontrés :)


Fonctionnalités principales
- Génération d'un sweep ESS (balayage exponentiel)
- Capture loopback + micro et estimation de la fonction de transfert
- Extraction de la réponse impulsionnelle (IR) et estimation du délai acoustique
- Détection d'inversion de polarité entre deux mesures
- Profils EQ de référence et suggestions d'égalisation basiques

Aperçu du fonctionnement
L'utilisation générale est la suivante :
1. Génération d'un signal de test (sweep ESS) ou capture d'une source.
2. Mesures par le Micro
3. Calcul spectre / fonction de transfert : $H(f)=M(f) / L(f)$ et déconvolution pour obtenir l'IR.
4. Estimation du délai par corrélation normalisée.
5. Alignement des IR et analyse de polarité par corrélation et comparaison +B vs −B.

Fichiers importants
- `main.py` : point d'entrée de l'application (lance l'interface).
- `requirements.txt` : dépendances Python nécessaires.
- `kalibre/core/sweep_analysis.py` : génération du sweep et analyse de la référence acoustique.
- `kalibre/core/signals.py` : génération de signaux, estimation du délai, utilitaires.
- `kalibre/core/polarity_analysis.py` : logique de comparaison de polarité.

COMMENT L'INSTALLER ??? tu vas me dire 

Installe les dépendances listées :

AVANT TOUT IL TE FAUT UNE VERSION DE GIT pour pouvoir télécharger le repos :

https://git-scm.com ( télécharge la dernière version et suit le setup d'installation classique) 

Une fois ceci fait tu peux télécharger le repos : 

Dans un CMD (clique droit sur le bureau "Ouvrir dans le terminal")  tape les commandes une après les autres suivantes : 

git clone https://github.com/kripztechnics/Kalibre.git

cd Kalibre

python -m venv .venv

.\.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

~~ Pour lancer l'appli une fois toutes les dépendances installées : 

python main.py

Il ne vous reste plus qu'à mesurer et envoyer la frappe. 

Bonne soirée/journée 

Kripziii 
