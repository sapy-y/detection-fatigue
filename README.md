# Détection de fatigue par webcam

Programme Python qui analyse les yeux en temps réel via la webcam et déclenche une alarme sonore lorsque les yeux restent fermés plus de 2 secondes.

Réalisé avec OpenCV et MediaPipe.

# Fonctionnalités

- Détection du visage et des yeux en temps réel
- Calcul de l'ouverture des yeux Eye Aspect Ratio
- Calibration automatique du seuil selon la personne
- Alerte visuelle et sonore après 2 secondes des yeux fermés
- Comptage des clignements et des alertes
- Courbe de l'EAR affichée en direct

# Principe

MediaPipe Face Landmarker détecte 478 points sur le visage. Six points par œil sont utilisés pour calculer l'EAR, le rapport entre la hauteur et la largeur de l'œil :


```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```


L'EAR vaut environ 0,30 les yeux ouverts et 0,10 les yeux fermés. Étant un rapport il ne dépend pas de la distance à la caméra.

Au démarrage, le programme mesure pendant 3 secondes l'EAR de la personne avec les yeux ouverts  fixe le seuil à 70 % de cette valeur. La médiane pour ignorer les clignements.

Ensuite :
- une fermeture de 0,05 à 0,4 s est comptée comme un clignement ;
- une fermeture de 2 s ou plus déclenche l'alerte.

# Technologies

- Python 3
- OpenCV : webcam, affichage
- MediaPipe : détection des points du visage
- NumPy : calculs
- winsound : alarme sonore (Windows)

# Installation


```
git clone https://github.com/sapy-y/detection-fatigue.git
cd detection-fatigue
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


Le modèle `face_landmarker.task` est téléchargé automatiquement au premier lancement.

# Utilisation


```
python detection_fatigue.py
```


Garder les yeux ouverts pendant les 3 secondes de calibration.

 Touche et Action

 touche q = Quitter 
 touche c = Recalibrer 

Paramètres modifiables en haut du fichier :

 Variable - Défaut - Rôle 

 la variable RATIO, defaut 0.7 et rôle Seuil = RATIO × EAR normal 
 la variable DUREE_CALIBRATION, defaut 3.0 et rôle Durée de la calibration par seconde 
 la variable DUREE_ALERTE, defaut 2.0 et rôle Durée d'yeux fermés avant l'alerte par seconde

# Structure


```
detection_fatigue.py   programme principal
etapes/                construction du projet en 8 étapes
alarme.wav             son de l'alarme
requirements.txt
```


Étapes : 1 webcam, 2 détection du visage, 3 points des yeux, 4 calcul de l'EAR, 5 durée de fermeture, 6 alerte, 7 calibration, 8 statistiques.

# Limites

- Son de l'alarme disponible uniquement sous Windows
- Nécessite un éclairage correct et un visage bien visible
- Projet pédagogique, pas un dispositif de sécurité

# Crédits

- Méthode EAR : Soukupová et Čech, Real-Time Eye Blink Detection using Facial Landmarks, 2016
- Modèle : MediaPipe Face Landmarker (Google)
- Son : "Alarm Clock" par freesound_community, Pixabay

# Auteur

Yasmine, étudiante en développement IA à l'ECE Paris
