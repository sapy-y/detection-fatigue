# 😴 Détection de fatigue par webcam

Programme Python qui surveille les yeux en temps réel avec la webcam et **déclenche une alarme sonore si les yeux restent fermés plus de 2 secondes**, signe d'endormissement (par exemple au volant ou pendant une longue séance de travail).

Projet réalisé avec **OpenCV** et **MediaPipe**, construit pas à pas en 8 étapes.

---

## ✨ Fonctionnalités

- 🎥 Analyse de la webcam en temps réel
- 👁️ Détection du visage et des yeux (478 points avec MediaPipe Face Landmarker)
- 📐 Calcul de l'ouverture des yeux (EAR, *Eye Aspect Ratio*)
- 🎯 **Calibration automatique** : le seuil s'adapte aux yeux de chaque personne
- 🚨 **Alerte** : écran rouge et alarme sonore en boucle après 2 s d'yeux fermés
- 📊 Statistiques en direct : nombre de clignements, clignements par minute, nombre d'alertes
- 📈 Courbe de l'EAR en temps réel, avec la ligne du seuil

---

## 🧠 Comment ça marche

### 1. Repérer les yeux
MediaPipe détecte 478 points sur le visage. On en garde **6 par œil** :

```
        p2    p3
  p1 ●──●─────●──● p4      p1, p4 = coins de l'œil (largeur)
        p6    p5           p2-p6, p3-p5 = haut/bas (hauteur)
```

### 2. Mesurer l'ouverture de l'œil (EAR)

```
        |p2 − p6| + |p3 − p5|
EAR = ─────────────────────────
           2 × |p1 − p4|
```

| État de l'œil | EAR |
|---|---|
| Ouvert | ≈ 0,25 – 0,35 |
| Fermé | ≈ 0,10 |

Comme c'est un rapport hauteur / largeur, la valeur ne dépend **pas** de la distance à la caméra.

### 3. Calibrer
Pendant les 3 premières secondes, le programme mesure l'EAR normal de la personne (yeux ouverts) et fixe le seuil à **70 %** de cette valeur. On utilise la **médiane** pour que les clignements pendant la calibration ne faussent pas la mesure.

### 4. Décider
| Durée sous le seuil | Interprétation |
|---|---|
| 0,05 – 0,4 s | Clignement (compté) |
| ≥ 2 s | **Alerte fatigue** 🚨 |

---

## 🛠️ Technologies

| Technologie | Rôle |
|---|---|
| Python 3 | Langage principal |
| OpenCV | Lecture de la webcam, affichage, dessin |
| MediaPipe (Face Landmarker) | Détection des points du visage |
| NumPy | Calculs (distances, médiane) |
| winsound | Alarme sonore (Windows) |

---

## 🚀 Installation

```powershell
git clone https://github.com/<ton-pseudo>/detection-fatigue.git
cd detection-fatigue

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Le modèle `face_landmarker.task` (~4 Mo) est téléchargé automatiquement au premier lancement.

## ▶️ Utilisation

```powershell
python detection_fatigue.py
```

1. Regarde la caméra, **yeux ouverts**, pendant les 3 s de calibration.
2. Ferme les yeux plus de 2 s : l'écran devient rouge et l'alarme sonne.
3. Rouvre les yeux : l'alarme s'arrête.

| Touche | Action |
|---|---|
| `q` | Quitter (affiche le résumé de la session) |
| `c` | Recalibrer (pour une nouvelle personne) |

### Réglages (en haut du fichier)

| Variable | Défaut | Effet |
|---|---|---|
| `RATIO` | `0.7` | Seuil = RATIO × EAR normal. Plus haut → plus sensible |
| `DUREE_CALIBRATION` | `3.0` | Durée de la calibration (s) |
| `DUREE_ALERTE` | `2.0` | Durée d'yeux fermés avant l'alerte (s) |

---

## 📚 Construction pas à pas

Le dossier `etapes/` contient chaque étape du projet, pour suivre la progression :

| Étape | Fichier | Contenu |
|---|---|---|
| 1 | `etape1.py` | Ouvrir la webcam |
| 2 | `etape2.py` | Détecter le visage et dessiner les 478 points |
| 3 | `etape3.py` | Isoler les 6 points de chaque œil |
| 4 | `etape4.py` | Calculer l'EAR |
| 5 | `etape5.py` | Savoir depuis combien de temps les yeux sont fermés |
| 6 | `etape6.py` | Alerte : écran rouge et alarme sonore |
| 7 | `etape7.py` | Calibration automatique |
| 8 | `etape8.py` | Statistiques et courbe en direct |

---

## ⚠️ Limites

- Fonctionne sous **Windows** (le son utilise `winsound`).
- Nécessite un éclairage correct et un visage bien visible.
- Les lunettes avec reflets peuvent perturber la détection.
- Projet pédagogique : **ce n'est pas un dispositif de sécurité certifié**.

## 🔮 Pistes d'amélioration

- Détecter les bâillements (ouverture de la bouche)
- Détecter la tête qui tombe (inclinaison du visage)
- Compatibilité macOS / Linux pour le son
- Enregistrer les statistiques de chaque session dans un fichier CSV

---

## 🙏 Crédits

- Méthode EAR : Soukupová & Čech, *Real-Time Eye Blink Detection using Facial Landmarks* (2016)
- Modèle de visage : [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Google)
- Son d'alarme : « Alarm Clock » par freesound_community, via [Pixabay](https://pixabay.com/sound-effects/)

## 👩‍💻 Autrice

**Yasmine** — étudiante en développement IA à l'ECE Paris
