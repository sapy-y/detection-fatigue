import os
import time
import urllib.request
import wave
import winsound         
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

DOSSIER = os.path.dirname(os.path.abspath(__file__))
MODELE = os.path.join(DOSSIER, "face_landmarker.task")
URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")

ALARME = os.path.join(DOSSIER, "alarme.wav")


def creer_alarme():
    """Alarme originale : rafales de bips aigus, volume maximal."""
    sr = 44100

    def bip(freq, duree):
        t = np.arange(int(sr * duree)) / sr
        
        s = (np.sin(2 * np.pi * freq * t)
             + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
             + 0.25 * np.sin(2 * np.pi * 3 * freq * t))
        
        env = np.minimum(1, np.minimum(t / 0.005, (duree - t) / 0.02))
        return s * env

    def silence(duree):
        return np.zeros(int(sr * duree))

    rafale = []
    for _ in range(4):                       # 4 paires de bips
        rafale += [bip(1800, 0.08), silence(0.04),
                   bip(2300, 0.08), silence(0.06)]
    son = np.concatenate(rafale + [silence(0.25)])   # petite pause

    son = np.tanh(3 * son / np.abs(son).max())       # compression
    son = (son / np.abs(son).max() * 0.98 * 32767).astype(np.int16)  #100 % du volume

    with wave.open(ALARME, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(son.tobytes())


def demarrer_alarme():
    winsound.PlaySound(ALARME, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)


def arreter_alarme():
    winsound.PlaySound(None, 0)
if not os.path.exists(MODELE):
    print("Téléchargement du modèle…")
    urllib.request.urlretrieve(URL, MODELE)


OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
OEIL_DROIT = [33, 160, 158, 133, 153, 144]
SEUIL_EAR = 0.22        # en dessous œil considéré fermé
DUREE_ALERTE = 2.0       # secondes d'yeux fermés avant l'alerte

def calcul_ear(pts):
    """pts = liste de 6 points (x, y) dans l'ordre p1 → p6"""
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    hauteur = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    largeur = np.linalg.norm(p1 - p4)
    return hauteur / (2 * largeur)


options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODELE),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
detecteur = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
t0 = time.monotonic()
echecs = 0 
ferme_depuis = None
alarme_active = False


while True:
    ok, frame = cap.read()
    if not ok:
        echecs += 1
        if echecs > 30:
            print("Impossible de lire la video")
            break
        continue
    echecs = 0      
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    temps_ms = int((time.monotonic() - t0) * 1000)
    resultat = detecteur.detect_for_video(image, temps_ms)

    if resultat.face_landmarks:
        points = resultat.face_landmarks[0]

        ears = []
        for oeil in (OEIL_GAUCHE, OEIL_DROIT):
            coords = []
            for indice in oeil:
                p = points[indice]
                coords.append((int(p.x * w), int(p.y * h)))
            for i in range(6):
                cv2.line(frame, coords[i], coords[(i + 1) % 6], (0, 255, 255), 1)
            ears.append(calcul_ear(coords))

        ear = (ears[0] + ears[1]) / 2
        maintenant = time.monotonic()

        if ear < SEUIL_EAR:                       # yeux fermés
            if ferme_depuis is None:              # ils viennent de se fermer
                ferme_depuis = maintenant
            duree = maintenant - ferme_depuis
            etat, couleur = f"Yeux FERMES : {duree:.1f} s", (0, 165, 255)
            if duree >= DUREE_ALERTE:
                rouge = frame.copy()
                rouge[:] = (0, 0, 255)
                frame = cv2.addWeighted(rouge, 0.35, frame, 0.65, 0)
                cv2.putText(frame, "ALERTE FATIGUE !", (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
                if not alarme_active:          # on ne la lance qu'une fois
                    demarrer_alarme()
                    alarme_active = True
        else:                                     # yeux ouverts
            ferme_depuis = None
            if alarme_active:
                arreter_alarme()
                alarme_active = False                   # on remet le chrono à zéro
            etat, couleur = "Yeux ouverts", (0, 255, 0)

        cv2.putText(frame, etat, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, couleur, 2)
        cv2.putText(frame, f"EAR : {ear:.3f}  (seuil {SEUIL_EAR})", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        ferme_depuis = None
        if alarme_active:
            arreter_alarme()
            alarme_active = False
        cv2.putText(frame, "Aucun visage", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Etape 4 - EAR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

detecteur.close()
arreter_alarme()
cap.release()
cv2.destroyAllWindows()
