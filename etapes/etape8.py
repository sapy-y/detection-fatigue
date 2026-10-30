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
from collections import deque

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
    for _ in range(4):                       
        rafale += [bip(1800, 0.08), silence(0.04),
                   bip(2300, 0.08), silence(0.06)]
    son = np.concatenate(rafale + [silence(0.25)])   

    son = np.tanh(3 * son / np.abs(son).max())       
    son = (son / np.abs(son).max() * 0.98 * 32767).astype(np.int16)  

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
RATIO = 0.7              
DUREE_CALIBRATION = 3.0   
DUREE_ALERTE = 2.0      

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
seuil = None             
calib_debut = time.monotonic()
calib_valeurs = []
alarme_active = False
nb_clignements = 0
clignements = deque()            # heures des clignements (pour calculer le nombre par minute)
nb_alertes = 0
historique = deque(maxlen=150)   # les 150 dernières valeurs d'EAR (environ 5 s)

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
        historique.append(ear)

        if seuil is None:                                   # --- calibration ---
            calib_valeurs.append(ear)
            reste = DUREE_CALIBRATION - (maintenant - calib_debut)
            etat, couleur = f"CALIBRATION : garde les yeux ouverts {max(reste, 0):.1f} s", (0, 165, 255)
            if reste <= 0:
                normal = float(np.median(calib_valeurs))    # médiane : ignore les clignements
                seuil = normal * RATIO
                print(f"EAR normal = {normal:.3f}  ->  seuil = {seuil:.3f}")

        elif ear < seuil:                                   # yeux fermés (avant : SEUIL_EAR)
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
                if not alarme_active:         
                    demarrer_alarme()
                    alarme_active = True
                    nb_alertes += 1 
        else:
            if ferme_depuis is not None:          
                duree_fermeture = maintenant - ferme_depuis
                if 0.05 < duree_fermeture < 0.4: 
                    nb_clignements += 1
                    clignements.append(maintenant)                                     
            ferme_depuis = None
            if alarme_active:
                arreter_alarme()
                alarme_active = False                   
            etat, couleur = "Yeux ouverts", (0, 255, 0)

        cv2.putText(frame, etat, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, couleur, 2)
        texte_seuil = f"{seuil:.3f}" if seuil else "--"
        cv2.putText(frame, f"EAR : {ear:.3f}  (seuil {texte_seuil})", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        ferme_depuis = None
        if alarme_active:
            arreter_alarme()
            alarme_active = False
        cv2.putText(frame, "Aucun visage", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
   
    while clignements and time.monotonic() - clignements[0] > 60:
        clignements.popleft()                    
    cv2.putText(frame, f"Clignements : {nb_clignements}  ({len(clignements)}/min)",
                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Alertes : {nb_alertes}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    if len(historique) > 1:
        gx, gy, gw, gh = w - 260, h - 110, 240, 90          
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (30, 30, 30), -1)

        def vers_y(v):                                       
            return int(gy + gh - min(v, 0.45) / 0.45 * gh)

        pts_courbe = [(gx + int(i * gw / (historique.maxlen - 1)), vers_y(v))
                      for i, v in enumerate(historique)]
        cv2.polylines(frame, [np.array(pts_courbe, np.int32)], False, (0, 255, 0), 2)
        if seuil:
            cv2.line(frame, (gx, vers_y(seuil)), (gx + gw, vers_y(seuil)), (0, 0, 255), 1)

    cv2.imshow("Etape 4 - EAR", frame)
    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("c"):                 
        seuil = None
        calib_debut = time.monotonic()
        calib_valeurs = []
        ferme_depuis = None
        if alarme_active:
            arreter_alarme()
            alarme_active = False

detecteur.close()
arreter_alarme()
cap.release()
cv2.destroyAllWindows()
print(f"Session terminée : {nb_clignements} clignements, {nb_alertes} alerte(s).")