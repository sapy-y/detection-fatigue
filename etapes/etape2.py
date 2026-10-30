import os
import time
import urllib.request


import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

#telechargement du modèle
MODELE = "face_landmark.task"
URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")

if not os.path.isfile(MODELE):
    print("Téléchargement du modèle...")
    urllib.request.urlretrieve(URL, MODELE)
    print("Téléchargement terminé.")


options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODELE),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

detecteur = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # DSHOW : pilote caméra plus fiable sous Windows
if not cap.isOpened():
    print("Webcam introuvable : essaie VideoCapture(1, cv2.CAP_DSHOW)")
    raise SystemExit

t0 = time.monotonic()
echecs = 0

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
   
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


    temps__ms = int((time.monotonic() - t0) * 1000)
    resultat = detecteur.detect_for_video(image, temps__ms)

    if resultat.face_landmarks:
        points = resultat.face_landmarks[0]
        for p in points:
            x = int(p.x * w)
            y = int(p.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        cv2.putText(frame, f"visage détecté : {len(points)} points", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "aucun visage détecté", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Etape 2 - Détection des points du visage", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detecteur.close()
cap.release()
cv2.destroyAllWindows()        

