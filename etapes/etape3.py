import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODELE = "face_landmarker.task"

# 🆕 Numéros des 6 points de chaque œil, dans l'ordre p1 → p6
OEIL_GAUCHE = [362, 385, 387, 263, 373, 380]
OEIL_DROIT = [33, 160, 158, 133, 153, 144]

options = vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODELE),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
detecteur = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    temps_ms = int((time.monotonic() - t0) * 1000)
    resultat = detecteur.detect_for_video(image, temps_ms)

    if resultat.face_landmarks:
        points = resultat.face_landmarks[0]

        #  Dessiner seulement les yeux
        for oeil in (OEIL_GAUCHE, OEIL_DROIT):
            coords = []
            for numero, indice in enumerate(oeil, start=1):
                p = points[indice]
                x, y = int(p.x * w), int(p.y * h)
                coords.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"p{numero}", (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            # relier les 6 points pour former le contour de l'œil
            for i in range(6):
                cv2.line(frame, coords[i], coords[(i + 1) % 6], (0, 255, 255), 1)
    else:
        cv2.putText(frame, "Aucun visage", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Etape 3 - Yeux", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

detecteur.close()
cap.release()
cv2.destroyAllWindows()