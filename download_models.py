import urllib.request
import os

MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer.task",
}

print("Téléchargement des modèles MediaPipe...")

for filename, url in MODELS.items():
    filepath = filename
    
    if os.path.exists(filepath):
        print(f"{filename} existe déjà")
    else:
        try:
            print(f"Téléchargement {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"{filename} téléchargé avec succès")
        except Exception as e:
            print(f"Erreur lors du téléchargement de {filename}: {e}")

print("Téléchargement terminé!")
