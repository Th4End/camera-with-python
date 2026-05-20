"""
Configuration centralisée du projet Camera
Contient les constantes utilisées par tous les scripts
"""

import os

# ===== PATHS =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug_data")
MODELS_DIR = PROJECT_ROOT

# Fichiers modèles
HAND_LANDMARKER_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
GESTURE_RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "gesture_recognizer.task")
ML_MODEL_FILE = os.path.join(PROJECT_ROOT, "lsf_model.pkl")

# ===== VIDEO SETTINGS =====
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30

# ===== MEDIAPIPE SETTINGS =====
# Hand detection
HAND_MIN_DETECTION_CONFIDENCE = 0.7
HAND_MIN_TRACKING_CONFIDENCE = 0.4
MAX_NUM_HANDS = 4

# Face detection
FACE_MIN_DETECTION_CONFIDENCE = 0.7

# ===== ML MODEL SETTINGS =====
# RandomForestClassifier parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_RANDOM_STATE = 42
RF_TEST_SIZE = 0.2

# ===== LABELS =====
# Default label for data collection in hand_detection.py
DEFAULT_LABEL = "bonjour"

# ===== KEYBOARD CONTROLS =====
KEY_QUIT = 'a'
KEY_CAPTURE = 's'

# ===== MODEL URLS =====
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer.task",
}
