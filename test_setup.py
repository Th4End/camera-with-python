import os
import sys

def test_imports():
    """Teste l'import des modules nécessaires"""
    print("[TEST 1] Vérification des imports...")
    try:
        import cv2
        print("  opencv-python")
        import mediapipe as mp
        print("  mediapipe")
        import numpy as np
        print("  numpy")
        from mediapipe.tasks import python
        print("  mediapipe.tasks")
        print("  TOUS les imports réussis\n")
        return True
    except ImportError as e:
        print(f" ERREUR: {e}\n")
        return False

def test_models():
    """Teste la disponibilité des modèles"""
    print("[TEST 2] Vérification des modèles MediaPipe...")
    models = [
        "hand_landmarker.task",
        "gesture_recognizer.task",
    ]
    
    all_exist = True
    for model in models:
        if os.path.exists(model):
            print(f"  {model}")
        else:
            print(f"  {model} - MANQUANT")
            all_exist = False
    
    if not all_exist:
        print("  \n  Exécutez: python download_models.py\n")
    else:
        print("  Tous les modèles sont présents\n")
    
    return all_exist

def test_webcam():
    """Teste si la webcam fonctionne"""
    print("[TEST 3] Vérification de la webcam...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Webcam détectée (résolution: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()
                print("  Webcam fonctionnelle\n")
                return True
            else:
                print("  Impossible de lire depuis la webcam\n")
                cap.release()
                return False
        else:
            print("  Webcam non détectée\n")
            return False
    except Exception as e:
        print(f"  ERREUR: {e}\n")
        return False

def test_models_loading():
    """Teste le chargement des modèles MediaPipe"""
    print("[TEST 4] Chargement des modèles MediaPipe...")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        if os.path.exists("hand_landmarker.task"):
            print("  Chargement de hand_landmarker...")
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=1
            )
            
            with HandLandmarker.create_from_options(options) as landmarker:
                print("  hand_landmarker chargé correctement")
        
        if os.path.exists("gesture_recognizer.task"):
            print("  Chargement de gesture_recognizer...")
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=1
            )
            
            with GestureRecognizer.create_from_options(options) as recognizer:
                print("  gesture_recognizer chargé correctement")
        
        print("  TOUS les modèles se chargent correctement\n")
        return True
    except Exception as e:
        print(f"  ERREUR lors du chargement: {e}\n")
        return False

def main():
    print("=" * 60)
    print("TEST D'INSTALLATION - Camera Project")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Modèles disponibles", test_models()))
    results.append(("Webcam", test_webcam()))
    
    if results[1][1]:
        results.append(("Chargement des modèles", test_models_loading()))
    
    print("=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    print(f"\nRésultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\nLE PROJET EST PRÊT À ÊTRE UTILISÉ!")
        print("\nLancez l'une de ces commandes:")
        print("  - python cam.py")
        print("  - python camera.py")
        print("  - python signe.py")
        return 0
    else:
        print("\nCERTAINS TESTS ONT ÉCHOUÉ")
        print("\nVérifiez les instructions d'installation dans README.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
