import os
import sys
import importlib.util

def check_module(module_name, package_name=None):
    """Vérifie si un module Python est installé"""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"{package_name}")
        return True
    else:
        print(f" {package_name} - MANQUANT")
        return False

def check_file(filepath):
    """Vérifie si un fichier existe"""
    if os.path.exists(filepath):
        print(f"  {filepath}")
        return True
    else:
        print(f" {filepath} - MANQUANT")
        return False

def main():
    print("=" * 60)
    print("Vérification de l'environnement du projet Camera")
    print("=" * 60)
    
    all_ok = True
    
    print("\n[1] Vérification des modules Python...")
    modules = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
    ]
    
    for module, package in modules:
        if not check_module(module, package):
            all_ok = False
    
    print("\n[2] Vérification des fichiers modèles MediaPipe...")
    models = [
        "hand_landmarker.task",
        "gesture_recognizer.task",
    ]
    
    for model in models:
        if not check_file(model):
            all_ok = False
    
    print("\n[3] Vérification des fichiers de données...")
    if os.path.exists("data"):
        npy_files = [f for f in os.listdir("data") if f.endswith(".npy")]
        if npy_files:
            print(f"Dossier 'data' avec {len(npy_files)} fichier(s)")
        else:
            print("Dossier 'data' vide - données d'entraînement manquantes")
    else:
        print("Dossier 'data' n'existe pas")
    
    print("\n[4] Vérification du modèle ML...")
    if check_file("lsf_model.pkl"):
        pass
    else:
        print(" lsf_model.pkl - MANQUANT (À générer avec train.py)")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("Tous les éléments essentiels sont en place!")
        print("\nVous pouvez maintenant lancer:")
        print("  - python cam.py           (détection mains + visages)")
        print("  - python camera.py        (détection mains uniquement)")
        print("  - python signe.py         (reconnaissance de gestes)")
        print("  - python predict.py       (prédiction LSF)")
        return 0
    else:
        print("Certains éléments manquent!")
        print("\nPour corriger, lancez:")
        print("  1. pip install -r requirements.txt")
        print("  2. python download_models.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
