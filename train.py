import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

X = []
y = []

for file in os.listdir("data"):
    if file.endswith(".npy"):
        X.append(np.load(f"data/{file}"))
        y.append(file.split("_")[0])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%")

joblib.dump(model, "lsf_model.pkl")
print("Modèle sauvegardé : lsf_model.pkl")
