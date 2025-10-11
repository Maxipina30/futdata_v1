import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump

# === 1️⃣ Cargar dataset ===
print("⚙️ Cargando dataset de modelado...\n")

path = r"C:\Users\maxip\Documents\futdata_v1\files\03_features\dataset_modelo.csv"
if not os.path.exists(path):
    raise FileNotFoundError(f"No se encontró el archivo: {path}")

df = pd.read_csv(path)

# === 2️⃣ Limpieza y selección de variables ===
df = df.dropna(subset=["Target"])
features = ["Δ_GF_rolling", "Δ_GA_rolling", "Δ_Goles_Dif_rolling", "Δ_Poss_rolling", "Δ_WinRate_rolling"]
df = df.dropna(subset=features)

X = df[features]
y = df["Target"]

# === 3️⃣ Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# === 4️⃣ Pipeline base (imputación + escalado + modelo) ===
base_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, solver="saga", multi_class="auto"))
])

# === 5️⃣ Grid Search para modelo base ===
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"],
    "model__class_weight": [None, "balanced"]
}

print("🔍 Ejecutando búsqueda de hiperparámetros (Grid Search)...\n")
grid = GridSearchCV(base_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("🏆 Mejor combinación (modelo base):")
print(grid.best_params_, "\n")

# === 6️⃣ Evaluar modelo base ===
best_model = grid.best_estimator_
y_pred_base = best_model.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)

print("📊 Reporte modelo base:")
print(classification_report(y_test, y_pred_base))

# === 7️⃣ Entrenar modelo balanceado manualmente ===
balanced_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=grid.best_params_["model__C"],
        penalty=grid.best_params_["model__penalty"],
        solver="saga",
        class_weight="balanced",
        max_iter=1000
    ))
])

balanced_model.fit(X_train, y_train)
y_pred_bal = balanced_model.predict(X_test)
acc_bal = accuracy_score(y_test, y_pred_bal)

print("📊 Reporte modelo balanceado:")
print(classification_report(y_test, y_pred_bal))

# === 8️⃣ Comparación ===
print("📈 Comparación de accuracy:")
print(f"Modelo Base       : {acc_base:.3f}")
print(f"Modelo Balanceado : {acc_bal:.3f}\n")

# === 9️⃣ Matrices de confusión ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_base = confusion_matrix(y_test, y_pred_base)
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Pierde (-1)", "Empata (0)", "Gana (1)"],
            yticklabels=["Pierde (-1)", "Empata (0)", "Gana (1)"])
axes[0].set_title("Modelo Base")

cm_bal = confusion_matrix(y_test, y_pred_bal)
sns.heatmap(cm_bal, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["Pierde (-1)", "Empata (0)", "Gana (1)"],
            yticklabels=["Pierde (-1)", "Empata (0)", "Gana (1)"])
axes[1].set_title("Modelo Balanceado")

plt.suptitle("Comparación de Matrices de Confusión", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# === 🔟 Guardar modelos ===
os.makedirs(r"C:\Users\maxip\Documents\futdata_v1\files\06_models", exist_ok=True)
dump(best_model, r"C:\Users\maxip\Documents\futdata_v1\files\06_models\logistic_gridsearch.joblib")
dump(balanced_model, r"C:\Users\maxip\Documents\futdata_v1\files\06_models\logistic_balanced.joblib")

print("💾 Modelos guardados en files/06_models/")
