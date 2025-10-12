import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "files", "03_features")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("⚙️ Entrenando modelo multiclase (Win/Draw/Loss)...\n")

    path = os.path.join(FEATURES_DIR, "dataset_modelo.csv")
    if not os.path.exists(path):
        print("❌ No se encontró dataset_modelo.csv en files/03_features/")
        return

    df = pd.read_csv(path)

    # === Selección de variables ===
    features = [
        "Local", "GF_rolling5", "GA_rolling5",
        "Poss_rolling5", "GolesDif_rolling5", "WinRate_rolling5"
    ]
    X = df[features]
    y = df["Target"]

    # === Split Train/Test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Escalado ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Modelo: Regresión Logística Multiclase ===
    model = LogisticRegression(multi_class="multinomial", max_iter=500)
    model.fit(X_train_scaled, y_train)

    # === Predicciones ===
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # === Reportes ===
    print("📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    print("🧩 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # === Guardar modelo y scaler ===
    model_path = os.path.join(MODEL_DIR, "modelo_resultados.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Modelo guardado en: {model_path}")
    print(f"💾 Scaler guardado en: {scaler_path}")

        # === Guardar predicciones con probabilidades ===
    proba_cols = [f"P_{c}" for c in model.classes_]  # P_-1, P_0, P_1
    df_pred = pd.DataFrame(y_proba, columns=proba_cols)
    df_pred["Prediccion_Final"] = y_pred
    df_pred["Real"] = y_test
    df_pred.to_csv(os.path.join(REPORT_DIR, "predicciones_test.csv"), index=False)

    print("\n💾 Archivo con probabilidades guardado en files/05_reports/predicciones_test.csv")

if __name__ == "__main__":
    main()
