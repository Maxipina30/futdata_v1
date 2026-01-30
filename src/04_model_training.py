import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "files", "03_features")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ========================
# SCRIPT PRINCIPAL
# ========================
def main():
    print("⚙️ Entrenando modelo multiclase (Win / Draw / Loss)...\n")

    # === 1. CARGAR DATASET ===
    path = os.path.join(FEATURES_DIR, "dataset_modelo_partidos.csv")
    if not os.path.exists(path):
        print("❌ No se encontró dataset_modelo_partidos.csv en files/03_features/")
        return

    df = pd.read_csv(path)
    print(f"📄 Registros cargados: {len(df)}")

    # === 2. LIMPIEZA ===
    # Eliminar filas con Target nulo o sin datos rolling
    df = df.dropna(subset=["Target"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="any")  # ✅ elimina filas incompletas de rolling

    print(f"✅ Registros tras limpieza: {len(df)}")

    # === 3. SELECCIÓN DE VARIABLES ===
    # Incluye todas las estadísticas rolling de local y visitante (3 y 5 partidos)
    features = [
        # Rolling global (local)
        "_gf_rolling5_local", "_ga_rolling5_local", "_poss_rolling5_local", "_winrate_rolling5_local",
        "_gf_rolling3_local", "_ga_rolling3_local", "_poss_rolling3_local", "_winrate_rolling3_local",

        # Rolling global (visitante)
        "_gf_rolling5_away", "_ga_rolling5_away", "_poss_rolling5_away", "_winrate_rolling5_away",
        "_gf_rolling3_away", "_ga_rolling3_away", "_poss_rolling3_away", "_winrate_rolling3_away",

        # Rolling contextual local
        "home_gf_rolling5_local", "home_ga_rolling5_local", "home_poss_rolling5_local", "home_winrate_rolling5_local",
        "home_gf_rolling3_local", "home_ga_rolling3_local", "home_poss_rolling3_local", "home_winrate_rolling3_local",

        # Rolling contextual visitante
        "away_gf_rolling5_away", "away_ga_rolling5_away", "away_poss_rolling5_away", "away_winrate_rolling5_away",
        "away_gf_rolling3_away", "away_ga_rolling3_away", "away_poss_rolling3_away", "away_winrate_rolling3_away"
    ]

    # Asegurar que todas las columnas existan
    features = [f for f in features if f in df.columns]
    print(f"🧩 Variables consideradas: {len(features)}\n")

    X = df[features]
    y = df["Target"]

    # === 4. SPLIT TRAIN / TEST ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === 5. ESCALADO ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === 6. MODELO ===
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # === 7. PREDICCIONES ===
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # === 8. REPORTES ===
    print("📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=3))
    print("\n🧩 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # === 9. GUARDAR MODELO Y SCALER ===
    model_path = os.path.join(MODEL_DIR, "modelo_partidos.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler_partidos.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Modelo guardado en: {model_path}")
    print(f"💾 Scaler guardado en: {scaler_path}")

    # === 10. GUARDAR PREDICCIONES ===
    proba_cols = [f"P_{c}" for c in model.classes_]  # P_-1, P_0, P_1
    df_pred = pd.DataFrame(y_proba, columns=proba_cols)
    df_pred["Prediccion_Final"] = y_pred
    df_pred["Real"] = y_test.to_numpy()

    pred_path = os.path.join(REPORT_DIR, "predicciones_partidos.csv")
    df_pred.to_csv(pred_path, index=False)

    print(f"💾 Archivo con probabilidades guardado en: {pred_path}")
    print("\n✅ Entrenamiento finalizado correctamente.")


# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    main()
