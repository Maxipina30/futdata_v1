import pandas as pd
import numpy as np
import os
import joblib

# ========================
# CONFIGURACIÓN PRINCIPAL
# ========================
MATCHWEEK_OBJETIVO = 25   # 👉 Fecha a analizar
EQUIPO_OBJETIVO = "La Serena"  # 👉 Equipo a explicar
WINDOW = 5

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "files", "02_processed")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")

# ========================
# FUNCIONES AUXILIARES
# ========================
def generar_features(df, window=5):
    df = df.sort_values(["equipo", "date"]).copy()
    group = df.groupby("equipo")

    gf_col = "gf_x" if "gf_x" in df.columns else "gf"
    ga_col = "ga_x" if "ga_x" in df.columns else "ga"
    poss_col = "poss_x" if "poss_x" in df.columns else "poss"

    # Rolling 5
    df["GF_rolling5"] = group[gf_col].transform(lambda x: x.shift().rolling(5, 1).mean())
    df["GA_rolling5"] = group[ga_col].transform(lambda x: x.shift().rolling(5, 1).mean())
    df["Poss_rolling5"] = group[poss_col].transform(lambda x: x.shift().rolling(5, 1).mean())
    df["GolesDif_rolling5"] = group.apply(
        lambda g: (g[gf_col] - g[ga_col]).shift().rolling(5, 1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df["WinRate_rolling5"] = group["result"].transform(lambda x: x.shift().eq("W").rolling(5, 1).mean())

    # Rolling 3
    df["GF_rolling3"] = group[gf_col].transform(lambda x: x.shift().rolling(3, 1).mean())
    df["GA_rolling3"] = group[ga_col].transform(lambda x: x.shift().rolling(3, 1).mean())
    df["Poss_rolling3"] = group[poss_col].transform(lambda x: x.shift().rolling(3, 1).mean())
    df["GolesDif_rolling3"] = group.apply(
        lambda g: (g[gf_col] - g[ga_col]).shift().rolling(3, 1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df["WinRate_rolling3"] = group["result"].transform(lambda x: x.shift().eq("W").rolling(3, 1).mean())

    return df


def main():
    print(f"⚽ Analizando explicación del modelo para {EQUIPO_OBJETIVO} (Matchweek {MATCHWEEK_OBJETIVO})...\n")

    # === Cargar datos ===
    path = os.path.join(RAW_DIR, "chile_clean_full.csv")
    if not os.path.exists(path):
        print("❌ No se encontró chile_clean_full.csv en files/02_processed/")
        return

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Local"] = (df["venue"].str.lower() == "home").astype(int)

    df = generar_features(df, window=WINDOW)

    # === Cargar modelo ===
    model_path = os.path.join(MODEL_DIR, "modelo_resultados.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(model_path):
        print("❌ No se encontró el modelo entrenado.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # === Variables del modelo ===
    features = [
        "Local",
        "GF_rolling5", "GA_rolling5", "Poss_rolling5", "GolesDif_rolling5", "WinRate_rolling5",
        "GF_rolling3", "GA_rolling3", "Poss_rolling3", "GolesDif_rolling3", "WinRate_rolling3"
    ]

    fila = df[(df["round_num"] == MATCHWEEK_OBJETIVO) & (df["equipo"] == EQUIPO_OBJETIVO)].copy()
    if fila.empty:
        print(f"⚠️ No se encontró {EQUIPO_OBJETIVO} en la fecha {MATCHWEEK_OBJETIVO}")
        return

    X = fila[features].fillna(0)
    X_scaled = scaler.transform(X)

    # === Logits ===
    logits = model.decision_function(X_scaled)[0]
    clases = model.classes_
    df_logits = pd.DataFrame({"Clase": clases, "Logit": logits})
    print("🧠 Logits del modelo (antes del softmax):")
    print(df_logits.to_string(index=False))

    # === Probabilidades ===
    probs = model.predict_proba(X_scaled)[0]
    df_probs = pd.DataFrame({"Clase": clases, "Probabilidad": probs})
    print("\n🎯 Probabilidades (softmax):")
    print(df_probs.to_string(index=False))

    # === Contribuciones por variable ===
    idx_gana = np.where(clases == 1)[0][0]
    coefs = model.coef_[idx_gana]
    contrib = X_scaled[0] * coefs

    df_contrib = pd.DataFrame({
        "Variable": features,
        "Valor (estandarizado)": X_scaled[0],
        "Coef_Gana": coefs,
        "Contribución": contrib
    }).sort_values("Contribución", ascending=False)

    print(f"\n📊 Contribución de variables para {EQUIPO_OBJETIVO} (Ganar):")
    print(df_contrib.to_string(index=False))

    total_logit = contrib.sum() + model.intercept_[idx_gana]
    print(f"\nΣ contribuciones + intercepto = {total_logit:.4f}")
    print("✅ Explicación generada correctamente 🇨🇱")


if __name__ == "__main__":
    main()
