import pandas as pd
import numpy as np
import os
import joblib

# ========================
# CONFIGURACIÓN PRINCIPAL
# ========================
MATCHWEEK_OBJETIVO = 24  # 👉 cambia aquí la fecha a predecir
WINDOW = 5

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "files", "02_processed")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ========================
# FUNCIONES AUXILIARES
# ========================
def generar_features(df, window=5):
    df = df.sort_values(["equipo", "date"]).copy()
    group = df.groupby("equipo")
    df["GF_rolling5"] = group["gf_x"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    df["GA_rolling5"] = group["ga_x"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    df["Poss_rolling5"] = group["poss"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    df["GolesDif_rolling5"] = group.apply(
        lambda g: (g["gf_x"] - g["ga_x"]).shift().rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df["WinRate_rolling5"] = group["result"].transform(lambda x: x.shift().eq("W").rolling(window, min_periods=1).mean())
    return df


def normalizar_partido(equipo, rival):
    return tuple(sorted([equipo.strip().lower(), rival.strip().lower()]))


# ========================
# SCRIPT PRINCIPAL
# ========================
def main():
    print(f"📅 Generando predicciones para la Matchweek {MATCHWEEK_OBJETIVO}...\n")

    # Cargar datos
    path_clean = os.path.join(RAW_DIR, "chile_clean.csv")
    if not os.path.exists(path_clean):
        print("❌ No se encontró chile_clean.csv en files/02_clean/")
        return

    df = pd.read_csv(path_clean)
    print(f"✅ Datos cargados: {df.shape[0]} registros totales")

    # Tipos correctos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["gf_x", "ga_x", "poss", "pts", "gd", "pts_stand_opp", "gd_stand_opp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Generar rolling features
    df = generar_features(df, window=WINDOW)

    # Filtrar Matchweek
    df_target = df[df["round_num"] == MATCHWEEK_OBJETIVO].copy()
    if df_target.empty:
        print(f"⚠️ No se encontraron partidos para la Matchweek {MATCHWEEK_OBJETIVO}.")
        return

    print(f"🎯 Fecha seleccionada: Matchweek {MATCHWEEK_OBJETIVO} ({len(df_target)} partidos)")

    # Eliminar duplicados espejo
    df_target["Partido_ID"] = df_target.apply(lambda x: normalizar_partido(x["equipo"], x["opponent"]), axis=1)
    df_target = df_target.drop_duplicates("Partido_ID")
    print(f"✅ Después de eliminar duplicados: {len(df_target)} partidos\n")

    # Cargar modelo y scaler
    model_path = os.path.join(MODEL_DIR, "modelo_resultados.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ No se encontró el modelo o el scaler. Ejecuta primero 04_model_training.py")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # ========================
    # Predicción base
    # ========================
    features = ["Local", "GF_rolling5", "GA_rolling5", "Poss_rolling5", "GolesDif_rolling5", "WinRate_rolling5"]
    df_target["Local"] = (df_target["venue"].str.lower() == "home").astype(int)
    X = df_target[features].fillna(0)
    X_scaled = scaler.transform(X)

    y_proba = model.predict_proba(X_scaled)
    y_pred = model.predict(X_scaled)

    proba_cols = [f"P_{c}" for c in model.classes_]
    df_pred = pd.DataFrame(y_proba, columns=proba_cols)
    df_pred["Prediccion"] = y_pred
    df_pred["Equipo"] = df_target["equipo"].values
    df_pred["Opponent"] = df_target["opponent"].values
    df_pred["Venue"] = df_target["venue"].values

    # ========================
    # DEBUG
    # ========================
    print("🔍 --- DEBUG POST-MODELO ---")
    print("Columnas disponibles en df_target:")
    print(list(df_target.columns))
    print("\nColumnas disponibles en df_pred:")
    print(list(df_pred.columns))
    print("\nPrimeras filas relevantes para ajuste:")
    cols_debug = ["equipo", "opponent", "pts", "pts_stand_opp", "gd", "gd_stand_opp"]
    cols_debug = [c for c in cols_debug if c in df_target.columns]
    print(df_target[cols_debug].head(10))
    print("\nPrimeras filas de probabilidades base:")
    print(df_pred[["Equipo", "Opponent", "P_1", "P_0", "P_-1"]].head(10))
    print("🔍 --- FIN DEBUG ---\n")

    # ========================
    # Ajuste post-modelo
    # ========================
    print("⚙️ Aplicando ajuste post-modelo según standings...")

    # Alinear nombres y mergear standings
    df_pred["equipo"] = df_pred["Equipo"]
    df_pred["opponent"] = df_pred["Opponent"]

    cols_extra = ["equipo", "pts", "gd", "pts_stand_opp", "gd_stand_opp"]
    df_pred = df_pred.merge(df_target[cols_extra], on="equipo", how="left")

    # Calcular diferencias relativas
    df_pred["diff_pts"] = df_pred["pts"] - df_pred["pts_stand_opp"]
    df_pred["diff_gd"] = df_pred["gd"] - df_pred["gd_stand_opp"]

    # Escalar y combinar factores
    df_pred["adj_factor"] = (
        0.1 * np.tanh(df_pred["diff_pts"] / 10) +
        0.1 * np.tanh(df_pred["diff_gd"] / 10)
    )

    # Aplicar ajuste
    df_pred["P_1_adj"] = np.clip(df_pred["P_1"] * (1 + df_pred["adj_factor"]), 0, 1)
    df_pred["P_-1_adj"] = np.clip(df_pred["P_-1"] * (1 - df_pred["adj_factor"]), 0, 1)

    # Renormalizar
    total = df_pred["P_1_adj"] + df_pred["P_0"] + df_pred["P_-1_adj"]
    df_pred["P_1_adj"] /= total
    df_pred["P_0_adj"] = df_pred["P_0"] / total
    df_pred["P_-1_adj"] /= total

    # Predicción final
    df_pred["Prediccion_final"] = df_pred[["P_1_adj", "P_0_adj", "P_-1_adj"]].idxmax(axis=1)
    df_pred["Prediccion_final"] = df_pred["Prediccion_final"].map(
        {"P_1_adj": "Gana", "P_0_adj": "Empata", "P_-1_adj": "Pierde"}
    )

    # ========================
    # Output final
    # ========================
    print("\n📊 Predicciones generadas (Local vs Visita):\n")
    for _, row in df_pred.iterrows():
        local = row["Equipo"] if row["Venue"].lower() == "home" else row["Opponent"]
        visita = row["Opponent"] if row["Venue"].lower() == "home" else row["Equipo"]

        # Ajuste: si el equipo es visitante, invertimos las probabilidades
        if row["Venue"].lower() == "home":
            p_local = row["P_1_adj"]
            p_visita = row["P_-1_adj"]
        else:
            p_local = row["P_-1_adj"]
            p_visita = row["P_1_adj"]

        print(f"{local} vs {visita}")
        print(f"   → Prob {local}: {p_local:.2f} | Empate: {row['P_0_adj']:.2f} | {visita}: {p_visita:.2f}")

        # Determinar ganador según probabilidad máxima
        if p_local > p_visita and p_local > row["P_0_adj"]:
            resultado = f"Gana {local}"
        elif p_visita > p_local and p_visita > row["P_0_adj"]:
            resultado = f"Gana {visita}"
        else:
            resultado = "Empate"

        print(f"   → Resultado esperado: {resultado}\n")

    # Guardar
    out_path = os.path.join(REPORT_DIR, f"predicciones_matchweek{MATCHWEEK_OBJETIVO}_ajustadas.csv")
    df_pred.to_csv(out_path, index=False)
    print(f"💾 Guardado en: {out_path}")
    print("✅ Proceso finalizado correctamente 🇨🇱")


if __name__ == "__main__":
    main()
