import pandas as pd
import numpy as np
import os
import joblib

# ========================
# CONFIGURACIÓN PRINCIPAL
# ========================
MATCHWEEK_OBJETIVO = 25  # 👉 cambia aquí la fecha a predecir
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

    gf_col = "gf_x" if "gf_x" in df.columns else "gf"
    ga_col = "ga_x" if "ga_x" in df.columns else "ga"
    poss_col = "poss_x" if "poss_x" in df.columns else "poss"

    # Rolling 5
    df["GF_rolling5"] = group[gf_col].transform(lambda x: x.shift().rolling(5, 1).mean())
    df["GA_rolling5"] = group[ga_col].transform(lambda x: x.shift().rolling(5, 1).mean())
    df["Poss_rolling5"] = group[poss_col].transform(lambda x: x.shift().rolling(5, 1).mean())
    df["GolesDif_rolling5"] = group.apply(
        lambda g: (g[gf_col] - g[ga_col]).shift().rolling(5, 1).mean(),
        include_groups=False  # 👈 evita el FutureWarning
    ).reset_index(level=0, drop=True)
    df["WinRate_rolling5"] = group["result"].transform(lambda x: x.shift().eq("W").rolling(5, 1).mean())

    # Rolling 3
    df["GF_rolling3"] = group[gf_col].transform(lambda x: x.shift().rolling(3, 1).mean())
    df["GA_rolling3"] = group[ga_col].transform(lambda x: x.shift().rolling(3, 1).mean())
    df["Poss_rolling3"] = group[poss_col].transform(lambda x: x.shift().rolling(3, 1).mean())
    df["GolesDif_rolling3"] = group.apply(
        lambda g: (g[gf_col] - g[ga_col]).shift().rolling(3, 1).mean(),
        include_groups=False  # 👈 evita el FutureWarning
    ).reset_index(level=0, drop=True)
    df["WinRate_rolling3"] = group["result"].transform(lambda x: x.shift().eq("W").rolling(3, 1).mean())

    return df



def normalizar_partido(equipo, rival):
    return tuple(sorted([equipo.strip().lower(), rival.strip().lower()]))

# ========================
# SCRIPT PRINCIPAL
# ========================
def main():
    print(f"📅 Generando predicciones para la Matchweek {MATCHWEEK_OBJETIVO}...\n")

    # Dataset procesado extendido
    path_clean = os.path.join(RAW_DIR, "chile_clean_full.csv")
    if not os.path.exists(path_clean):
        print("❌ No se encontró chile_clean_full.csv en files/02_processed/")
        return

    df = pd.read_csv(path_clean)
    print(f"✅ Datos cargados: {df.shape[0]} registros totales")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["gf", "ga", "poss", "pts_stand", "gd_stand",
                "pts_stand_stand_opp", "gd_stand_stand_opp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = generar_features(df, window=WINDOW)

    df_target = df[df["round_num"] == MATCHWEEK_OBJETIVO].copy()
    if df_target.empty:
        print(f"⚠️ No se encontraron partidos para la Matchweek {MATCHWEEK_OBJETIVO}.")
        return

    print(f"🎯 Fecha seleccionada: Matchweek {MATCHWEEK_OBJETIVO} ({len(df_target)} partidos)")
    df_target["Partido_ID"] = df_target.apply(lambda x: normalizar_partido(x["equipo"], x["opponent"]), axis=1)
    df_target = df_target.drop_duplicates("Partido_ID")
    print(f"✅ Después de eliminar duplicados: {len(df_target)} partidos\n")

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
    features = [
        "Local",
        "GF_rolling5", "GA_rolling5", "Poss_rolling5", "GolesDif_rolling5", "WinRate_rolling5",
        "GF_rolling3", "GA_rolling3", "Poss_rolling3", "GolesDif_rolling3", "WinRate_rolling3"
    ]
    df_target["Local"] = (df_target["venue"].str.lower() == "home").astype(int)
    X = df_target[features].fillna(0)
    X_scaled = scaler.transform(X)

    y_proba = model.predict_proba(X_scaled)
    y_pred = model.predict(X_scaled)

    proba_cols = [f"P_{c}" for c in model.classes_]
    df_pred = pd.DataFrame(y_proba, columns=proba_cols)

    # 🔧 Normalizar nombres por si las clases son floats (ej: -1.0, 0.0, 1.0)
    df_pred.columns = df_pred.columns.str.replace(".0", "", regex=False)

    df_pred["Prediccion"] = y_pred
    df_pred["Equipo"] = df_target["equipo"].values
    df_pred["Opponent"] = df_target["opponent"].values
    df_pred["Venue"] = df_target["venue"].values

    # ========================
    # AJUSTE POST-MODELO
    # ========================
    print("⚙️ Aplicando ajuste post-modelo según standings y rendimiento reciente...")

    df_pred["equipo"] = df_pred["Equipo"]
    df_pred["opponent"] = df_pred["Opponent"]

    pts_col = "pts_stand" if "pts_stand" in df_target.columns else None
    gd_col = "gd_stand" if "gd_stand" in df_target.columns else None
    pts_opp_col = "pts_stand_stand_opp" if "pts_stand_stand_opp" in df_target.columns else None
    gd_opp_col = "gd_stand_stand_opp" if "gd_stand_stand_opp" in df_target.columns else None

    cols_extra = ["equipo"]
    for col in [pts_col, gd_col, pts_opp_col, gd_opp_col,
                "performance_g+a_for", "performance_g+a_agn",
                "performance_g+a_for_for_opp", "performance_g+a_agn_agn_opp",
                "HomePts_avg", "AwayPts_avg", "Poss_rolling5"]:
        if col and col in df_target.columns:
            cols_extra.append(col)

    df_pred = df_pred.merge(df_target[cols_extra], on="equipo", how="left")

    rename_map = {}
    if pts_col: rename_map[pts_col] = "pts"
    if gd_col: rename_map[gd_col] = "gd"
    if pts_opp_col: rename_map[pts_opp_col] = "pts_opp"
    if gd_opp_col: rename_map[gd_opp_col] = "gd_opp"
    df_pred.rename(columns=rename_map, inplace=True)

    # --- Diferencias clave ---
    df_pred["pts"] = df_pred.get("pts", 0)
    df_pred["gd"] = df_pred.get("gd", 0)
    df_pred["pts_opp"] = df_pred.get("pts_opp", 0)
    df_pred["gd_opp"] = df_pred.get("gd_opp", 0)
    df_pred["diff_pts"] = df_pred["pts"] - df_pred["pts_opp"]
    df_pred["diff_gd"]  = df_pred["gd"]  - df_pred["gd_opp"]

    # Producción ofensiva y defensiva (G+A reales)
    df_pred["diff_g+a_for"] = (
        df_pred.get("performance_g+a_for", 0) -
        df_pred.get("performance_g+a_agn_agn_opp", 0)
    )
    df_pred["diff_g+a_agn"] = (
        df_pred.get("performance_g+a_agn", 0) -
        df_pred.get("performance_g+a_for_for_opp", 0)
    )

    # Localía efectiva
    if "HomePts_avg" in df_pred.columns and "AwayPts_avg" in df_pred.columns:
        df_pred["diff_homeaway"] = np.where(
            df_pred["Venue"].str.lower() == "home",
            df_pred["HomePts_avg"] - df_pred["AwayPts_avg"],
            df_pred["AwayPts_avg"] - df_pred["HomePts_avg"]
        )
    else:
        df_pred["diff_homeaway"] = 0

    # --- Ajuste ponderado reescalado ---
    df_pred["adj_factor"] = (
          0.14 * np.tanh(df_pred["diff_pts"] / 10)
        + 0.12 * np.tanh(df_pred["diff_gd"]  / 10)
        + 0.13 * np.tanh(df_pred["diff_g+a_for"] / 5)
        - 0.18 * np.tanh(df_pred["diff_g+a_agn"] / 5)
        + 0.13 * np.tanh(df_pred["diff_homeaway"] / 3)
    )

    # Aplicar ajuste y renormalizar
    df_pred["P_1_adj"] = np.clip(df_pred["P_1"] * (1 + df_pred["adj_factor"]), 0, 1)
    df_pred["P_-1_adj"] = np.clip(df_pred["P_-1"] * (1 - df_pred["adj_factor"]), 0, 1)
    total = df_pred["P_1_adj"] + df_pred["P_0"] + df_pred["P_-1_adj"]
    df_pred["P_1_adj"] /= total
    df_pred["P_0_adj"] = df_pred["P_0"] / total
    df_pred["P_-1_adj"] /= total

    df_pred["Prediccion_final"] = df_pred[["P_1_adj", "P_0_adj", "P_-1_adj"]].idxmax(axis=1)
    df_pred["Prediccion_final"] = df_pred["Prediccion_final"].map(
        {"P_1_adj": "Gana", "P_0_adj": "Empata", "P_-1_adj": "Pierde"}
    )

    # ========================
    # OUTPUT FINAL
    # ========================
    print("\n📊 Predicciones generadas (Local vs Visita):\n")
    for _, row in df_pred.iterrows():
        local = row["Equipo"] if row["Venue"].lower() == "home" else row["Opponent"]
        visita = row["Opponent"] if row["Venue"].lower() == "home" else row["Equipo"]

        if row["Venue"].lower() == "home":
            p_local = row["P_1_adj"]
            p_visita = row["P_-1_adj"]
        else:
            p_local = row["P_-1_adj"]
            p_visita = row["P_1_adj"]

        print(f"{local} vs {visita}")
        print(f"   → Prob {local}: {p_local:.2f} | Empate: {row['P_0_adj']:.2f} | {visita}: {p_visita:.2f}")

        if p_local > p_visita and p_local > row["P_0_adj"]:
            resultado = f"Gana {local}"
        elif p_visita > p_local and p_visita > row["P_0_adj"]:
            resultado = f"Gana {visita}"
        else:
            resultado = "Empate"

        print(f"   → Resultado esperado: {resultado}\n")

    out_path = os.path.join(REPORT_DIR, f"predicciones_matchweek{MATCHWEEK_OBJETIVO}_ajustadas.csv")
    df_pred.to_csv(out_path, index=False)
    print(f"💾 Guardado en: {out_path}")
    print("✅ Proceso finalizado correctamente 🇨🇱")


if __name__ == "__main__":
    main()
