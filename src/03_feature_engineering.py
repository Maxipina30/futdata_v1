import pandas as pd
import numpy as np
import os

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "files", "02_processed")
OUT_DIR = os.path.join(BASE_DIR, "files", "03_features")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_ROUND_MIN = 1
TRAIN_ROUND_MAX = 28
TEST_ROUND_MIN = 29
TEST_ROUND_MAX = 33
PREDICT_ROUND_MIN = 34


# ========================
# FUNCIONES AUXILIARES
# ========================

def rolling_stats(df, group_cols, prefix, window):
    """
    Calcula rolling means (GF, GA, Poss, WinRate) agrupado por equipo o por tipo de venue.
    Usa min_periods=window para asegurar que solo se consideren promedios completos.
    """
    df = df.sort_values(["equipo", "date"]).copy()
    group = df.groupby(group_cols, group_keys=False)

    df[f"{prefix}_gf_rolling{window}"] = group["gf"].transform(
        lambda x: x.shift().rolling(window=window, min_periods=window).mean()
    )
    df[f"{prefix}_ga_rolling{window}"] = group["ga"].transform(
        lambda x: x.shift().rolling(window=window, min_periods=window).mean()
    )
    df[f"{prefix}_poss_rolling{window}"] = group["poss"].transform(
        lambda x: x.shift().rolling(window=window, min_periods=window).mean()
    )
    df[f"{prefix}_winrate_rolling{window}"] = group["result"].transform(
        lambda x: x.shift().eq("W").rolling(window=window, min_periods=window).mean()
    )

    return df


def build_match_level_dataset(df):
    """
    Genera dataset a nivel de partido centrado en el LOCAL,
    combinando las features rolling de local y visitante.
    """
    df_local = df[df["venue"].str.lower() == "home"].copy()
    df_away = df[df["venue"].str.lower() == "away"].copy()

    # Solo renombrar columnas de rolling
    rolling_cols = [c for c in df.columns if "rolling" in c]
    rename_local = {c: f"{c}_local" for c in rolling_cols}
    rename_away = {c: f"{c}_away" for c in rolling_cols}

    df_local = df_local.rename(columns=rename_local)
    df_away = df_away.rename(columns=rename_away)

    # Merge local vs visitante por fecha y rival
    merged = df_local.merge(
        df_away,
        left_on=["date", "opponent"],
        right_on=["date", "equipo"],
        how="inner",
        suffixes=("_local", "_away")
    )

    # Target: resultado desde perspectiva local
    mapping = {"W": 1, "D": 0, "L": -1}
    merged["Target"] = merged["result_local"].map(mapping)

    # Crear features diferenciales local - visitante para evitar pares duplicados.
    rolling_cols = [c for c in df.columns if "rolling" in c]
    diff_cols = []
    for col in rolling_cols:
        local_col = f"{col}_local"
        away_col = f"{col}_away"
        if local_col in merged.columns and away_col in merged.columns:
            diff_col = f"diff_{col.lstrip('_')}"
            merged[diff_col] = merged[local_col] - merged[away_col]
            diff_cols.append(diff_col)

    # Seleccionar columnas relevantes
    feature_cols = [c for c in merged.columns if "rolling" in c and not c.startswith("diff_")]
    cols_keep = ["date", "round_num_local", "equipo_local", "opponent_local", "Target"] + feature_cols + diff_cols
    cols_keep = [c for c in cols_keep if c in merged.columns]

    merged = merged[cols_keep].rename(columns={
        "equipo_local": "local_team",
        "opponent_local": "away_team",
        "round_num_local": "round_num"
    })

    return merged


# ========================
# SCRIPT PRINCIPAL
# ========================
def main():
    print("⚙️ Generando dataset de modelado (rolling global + contextual, 3 y 5 partidos)...\n")

    path = os.path.join(PROCESSED_DIR, "chile_clean_full.csv")
    if not os.path.exists(path):
        print("❌ No se encontró chile_clean_full.csv en files/02_processed/")
        return

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"📄 Registros cargados: {df.shape[0]}")

    # Normalizar columnas
    df = df.rename(columns=str.lower)
    if "venue" not in df.columns and "home_away" in df.columns:
        df = df.rename(columns={"home_away": "venue"})

    # Asegurar columnas mínimas necesarias
    required = {"equipo", "opponent", "date", "gf", "ga", "poss", "venue", "result"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ Faltan columnas esenciales: {missing}")
        return

    # ========================
    # CALCULAR ROLLING FEATURES
    # ========================
    print("📊 Calculando rolling global (últimos 5 y 3 partidos)...")
    df = rolling_stats(df, ["equipo"], "", 5)
    df = rolling_stats(df, ["equipo"], "", 3)

    print("🏟️ Calculando rolling contextual (por tipo de venue, últimos 5 y 3)...")
    df = rolling_stats(df, ["equipo", "venue"], "home", 5)
    df = rolling_stats(df, ["equipo", "venue"], "home", 3)
    df = rolling_stats(df, ["equipo", "venue"], "away", 5)
    df = rolling_stats(df, ["equipo", "venue"], "away", 3)

    # ========================
    # FILTRAR SOLO SI FALTAN ROLLING GLOBALES (CORREGIDO)
    # ========================
    print("🧹 Filtrando partidos sin suficientes datos globales (mínimo 3-5 previos)...")

    # 🔧 CORRECCIÓN: los nombres correctos NO llevan '_' al inicio
    global_features = [
        "_gf_rolling5", "_ga_rolling5", "_poss_rolling5", "_winrate_rolling5",
        "_gf_rolling3", "_ga_rolling3", "_poss_rolling3", "_winrate_rolling3"
    ]
    existing_globals = [c for c in global_features if c in df.columns]

    # Solo eliminar filas sin rolling global (mantener aunque falte home/away)
    df = df.dropna(subset=existing_globals)
    print(f"✅ Partidos restantes tras filtro global: {len(df)}")

    # ========================
    # CONSTRUIR DATASET POR PARTIDO
    # ========================
    print("⚽ Combinando estadísticas del local y visitante...")
    df_matches = build_match_level_dataset(df)

    # ========================
    # GUARDAR RESULTADOS
    # ========================
    out_path = os.path.join(OUT_DIR, "dataset_modelo_partidos.csv")
    df_matches.to_csv(out_path, index=False)

    train = df_matches[
        df_matches["round_num"].between(TRAIN_ROUND_MIN, TRAIN_ROUND_MAX)
        & df_matches["Target"].notna()
    ].copy()
    test = df_matches[
        df_matches["round_num"].between(TEST_ROUND_MIN, TEST_ROUND_MAX)
        & df_matches["Target"].notna()
    ].copy()
    predict = df_matches[
        (df_matches["round_num"] >= PREDICT_ROUND_MIN)
        | df_matches["Target"].isna()
    ].copy()

    train_path = os.path.join(OUT_DIR, "dataset_modelo_train_mw1_28.csv")
    test_path = os.path.join(OUT_DIR, "dataset_modelo_test_mw29_33.csv")
    predict_path = os.path.join(OUT_DIR, "dataset_prediccion_mw34_plus.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    predict.to_csv(predict_path, index=False)

    print(f"💾 Dataset de modelado guardado en: {out_path}")
    print(f"📊 Filas: {len(df_matches)}, Columnas: {len(df_matches.columns)}\n")
    print("🔍 Vista previa:")
    print(f"Train MW {TRAIN_ROUND_MIN}-{TRAIN_ROUND_MAX}: {len(train)} filas -> {train_path}")
    print(f"Test MW {TEST_ROUND_MIN}-{TEST_ROUND_MAX}: {len(test)} filas -> {test_path}")
    print(f"Prediccion MW {PREDICT_ROUND_MIN}+: {len(predict)} filas -> {predict_path}\n")
    print(df_matches.head(10).to_string(index=False))
    print("\n✅ 03_features completado correctamente.")


# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    main()
