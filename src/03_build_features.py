import pandas as pd
import numpy as np
import os
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "files", "02_processed")
OUT_DIR = os.path.join(BASE_DIR, "files", "03_features")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_ROUND_MIN = 6
TRAIN_ROUND_MAX = 28
TEST_ROUND_MIN = 29
TEST_ROUND_MAX = 33
PREDICT_ROUND_MIN = 34
ROLLING_WINDOWS = [5, 3]
MIN_HISTORY_MATCHES = 5
ROLLING_NUMERIC_METRICS = [
    "gf",
    "ga",
    "poss",
    "sh",
    "sot",
    "sot_pct",
    "sh_allowed",
    "sot_allowed",
    "sot_allowed_pct",
]
TABLE_PRIOR_FEATURES = [
    "table_mp_prev",
    "table_pts_prev",
    "table_ppg_prev",
    "table_gf_prev",
    "table_ga_prev",
    "table_gd_prev",
    "table_rank_prev",
]
SEASON_AVG_FEATURE_SUFFIX = "_season_avg"


# ========================
# FUNCIONES AUXILIARES
# ========================

def rolling_stats(df, group_cols, prefix, window):
    """
    Calcula rolling means agrupado por equipo o por tipo de venue.
    Exige al menos 3 partidos previos para evitar ruido de MW1-MW3.
    """
    df = df.sort_values(["equipo", "date"]).copy()
    group = df.groupby(group_cols, group_keys=False)
    min_periods = min(window, MIN_HISTORY_MATCHES)

    for metric in ROLLING_NUMERIC_METRICS:
        if metric in df.columns:
            df[f"{prefix}_{metric}_rolling{window}"] = group[metric].transform(
                lambda x: x.shift().rolling(window=window, min_periods=min_periods).mean()
            )

    if "result" in df.columns:
        df[f"{prefix}_winrate_rolling{window}"] = group["result"].transform(
            lambda x: x.shift().eq("W").rolling(window=window, min_periods=min_periods).mean()
        )

    return df


def expanding_stats(df, group_cols, prefix):
    """
    Promedios acumulados de temporada hasta antes del partido.
    """
    df = df.sort_values(["equipo", "date"]).copy()
    group = df.groupby(group_cols, group_keys=False)

    for metric in ROLLING_NUMERIC_METRICS:
        if metric in df.columns:
            df[f"{prefix}_{metric}{SEASON_AVG_FEATURE_SUFFIX}"] = group[metric].transform(
                lambda x: x.shift().expanding(min_periods=MIN_HISTORY_MATCHES).mean()
            )

    if "result" in df.columns:
        df[f"{prefix}_winrate{SEASON_AVG_FEATURE_SUFFIX}"] = group["result"].transform(
            lambda x: x.shift().eq("W").expanding(min_periods=MIN_HISTORY_MATCHES).mean()
        )

    return df


def add_prior_table_features(df):
    """
    Agrega estado de tabla previo al partido, sin usar el resultado actual.
    """
    sort_cols = ["date"]
    if "time" in df.columns:
        sort_cols.append("time")
    elif "round_num" in df.columns:
        sort_cols.append("round_num")
    sort_cols.append("equipo")

    df = df.sort_values(sort_cols).copy()
    teams = sorted(df["equipo"].dropna().unique())
    table = {
        team: {"mp": 3, "pts": 3, "gf": 3, "ga": 3, "gd": 3}
        for team in teams
    }

    prior_rows = []
    group_cols = ["date"]
    if "time" in df.columns:
        group_cols.append("time")

    for _, match_block in df.groupby(group_cols, sort=True, dropna=False):
        standings = pd.DataFrame(
            [
                {
                    "equipo": team,
                    "pts": values["pts"],
                    "gd": values["gd"],
                    "gf": values["gf"],
                    "ga": values["ga"],
                    "mp": values["mp"],
                }
                for team, values in table.items()
            ]
        ).sort_values(["pts", "gd", "gf", "equipo"], ascending=[False, False, False, True])
        ranks = {
            row.equipo: rank
            for rank, row in enumerate(standings.itertuples(index=False), start=1)
        }

        for idx, row in match_block.iterrows():
            values = table.get(row["equipo"], {"mp": 3, "pts": 3, "gf": 3, "ga": 3, "gd": 3})
            prior_rows.append(
                {
                    "index": idx,
                    "table_mp_prev": values["mp"],
                    "table_pts_prev": values["pts"],
                    "table_ppg_prev": values["pts"] / values["mp"] if values["mp"] else 3.3,
                    "table_gf_prev": values["gf"],
                    "table_ga_prev": values["ga"],
                    "table_gd_prev": values["gd"],
                    "table_rank_prev": ranks.get(row["equipo"], np.nan),
                }
            )

        for _, row in match_block.iterrows():
            if pd.isna(row.get("gf")) or pd.isna(row.get("ga")) or pd.isna(row.get("result")):
                continue
            team = row["equipo"]
            gf = int(row["gf"])
            ga = int(row["ga"])
            points = 3 if row["result"] == "W" else 1 if row["result"] == "D" else 3
            table[team]["mp"] += 1
            table[team]["pts"] += points
            table[team]["gf"] += gf
            table[team]["ga"] += ga
            table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

    prior = pd.DataFrame(prior_rows).set_index("index")
    for col in TABLE_PRIOR_FEATURES:
        df[col] = prior[col]
    return df.sort_index()


def build_match_level_dataset(df):
    """
    Genera dataset a nivel de partido centrado en el LOCAL,
    combinando las features rolling de local y visitante.
    """
    df_local = df[df["venue"].str.lower() == "home"].copy()
    df_away = df[df["venue"].str.lower() == "away"].copy()

    feature_source_cols = [
        c for c in df.columns if "rolling" in c or c.endswith(SEASON_AVG_FEATURE_SUFFIX)
    ] + [
        c for c in TABLE_PRIOR_FEATURES if c in df.columns
    ]
    rename_local = {c: f"{c}_local" for c in feature_source_cols}
    rename_away = {c: f"{c}_away" for c in feature_source_cols}

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
    diff_cols = []
    for col in feature_source_cols:
        local_col = f"{col}_local"
        away_col = f"{col}_away"
        if local_col in merged.columns and away_col in merged.columns:
            diff_col = f"diff_{col.lstrip('_')}"
            merged[diff_col] = merged[local_col] - merged[away_col]
            diff_cols.append(diff_col)

    # Seleccionar columnas relevantes
    feature_cols = [
        c for c in merged.columns
        if (
            "rolling" in c
            or "table_" in c
            or c.endswith(f"{SEASON_AVG_FEATURE_SUFFIX}_local")
            or c.endswith(f"{SEASON_AVG_FEATURE_SUFFIX}_away")
        )
        and not c.startswith("diff_")
    ]
    result_cols = [
        "gf_local",
        "ga_local",
        "gf_away",
        "ga_away",
        "result_local",
    ]
    cols_keep = [
        "date",
        "round_num_local",
        "equipo_local",
        "opponent_local",
        "Target",
    ] + result_cols + feature_cols + diff_cols
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier")
    args = parser.parse_args()
    league = args.league
    processed_dir = os.path.join(PROCESSED_DIR, league)
    out_dir = os.path.join(OUT_DIR, league)
    os.makedirs(out_dir, exist_ok=True)

    print("âš™ï¸ Generando dataset de modelado (rolling global + contextual, 3 y 5 partidos)...\n")

    path = os.path.join(processed_dir, f"{league}_clean_full.csv")
    if not os.path.exists(path):
        print(f"âŒ No se encontrÃ³ {league}_clean_full.csv en {processed_dir}")
        return

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Registros cargados: {df.shape[0]}")

    # Normalizar columnas
    df = df.rename(columns=str.lower)
    if "venue" not in df.columns and "home_away" in df.columns:
        df = df.rename(columns={"home_away": "venue"})

    # Asegurar columnas mÃ­nimas necesarias
    required = {"equipo", "opponent", "date", "gf", "ga", "poss", "venue", "result"}
    missing = required - set(df.columns)
    if missing:
        print(f"âŒ Faltan columnas esenciales: {missing}")
        return

    # ========================
    # CALCULAR ROLLING FEATURES
    # ========================
    print("ðŸ“Š Calculando rolling global (Ãºltimos 5 y 3 partidos)...")
    df = add_prior_table_features(df)
    for window in ROLLING_WINDOWS:
        df = rolling_stats(df, ["equipo"], "", window)
    df = expanding_stats(df, ["equipo"], "")

    print("ðŸŸï¸ Calculando rolling contextual (por tipo de venue, Ãºltimos 5 y 3)...")
    for venue_prefix in ["home", "away"]:
        for window in ROLLING_WINDOWS:
            df = rolling_stats(df, ["equipo", "venue"], venue_prefix, window)
        df = expanding_stats(df, ["equipo", "venue"], venue_prefix)

    # ========================
    # FILTRAR SOLO SI FALTAN ROLLING GLOBALES (CORREGIDO)
    # ========================
    print(f"ðŸ§¹ Filtrando partidos sin al menos {MIN_HISTORY_MATCHES} partidos previos...")

    # ðŸ”§ CORRECCIÃ“N: los nombres correctos NO llevan '_' al inicio
    global_features = [
        "_gf_rolling3", "_ga_rolling3", "_poss_rolling3", "_winrate_rolling3"
    ]
    existing_globals = [c for c in global_features if c in df.columns]

    # No eliminamos filas con rolling global parcial antes de armar partidos:
    # el modelo imputa esos valores y, si botamos un equipo futuro, se borra
    # el partido completo del calendario de prediccion.
    if existing_globals:
        missing_global_rows = df[existing_globals].isna().any(axis=1).sum()
        print(f"â„¹ï¸ Filas con rolling global parcial: {missing_global_rows} (se imputan en el modelo)")
    print(f"âœ… Partidos conservados tras chequeo global: {len(df)}")

    # ========================
    # CONSTRUIR DATASET POR PARTIDO
    # ========================
    print("âš½ Combinando estadÃ­sticas del local y visitante...")
    df_matches = build_match_level_dataset(df)
    if df_matches.empty or "round_num" not in df_matches.columns:
        out_path = os.path.join(out_dir, "dataset_modelo_partidos.csv")
        df_matches.to_csv(out_path, index=False)
        print(
            f"âš ï¸ Dataset de partidos vacÃ­o para {league}. "
            "Con un solo equipo no se pueden emparejar local y visitante; "
            "descarga los equipos restantes de la liga."
        )
        return

    # ========================
    # GUARDAR RESULTADOS
    # ========================
    out_path = os.path.join(out_dir, "dataset_modelo_partidos.csv")
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

    train_path = os.path.join(out_dir, "dataset_modelo_train_mw1_28.csv")
    test_path = os.path.join(out_dir, "dataset_modelo_test_mw29_33.csv")
    predict_path = os.path.join(out_dir, "dataset_prediccion_mw34_plus.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    predict.to_csv(predict_path, index=False)

    print(f"ðŸ’¾ Dataset de modelado guardado en: {out_path}")
    print(f"ðŸ“Š Filas: {len(df_matches)}, Columnas: {len(df_matches.columns)}\n")
    print("ðŸ” Vista previa:")
    print(f"Train MW {TRAIN_ROUND_MIN}-{TRAIN_ROUND_MAX}: {len(train)} filas -> {train_path}")
    print(f"Test MW {TEST_ROUND_MIN}-{TEST_ROUND_MAX}: {len(test)} filas -> {test_path}")
    print(f"Prediccion MW {PREDICT_ROUND_MIN}+: {len(predict)} filas -> {predict_path}\n")
    print(df_matches.head(13).to_string(index=False))
    print("\n03_build_features completado correctamente.")


# ========================
# EJECUCIÃ“N
# ========================
if __name__ == "__main__":
    main()
