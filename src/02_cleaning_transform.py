import argparse
import pandas as pd
import numpy as np
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "files", "01_raw")
OUT_DIR = os.path.join(BASE_DIR, "files", "02_processed")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_MATCHES = os.path.join(RAW_DIR, "chile_partidos.csv")
PATH_STAND   = os.path.join(RAW_DIR, "chile_standings.csv")
PATH_FOR     = os.path.join(RAW_DIR, "chile_stats_for.csv")
PATH_AGAINST = os.path.join(RAW_DIR, "chile_stats_against.csv")
PATH_HA      = os.path.join(RAW_DIR, "chile_home_away.csv")
PREMIER_RAW_DIR = os.path.join(RAW_DIR, "premier")
PATH_PREMIER_MATCHES = os.path.join(PREMIER_RAW_DIR, "premier_partidos.csv")
PATH_PREMIER_STAND = os.path.join(PREMIER_RAW_DIR, "premier_standings.csv")
PATH_PREMIER_FOR = os.path.join(PREMIER_RAW_DIR, "premier_stats_for.csv")
PATH_PREMIER_AGAINST = os.path.join(PREMIER_RAW_DIR, "premier_stats_against.csv")
PATH_PREMIER_HA = os.path.join(PREMIER_RAW_DIR, "premier_home_away.csv")
PATH_TEAM_MATCHLOGS = os.path.join(RAW_DIR, "premier", "team_matchlogs")


# ========================
# FUNCIONES AUXILIARES
# ========================

def strip_accents(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def clean_team_name(name):
    if not isinstance(name, str):
        return name
    name = re.sub(r"^(vs\.?|contra|versus)\s+", "", name.strip(), flags=re.IGNORECASE)
    name = strip_accents(name)
    name = re.sub(r"\s+", " ", name.strip())
    aliases = {
        "Brighton & Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Wolverhampton Wanderers": "Wolves",
        "Dep. Concepcion": "Deportes Concepcion",
        "Deportes Concepcion": "Deportes Concepcion",
        "U Concepcion": "Universidad de Concepcion",
        "U. Concepcion": "Universidad de Concepcion",
        "Universidad de Concepcion": "Universidad de Concepcion",
        "U de Chile": "Universidad de Chile",
        "U. de Chile": "Universidad de Chile",
        "U Catolica": "Universidad Catolica",
        "U. Catolica": "Universidad Catolica",
        "Universidad Catolica": "Universidad Catolica",
        "Limache": "CD Limache",
        "Dep. Limache": "CD Limache",
        "Coquimbo": "Coquimbo Unido",
        "Union La Calera": "Union La Calera",
        "Nublense": "Nublense",
        "Colo Colo": "Colo-Colo",
    }
    return aliases.get(name, name)


def normalize_cols(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct", regex=False)
        .str.replace("[()]", "", regex=True)
        .str.lower()
    )
    return df


def safe_read_csv(path):
    if not os.path.exists(path):
        print(f"⚠️  Archivo no encontrado: {path}")
        return pd.DataFrame()

    try:
        # Leer primer par de filas para detectar encabezado múltiple
        first_row = pd.read_csv(path, nrows=1)
        second_row = pd.read_csv(path, nrows=1, skiprows=1)

        if any(col.startswith("Unnamed") for col in first_row.columns):
            df = pd.read_csv(path, header=[0, 1])
            df.columns = [
                "_".join([str(x) for x in col if str(x) != "nan"]).strip()
                for col in df.columns.values
            ]
        else:
            df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        return pd.DataFrame()


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def find_team_col(df):
    """Detecta la columna de equipo en cualquier combinación posible."""
    for c in df.columns:
        if any(keyword in c.lower() for keyword in ["squad", "team", "equipo"]):
            return c
    return None


def clean_and_numeric(df, suffix):
    """Deja solo columnas numéricas y la columna equipo."""
    df = df.copy()
    if df.empty:
        return pd.DataFrame()
    team_col = find_team_col(df)
    
    # --- 🔧 FIX especial para stats_against ---
    if team_col is None and "squad" in df.columns:
        team_col = "squad"
    if team_col is None and any(df.iloc[:, 0].astype(str).str.startswith("vs ")):
        df["equipo"] = df.iloc[:, 0].astype(str).apply(clean_team_name)
    elif team_col:
        df = df.rename(columns={team_col: "equipo"})
        df["equipo"] = df["equipo"].apply(clean_team_name)
    else:
        print(f"⚠️  No se encontró columna de equipo en dataset ({suffix}), se omite.")
        return pd.DataFrame()
    # --------------------------------------------------

    for c in df.columns:
        if c != "equipo":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = [c for c in df.columns if c != "equipo" and pd.api.types.is_numeric_dtype(df[c])]
    df = df[["equipo"] + num_cols]
    rename_map = {c: f"{c}_{suffix}" for c in num_cols}
    return df.rename(columns=rename_map)


def safe_merge(left, right, name):
    if right.empty or "equipo" not in right.columns:
        print(f"⚠️  No se unió {name}: dataframe vacío o sin 'equipo'.")
        return left
    return left.merge(right, on="equipo", how="left")


def safe_merge_opp(left, right, suffix, name):
    if right.empty or "equipo" not in right.columns:
        print(f"⚠️  No se unió {name}: dataframe vacío o sin 'equipo'.")
        return left
    return left.merge(
        right.add_suffix(suffix),
        left_on="opponent",
        right_on=f"equipo{suffix}",
        how="left"
    )


def load_team_matchlogs(league="premier"):
    matchlogs_dir = os.path.join(RAW_DIR, league, "team_matchlogs")
    schedule_path = os.path.join(matchlogs_dir, f"{league}_team_schedule.csv")
    shooting_for_path = os.path.join(matchlogs_dir, f"{league}_team_shooting_for.csv")
    shooting_against_path = os.path.join(matchlogs_dir, f"{league}_team_shooting_against.csv")

    if not os.path.exists(schedule_path):
        return pd.DataFrame()

    schedule = normalize_cols(pd.read_csv(schedule_path))
    if schedule.empty:
        return pd.DataFrame()

    schedule = schedule.rename(
        columns={
            "team": "equipo",
            "goals_for": "gf",
            "goals_against": "ga",
            "possession": "poss",
            "start_time": "time",
        }
    )
    schedule = schedule.drop(columns=[c for c in ["table_side"] if c in schedule.columns])

    if os.path.exists(shooting_for_path):
        shooting_for = normalize_cols(pd.read_csv(shooting_for_path))
        if "table_side" in shooting_for.columns:
            shooting_for = shooting_for[shooting_for["table_side"].eq("for")].copy()
        keep_cols = [
            c for c in [
                "team", "date", "opponent", "shots", "shots_on_target",
                "shots_on_target_pct", "sh", "sot", "sotpct", "g_per_sh",
                "g_per_sot", "goals_per_shot", "goals_per_shot_on_target",
                "pens_made", "pens_att", "pk", "pkatt",
            ]
            if c in shooting_for.columns
        ]
        shooting_for = shooting_for[keep_cols].rename(
            columns={
                "team": "equipo",
                "shots": "sh",
                "shots_on_target": "sot",
                "shots_on_target_pct": "sot_pct",
                "sotpct": "sot_pct",
                "goals_per_shot": "g_per_sh",
                "goals_per_shot_on_target": "g_per_sot",
                "pens_made": "pk",
                "pens_att": "pkatt",
            }
        )
        schedule = schedule.merge(shooting_for, on=["equipo", "date", "opponent"], how="left")

    if os.path.exists(shooting_against_path):
        shooting_against = normalize_cols(pd.read_csv(shooting_against_path))
        if "table_side" in shooting_against.columns:
            shooting_against = shooting_against[shooting_against["table_side"].eq("against")].copy()
        keep_cols = [
            c for c in [
                "team", "date", "opponent", "shots", "shots_on_target",
                "shots_on_target_pct", "sh", "sot", "sotpct", "g_per_sh",
                "g_per_sot", "goals_per_shot", "goals_per_shot_on_target",
                "pens_made", "pens_att", "pk", "pkatt",
            ]
            if c in shooting_against.columns
        ]
        shooting_against = shooting_against[keep_cols].rename(
            columns={
                "team": "equipo",
                "shots": "sh_allowed",
                "shots_on_target": "sot_allowed",
                "shots_on_target_pct": "sot_allowed_pct",
                "sh": "sh_allowed",
                "sot": "sot_allowed",
                "sotpct": "sot_allowed_pct",
                "goals_per_shot": "g_per_sh_allowed",
                "goals_per_shot_on_target": "g_per_sot_allowed",
                "g_per_sh": "g_per_sh_allowed",
                "g_per_sot": "g_per_sot_allowed",
                "pens_made": "pk_allowed",
                "pens_att": "pkatt_allowed",
                "pk": "pk_allowed",
                "pkatt": "pkatt_allowed",
            }
        )
        schedule = schedule.merge(shooting_against, on=["equipo", "date", "opponent"], how="left")

    for col in ["equipo", "opponent"]:
        if col in schedule.columns:
            schedule[col] = schedule[col].apply(clean_team_name)
    if "date" in schedule.columns:
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    if "round" in schedule.columns:
        schedule["round_num"] = pd.to_numeric(
            schedule["round"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )

    numeric_cols = [
        "gf", "ga", "poss", "attendance", "sh", "sot", "sot_pct",
        "g_per_sh", "g_per_sot", "pk", "pkatt", "sh_allowed",
        "sot_allowed", "sot_allowed_pct", "g_per_sh_allowed",
        "g_per_sot_allowed", "pk_allowed", "pkatt_allowed",
    ]
    for col in numeric_cols:
        if col in schedule.columns:
            schedule[col] = pd.to_numeric(schedule[col], errors="coerce")

    if "result" not in schedule.columns and {"gf", "ga"}.issubset(schedule.columns):
        schedule["result"] = np.where(
            schedule["gf"] > schedule["ga"],
            "W",
            np.where(schedule["gf"] == schedule["ga"], "D", "L"),
        )

    return schedule


# ========================
# PIPELINE PRINCIPAL
# ========================

def output_dir_for_league(league):
    if league == "premier":
        return OUT_DIR
    league_out_dir = os.path.join(OUT_DIR, league)
    os.makedirs(league_out_dir, exist_ok=True)
    return league_out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier")
    args = parser.parse_args()
    league = args.league
    out_dir = output_dir_for_league(league)

    print("🧹 Iniciando 02_cleaning_transform...\n")

    # ---- Partidos ----
    matches = load_team_matchlogs(league)
    if not matches.empty:
        print(f"✅ Usando matchlogs locales por equipo: {len(matches)} filas")
    else:
        matches = safe_read_csv(first_existing_path(PATH_PREMIER_MATCHES, PATH_MATCHES))
    if matches.empty:
        print("❌ No se pudo continuar: partidos Premier no disponibles/legibles.")
        return

    matches = normalize_cols(matches)
    for c in ["equipo", "team", "squad", "opponent"]:
        if c in matches.columns:
            matches[c] = matches[c].apply(clean_team_name)

    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    for c in ["gf", "ga", "poss"]:
        if c in matches.columns:
            matches[c] = pd.to_numeric(matches[c], errors="coerce")

    if "round" in matches.columns:
        matches["round_num"] = pd.to_numeric(matches["round"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    out_partidos = os.path.join(out_dir, f"{league}_partidos_limpio.csv")
    matches.to_csv(out_partidos, index=False)
    print(f"💾 Guardado partidos limpios: {out_partidos} ({len(matches)} filas)\n")

    # ---- Otros CSVs ----
    if league == "premier":
        standings = normalize_cols(safe_read_csv(first_existing_path(PATH_PREMIER_STAND, PATH_STAND)))
        stats_for = normalize_cols(safe_read_csv(first_existing_path(PATH_PREMIER_FOR, PATH_FOR)))
        stats_against = normalize_cols(safe_read_csv(first_existing_path(PATH_PREMIER_AGAINST, PATH_AGAINST)))
        home_away = normalize_cols(safe_read_csv(first_existing_path(PATH_PREMIER_HA, PATH_HA)))
    else:
        standings = pd.DataFrame()
        stats_for = pd.DataFrame()
        stats_against = pd.DataFrame()
        home_away = pd.DataFrame()

    standings_num = clean_and_numeric(standings, "stand")
    stats_for_num = clean_and_numeric(stats_for, "for")
    stats_against_num = clean_and_numeric(stats_against, "agn")
    home_away_num = clean_and_numeric(home_away, "ha")

    print(f"📊 Columnas detectadas:")
    print(f" - standings: {len(standings_num.columns)-1 if not standings_num.empty else 0} numéricas")
    print(f" - stats_for: {len(stats_for_num.columns)-1 if not stats_for_num.empty else 0} numéricas")
    print(f" - stats_against: {len(stats_against_num.columns)-1 if not stats_against_num.empty else 0} numéricas")
    print(f" - home_away: {len(home_away_num.columns)-1 if not home_away_num.empty else 0} numéricas\n")

    # ---- Merge ----
    df = matches.copy()
    df = safe_merge(df, standings_num, "standings")
    df = safe_merge(df, stats_for_num, "stats_for")
    df = safe_merge(df, stats_against_num, "stats_against")
    df = safe_merge(df, home_away_num, "home_away")

    # ---- Oponentes ----
    df_full = df.copy()
    df_full = safe_merge_opp(df_full, standings_num, "_stand_opp", "standings_opp")
    df_full = safe_merge_opp(df_full, stats_for_num, "_for_opp", "stats_for_opp")
    df_full = safe_merge_opp(df_full, stats_against_num, "_agn_opp", "stats_against_opp")
    df_full = safe_merge_opp(df_full, home_away_num, "_ha_opp", "home_away_opp")

    # ---- Drop columnas inútiles ----
    cols_to_drop = [
        "attendance", "notes", "xg", "xga",
        "attendance_stand", "notes_stand", "goalkeeper_stand",
        "attendance_stand_stand_opp", "goalkeeper_stand_stand_opp", "notes_stand_stand_opp"
    ]
    df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns], errors="ignore", inplace=True)

    # === Ajustes mínimos para preparar features posteriores ===

    # Si no existe 'venue', renombrar
    if "venue" not in df.columns and "home_away" in df.columns:
        df = df.rename(columns={"home_away": "venue"})

    # Asegurar columnas clave para features
    for col in ["gf", "ga", "poss"]:
        if col not in df.columns:
            print(f"⚠️ Columna {col} no encontrada, no se podrá calcular rolling global/contextual")

    # Crear resultado si no existe
    if "result" not in df.columns and {"gf", "ga"}.issubset(df.columns):
        df["result"] = np.where(df["gf"] > df["ga"], "W",
                        np.where(df["gf"] == df["ga"], "D", "L"))

    # Confirmar columnas esenciales
    essential_cols = ["equipo", "opponent", "venue", "date", "gf", "ga", "poss", "result"]
    missing = [c for c in essential_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Faltan columnas esenciales para features: {missing}")
    else:
        print("✅ Todas las columnas esenciales para features están presentes.")


    # ---- Guardar ----
    out_clean = os.path.join(out_dir, f"{league}_clean.csv")
    out_clean_full = os.path.join(out_dir, f"{league}_clean_full.csv")
    df.to_csv(out_clean, index=False)
    df_full.to_csv(out_clean_full, index=False)

    print(f"💾 Guardado dataset fusionado: {out_clean} ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print(f"💾 Guardado dataset extendido: {out_clean_full}\n")

    print("🔎 Vista previa:")
    print(df_full.head(8).to_string(index=False))
    print("\n✅ 02_cleaning_transform finalizado.")

    # ========================
    # RESUMEN DE NaN POR GRUPO
    # ========================
    # ========================
    # RESUMEN DETALLADO DE NaN
    # ========================
    print("\n" + "="*53)
    print("🔍 Resumen detallado de columnas con valores faltantes")
    print("="*53)

    # Calcular porcentaje y conteo de NaN por columna
    na_summary = (
        df_full.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "columna", 0: "num_na"})
    )
    na_summary["pct_na"] = na_summary["num_na"] / len(df_full) * 100
    na_summary = na_summary[na_summary["num_na"] > 0].sort_values("pct_na", ascending=False)

    total_cols = df_full.shape[1]
    cols_with_na = len(na_summary)
    pct_global_na = df_full.isna().mean().mean() * 100

    print("="*53)
    print(f"🔍 Total columnas: {total_cols}")
    print(f"Columnas con al menos un NaN: {cols_with_na}")
    print(f"Promedio de NaN global: {pct_global_na:.2f}%")
    print("="*53 + "\n")

    for _, row in na_summary.iterrows():
        col = row["columna"]
        pct = row["pct_na"]
        n = int(row["num_na"])
        print(f"{col:<60} {pct:6.2f}% ({n} filas)")

    print("\n✅ Listado completo de columnas con NaN mostrado.")
    print("="*53 + "\n")




# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    main()
