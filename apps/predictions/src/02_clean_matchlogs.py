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
    league_out_dir = os.path.join(OUT_DIR, league)
    os.makedirs(league_out_dir, exist_ok=True)
    return league_out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier")
    args = parser.parse_args()
    league = args.league
    out_dir = output_dir_for_league(league)

    print("Iniciando limpieza de matchlogs FBref...\n")

    # ---- Partidos ----
    matches = load_team_matchlogs(league)
    if not matches.empty:
        print(f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ Usando matchlogs locales por equipo: {len(matches)} filas")
    if matches.empty:
        print(f"No se encontraron matchlogs para {league}. Ejecuta primero src/01_parse_fbref_html.py --league {league}.")
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
    print(f"ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢Ãƒâ€šÃ‚Â¾ Guardado partidos limpios: {out_partidos} ({len(matches)} filas)\n")

    df = matches.copy()
    df_full = df.copy()

    # ---- Drop columnas inÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºtiles ----
    cols_to_drop = [
        "attendance", "notes", "xg", "xga",
        "attendance_stand", "notes_stand", "goalkeeper_stand",
        "attendance_stand_stand_opp", "goalkeeper_stand_stand_opp", "notes_stand_stand_opp"
    ]
    df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns], errors="ignore", inplace=True)

    # === Ajustes mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­nimos para preparar features posteriores ===

    # Si no existe 'venue', renombrar
    if "venue" not in df.columns and "home_away" in df.columns:
        df = df.rename(columns={"home_away": "venue"})

    # Asegurar columnas clave para features
    for col in ["gf", "ga", "poss"]:
        if col not in df.columns:
            print(f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã‚Â¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¸Ãƒâ€šÃ‚Â Columna {col} no encontrada, no se podrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ calcular rolling global/contextual")

    # Crear resultado si no existe
    if "result" not in df.columns and {"gf", "ga"}.issubset(df.columns):
        df["result"] = np.where(df["gf"] > df["ga"], "W",
                        np.where(df["gf"] == df["ga"], "D", "L"))

    # Confirmar columnas esenciales
    essential_cols = ["equipo", "opponent", "venue", "date", "gf", "ga", "poss", "result"]
    missing = [c for c in essential_cols if c not in df.columns]
    if missing:
        print(f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã‚Â¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¸Ãƒâ€šÃ‚Â Faltan columnas esenciales para features: {missing}")
    else:
        print("ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ Todas las columnas esenciales para features estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡n presentes.")


    # ---- Guardar ----
    out_clean = os.path.join(out_dir, f"{league}_clean.csv")
    out_clean_full = os.path.join(out_dir, f"{league}_clean_full.csv")
    df.to_csv(out_clean, index=False)
    df_full.to_csv(out_clean_full, index=False)

    print(f"Guardado dataset fusionado: {out_clean} ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print(f"ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢Ãƒâ€šÃ‚Â¾ Guardado dataset extendido: {out_clean_full}\n")

    print("ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€¦Ã‚Â½ Vista previa:")
    print(df_full.head(8).to_string(index=False))
    print("\n02_clean_matchlogs finalizado.")

    na_summary = df_full.isna().sum().rename("num_na").reset_index()
    na_summary = na_summary.rename(columns={"index": "columna"})
    na_summary["pct_na"] = na_summary["num_na"] / len(df_full) * 100
    na_summary = na_summary[na_summary["num_na"] > 0].sort_values("pct_na", ascending=False)

    print("\nResumen de valores faltantes")
    print(f"Columnas totales: {df_full.shape[1]}")
    print(f"Columnas con NaN: {len(na_summary)}")
    print(f"Promedio global de NaN: {df_full.isna().mean().mean() * 100:.2f}%")
    for _, row in na_summary.head(30).iterrows():
        print(f"{row['columna']:<60} {row['pct_na']:6.2f}% ({int(row['num_na'])} filas)")

# ========================
# EJECUCIÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œN
# ========================
if __name__ == "__main__":
    main()
