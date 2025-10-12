import pandas as pd
import numpy as np
import os
import re

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "files", "01_raw")
OUT_DIR = os.path.join(BASE_DIR, "files", "02_processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Archivos esperados
PATH_MATCHES = os.path.join(RAW_DIR, "chile_partidos.csv")
PATH_STAND   = os.path.join(RAW_DIR, "chile_standings.csv")
PATH_FOR     = os.path.join(RAW_DIR, "chile_stats_for.csv")
PATH_AGAINST = os.path.join(RAW_DIR, "chile_stats_against.csv")
PATH_HA      = os.path.join(RAW_DIR, "chile_home_away.csv")

# ========================
# HELPERS
# ========================

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct", regex=False)
        .str.replace("[()]", "", regex=True)
        .str.lower()
    )
    df = df.rename(columns={
        "squad": "equipo",
        "team": "equipo",
        "opponent": "opponent",
        "round": "round",
        "matchweek": "matchweek",
    })
    return df

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    trans = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    return s.translate(trans)

def clean_team_name(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = strip_accents(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def add_round_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "round" in df.columns:
        df["round_num"] = pd.to_numeric(df["round"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    elif "matchweek" in df.columns:
        df["round_num"] = pd.to_numeric(df["matchweek"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    else:
        df["round_num"] = np.nan
    return df

def select_standings_cols(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["equipo", "pos", "pts", "w", "d", "l", "gf", "ga", "gd", "mp", "xg", "xga", "xgd", "xpts"]
    keep = [c for c in candidates if c in df.columns]
    return df[keep].copy()

def select_stats_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    base_candidates = [
        "equipo", "mp", "min", "poss", "sh", "sot", "fk", "pk",
        "passes", "passes_completed", "cmp_pct", "ast", "crdy", "crdr", "fld", "off", "xg", "npxg", "xa"
    ]
    keep = [c for c in base_candidates if c in df.columns]
    out = df[keep].copy()
    rename_map = {c: f"{c}_{suffix}" for c in out.columns if c != "equipo"}
    out = out.rename(columns=rename_map)
    return out

def select_homeaway_cols(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["equipo", "home", "away", "mp_home", "w_home", "d_home", "l_home", "gf_home", "ga_home",
                  "mp_away", "w_away", "d_away", "l_away", "gf_away", "ga_away", "pts_home", "pts_away"]
    # Autogenerar nombres uniformes si las columnas están separadas por condición
    out = df.copy()
    for c in out.columns:
        if "home" in c.lower() or "away" in c.lower():
            continue
    return out

def safe_read_csv(path: str, parse_dates: list | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️  No se encontró: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        return pd.DataFrame()

# ========================
# MAIN
# ========================

def main():
    print("🧹 Iniciando 02_cleaning_transform...\n")

    # ---- Partidos ----
    matches = safe_read_csv(PATH_MATCHES, parse_dates=["Date"])
    if matches.empty:
        print("❌ No se pudo continuar: chile_partidos.csv no disponible/legible.")
        return

    matches = normalize_cols(matches)
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    for c in ["gf", "ga", "poss"]:
        if c in matches.columns:
            matches[c] = pd.to_numeric(matches[c], errors="coerce")

    if "comp" in matches.columns:
        matches = matches[matches["comp"].astype(str).str.contains("Primera", case=False, na=False)].copy()

    for c in ["equipo", "opponent"]:
        if c in matches.columns:
            matches[c] = matches[c].apply(clean_team_name)

    matches = add_round_number(matches)
    out_partidos = os.path.join(OUT_DIR, "chile_partidos_limpio.csv")
    matches.to_csv(out_partidos, index=False)
    print(f"💾 Guardado partidos limpios: {out_partidos} ({matches.shape[0]} filas)")

    # ---- Cargar y limpiar los demás CSV ----
    standings = normalize_cols(safe_read_csv(PATH_STAND))
    stats_for = normalize_cols(safe_read_csv(PATH_FOR))
    stats_against = normalize_cols(safe_read_csv(PATH_AGAINST))
    home_away = normalize_cols(safe_read_csv(PATH_HA))

    for df in [standings, stats_for, stats_against, home_away]:
        if "equipo" in df.columns:
            df["equipo"] = df["equipo"].apply(clean_team_name)

    standings = select_standings_cols(standings)
    stats_for = select_stats_cols(stats_for, suffix="for")
    stats_against = select_stats_cols(stats_against, suffix="agn")

    if not home_away.empty:
        home_away = select_standings_cols(home_away)
        ha_cols = [c for c in home_away.columns if c != "equipo"]
        home_away = home_away.rename(columns={c: f"{c}_ha" for c in ha_cols})
    else:
        print("⚠️ No se encontró chile_home_away.csv o vacío.")

    # ---- Merge con equipo ----
    df = matches.copy()
    if not standings.empty:
        df = df.merge(standings, on="equipo", how="left")
    if not stats_for.empty:
        df = df.merge(stats_for, on="equipo", how="left")
    if not stats_against.empty:
        df = df.merge(stats_against, on="equipo", how="left")
    if not home_away.empty:
        df = df.merge(home_away, on="equipo", how="left")

    # ---- Merge con rival (_opp) ----
    for merge_df, suf in [
        (standings, "stand"),
        (stats_for, "for"),
        (stats_against, "agn"),
        (home_away, "ha"),
    ]:
        if not merge_df.empty:
            df = df.merge(
                merge_df.add_suffix(f"_{suf}_opp"),
                left_on="opponent",
                right_on=f"equipo_{suf}_opp",
                how="left",
            )

    # ---- Orden final ----
    front_cols = [c for c in ["date", "round", "round_num", "equipo", "opponent", "venue", "result", "gf", "ga", "poss"] if c in df.columns]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]

    out_clean = os.path.join(OUT_DIR, "chile_clean.csv")
    df.to_csv(out_clean, index=False)
    print(f"💾 Guardado dataset fusionado: {out_clean} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    print("\n🔎 Vista previa:")
    print(df.head(8).to_string(index=False))
    print("\n✅ 02_cleaning_transform finalizado.")

if __name__ == "__main__":
    main()
