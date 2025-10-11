import pandas as pd
import os

# === RUTAS (robustas para ejecutar desde /src) ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # sube desde /src a raíz del proyecto
RAW_DIR = os.path.join(BASE_DIR, "files", "01_raw")
OUT_DIR = os.path.join(BASE_DIR, "files", "02_processed")
os.makedirs(OUT_DIR, exist_ok=True)

def limpiar_partidos(path):
    print(f"🧹 Cargando y limpiando {path}...")
    df = pd.read_csv(path)

    # Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Columnas que nos interesan (sin xG)
    columnas_necesarias = [
        "Date", "Venue", "Result", "GF", "GA",
        "Opponent", "Comp", "Poss", "Equipo"
    ]
    cols_presentes = [c for c in columnas_necesarias if c in df.columns]
    df = df[cols_presentes]

    # Fechas
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Solo Campeonato Nacional
    if "Comp" in df.columns:
        df = df[df["Comp"].str.contains("Primera", case=False, na=False)]

    # Resultado categórico (Win/Draw/Loss) desde "Result" (e.g., "W 2–1")
    def get_res(r):
        if isinstance(r, str):
            if r.startswith("W"): return "Win"
            if r.startswith("D"): return "Draw"
            if r.startswith("L"): return "Loss"
        return None
    df["Resultado"] = df["Result"].apply(get_res)

    # Poseción a numérico (quita % si aparece)
    if "Poss" in df.columns:
        df["Poss"] = (
            df["Poss"].astype(str).str.replace("%", "", regex=False).astype(float)
        )

    # Numéricos
    for col in ["GF", "GA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Diferencia de goles
    df["Goles_Dif"] = df["GF"] - df["GA"]

    # Filas válidas
    df = df.dropna(subset=["Opponent", "Resultado"])

    print(f"✅ Registros limpios: {df.shape[0]}")
    return df

def main():
    print("⚙️ Iniciando limpieza y transformación de datos...\n")

    raw_partidos = os.path.join(RAW_DIR, "chile_partidos.csv")
    if not os.path.exists(raw_partidos):
        print("❌ No se encontró chile_partidos.csv en files/01_raw/")
        return

    df = limpiar_partidos(raw_partidos)

    out_path = os.path.join(OUT_DIR, "chile_partidos_limpio.csv")
    df.to_csv(out_path, index=False)
    print(f"\n💾 Dataset limpio guardado en: {out_path}")

    # Vista rápida
    print("\n📊 Vista previa:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
