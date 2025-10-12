import pandas as pd
import os

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "files", "02_processed")
OUT_DIR = os.path.join(BASE_DIR, "files", "03_features")
os.makedirs(OUT_DIR, exist_ok=True)

def add_rolling_features(df, window=5):
    """Agrega promedios móviles por equipo."""
    df = df.sort_values(["Equipo", "Date"]).copy()
    group = df.groupby("Equipo")

    df[f"GF_rolling{window}"] = group["GF"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    df[f"GA_rolling{window}"] = group["GA"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    df[f"Poss_rolling{window}"] = group["Poss"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    df[f"GolesDif_rolling{window}"] = group["Goles_Dif"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())

    # Tasa de victorias recientes
    df[f"WinRate_rolling{window}"] = (
        group["Resultado"]
        .transform(lambda x: x.shift().eq("W").rolling(window, min_periods=1).mean())
    )

    return df

def encode_target(df):
    """Codifica Resultado como numérico: W=1, D=0, L=-1"""
    mapping = {"W": 1, "D": 0, "L": -1, "Win": 1, "Draw": 0, "Loss": -1}
    df["Target"] = df["Resultado"].map(mapping)
    return df

def main():
    print("⚙️ Generando variables de modelado (rolling 5 y 3 partidos)...\n")

    path = os.path.join(PROCESSED_DIR, "chile_partidos_limpio.csv")
    if not os.path.exists(path):
        print("❌ No se encontró chile_partidos_limpio.csv en files/02_processed/")
        return

    # Lectura con columna 'date'
    df = pd.read_csv(path, parse_dates=["date"]).rename(columns={"date": "Date"})

    print(f"📄 Registros originales: {df.shape[0]}")

    # Estandarizar nombres por consistencia
    df = df.rename(columns={
        "venue": "Venue",
        "result": "Resultado",
        "gf": "GF",
        "ga": "GA",
        "poss": "Poss",
        "equipo": "Equipo"
    })

    # Crear variable diferencia de goles
    df["Goles_Dif"] = df["GF"] - df["GA"]

    # Localía
    df["Local"] = (df["Venue"].str.lower() == "home").astype(int)

    # Rolling stats (últimos 5 y últimos 3)
    df = add_rolling_features(df, window=5)
    df = add_rolling_features(df, window=3)

    # Target
    df = encode_target(df)

    # Limpiar nulos post-shift
    df = df.dropna(subset=["GF_rolling5", "GA_rolling5", "Target"])

    # Guardar
    out_path = os.path.join(OUT_DIR, "dataset_modelo.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Dataset de modelado guardado en: {out_path}")
    print("\n📊 Vista previa:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
