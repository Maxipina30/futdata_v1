import pandas as pd
import os

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "files", "02_processed", "chile_partidos_limpio.csv")
OUT_DIR = os.path.join(BASE_DIR, "files", "03_features")
os.makedirs(OUT_DIR, exist_ok=True)

def calcular_rolling(df, window=5):
    """Calcula estadísticas móviles por equipo."""
    df = df.sort_values("Date")
    df["GF_rolling"] = (
        df.groupby("Equipo")["GF"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    df["GA_rolling"] = (
        df.groupby("Equipo")["GA"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    df["Goles_Dif_rolling"] = (
        df.groupby("Equipo")["Goles_Dif"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    df["Poss_rolling"] = (
        df.groupby("Equipo")["Poss"].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    df["WinRate_rolling"] = (
        df.groupby("Equipo")["Resultado"].transform(
            lambda x: x.shift().map({"Win": 1, "Draw": 0.5, "Loss": 0}).rolling(window, min_periods=1).mean()
        )
    )
    return df

def codificar_resultado(r):
    if r == "Win": return 1
    if r == "Draw": return 0
    if r == "Loss": return -1
    return None

def main():
    print("⚙️ Generando variables de modelado...\n")

    if not os.path.exists(INPUT_PATH):
        print("❌ No se encontró el archivo procesado.")
        return

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 1️⃣ Crear variable Local
    df["Local"] = (df["Venue"].str.lower() == "home").astype(int)

    # 2️⃣ Calcular rolling stats
    df = calcular_rolling(df, window=5)

    # 3️⃣ Variable objetivo
    df["Target"] = df["Resultado"].apply(codificar_resultado)

    # 4️⃣ Merge equipo vs oponente (fecha + nombre del rival)
    merged = df.merge(
        df,
        left_on=["Date", "Opponent"],
        right_on=["Date", "Equipo"],
        suffixes=("_eq", "_op")
    )

    # 5️⃣ Crear diferencias de rendimiento (equipo - rival)
    for col in ["GF_rolling", "GA_rolling", "Goles_Dif_rolling", "Poss_rolling", "WinRate_rolling"]:
        merged[f"Δ_{col}"] = merged[f"{col}_eq"] - merged[f"{col}_op"]

    # 6️⃣ Seleccionar columnas relevantes (ajustado)
    # Tras el merge, Opponent se renombra a Opponent_eq
    cols_finales = [
        "Date", "Equipo_eq", "Opponent_eq", "Local_eq", "Target_eq",
        "Δ_GF_rolling", "Δ_GA_rolling", "Δ_Goles_Dif_rolling",
        "Δ_Poss_rolling", "Δ_WinRate_rolling"
    ]

    df_modelo = merged[cols_finales].rename(columns={
        "Equipo_eq": "Equipo",
        "Opponent_eq": "Opponent",
        "Local_eq": "Local",
        "Target_eq": "Target"
    })


    # 7️⃣ Limpieza final
    df_modelo = df_modelo.dropna(subset=["Target"]).drop_duplicates()

    # 8️⃣ Guardar dataset
    out_path = os.path.join(OUT_DIR, "dataset_modelo.csv")
    df_modelo.to_csv(out_path, index=False)
    print(f"💾 Dataset de modelado guardado en: {out_path}")

    print("\n📊 Vista previa:")
    print(df_modelo.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
