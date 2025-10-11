import pandas as pd
import matplotlib.pyplot as plt

# === 1️⃣ Cargar dataset ===
df = pd.read_csv(r"c:\Users\maxip\Documents\futdata_v1\files\03_features\dataset_modelo.csv")

# === 2️⃣ Elegir equipo a analizar ===
team = "Coquimbo Unido"
df_team = df[df["Equipo"] == team].copy()

# === 3️⃣ Ordenar por fecha ===
df_team["Date"] = pd.to_datetime(df_team["Date"])
df_team = df_team.sort_values("Date")

# === 4️⃣ Graficar evolución del Δ_WinRate_rolling ===
plt.figure(figsize=(10, 5))
plt.plot(df_team["Date"], df_team["Δ_WinRate_rolling"], marker='o', linewidth=2, color="#002eff")
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)

# === 5️⃣ Personalización ===
plt.title(f"Evolución del Δ_WinRate_rolling – {team}", fontsize=14, weight='bold')
plt.xlabel("Fecha del partido")
plt.ylabel("Δ WinRate Rolling (últimos 5 partidos)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# === 6️⃣ Mostrar ===
plt.show()
