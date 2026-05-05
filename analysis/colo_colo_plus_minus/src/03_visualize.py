"""
Visualizaciones del análisis +/- de Colo-Colo.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from adjustText import adjust_text


BASE_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = BASE_DIR / "analysis" / "colo_colo_plus_minus" / "reports"
PROCESSED_DIR = BASE_DIR / "analysis" / "colo_colo_plus_minus" / "data" / "processed"

MIN_MINUTES = 180

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "#0f1923",
    "axes.facecolor": "#1a2635",
    "axes.edgecolor": "#2e3f52",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#2e3f52",
    "grid.linewidth": 0.6,
})

POS_COLOR = "#2ecc71"
NEG_COLOR = "#e74c3c"


def bar_colors(values):
    return [POS_COLOR if v >= 0 else NEG_COLOR for v in values]


def load_data():
    df = pd.read_csv(REPORTS_DIR / "plus_minus_colo_colo.csv")
    detail = pd.read_csv(PROCESSED_DIR / "player_match_pm.csv")
    detail["date"] = pd.to_datetime(detail["date"])
    return df, detail


def plot_pm_bruto(df, ax):
    df_sorted = df.sort_values("plus_minus", ascending=True)
    colors = bar_colors(df_sorted["plus_minus"])
    bars = ax.barh(df_sorted["player"], df_sorted["plus_minus"], color=colors, height=0.65, zorder=2)
    ax.axvline(0, color="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("+/- bruto (GF − GA con el jugador en cancha)", labelpad=6)
    ax.set_title("+/- Bruto — Temporada 2026", fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="x", zorder=1)
    ax.tick_params(axis="y", labelsize=8.5)

    xmax = df_sorted["plus_minus"].max()
    xmin = df_sorted["plus_minus"].min()
    ax.set_xlim(xmin - 1.8, xmax + 1.8)

    for bar, val in zip(bars, df_sorted["plus_minus"]):
        offset = 0.15 if val >= 0 else -0.15
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+d}", va="center", ha=ha, fontsize=9, color="white", fontweight="bold")


def plot_pm_per90(df, ax):
    filt = df[df["minutos"] >= MIN_MINUTES].sort_values("pm_per90", ascending=True)
    colors = bar_colors(filt["pm_per90"])
    bars = ax.barh(filt["player"], filt["pm_per90"], color=colors, height=0.65, zorder=2)
    ax.axvline(0, color="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("+/- por 90 minutos", labelpad=6)
    ax.set_title(f"+/- por 90 min  (mín. {MIN_MINUTES} min)", fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="x", zorder=1)
    ax.tick_params(axis="y", labelsize=8.5)

    xmax = filt["pm_per90"].max()
    xmin = filt["pm_per90"].min()
    ax.set_xlim(xmin - 0.35, xmax + 0.35)

    for bar, val in zip(bars, filt["pm_per90"]):
        offset = 0.03 if val >= 0 else -0.03
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=9, color="white", fontweight="bold")


def plot_gf_ga_scatter(df, ax):
    filt = df[df["minutos"] >= MIN_MINUTES].copy()
    sizes = (filt["minutos"] / filt["minutos"].max() * 350).clip(lower=70)
    colors = bar_colors(filt["plus_minus"])
    ax.scatter(filt["gf_on"], filt["ga_on"], s=sizes, c=colors,
               edgecolors="white", linewidths=0.5, zorder=3, alpha=0.9)

    lim = max(filt["gf_on"].max(), filt["ga_on"].max()) + 1.5
    ax.plot([0, lim], [0, lim], color="white", linewidth=0.8, linestyle="--", zorder=2, alpha=0.4)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)

    texts = []
    for _, row in filt.iterrows():
        apellido = row["player"].split()[-1]
        t = ax.text(row["gf_on"], row["ga_on"], apellido,
                    fontsize=8, color="white", alpha=0.95, zorder=4)
        texts.append(t)

    adjust_text(
        texts,
        x=filt["gf_on"].values,
        y=filt["ga_on"].values,
        ax=ax,
        expand=(1.4, 1.6),
        arrowprops=dict(arrowstyle="-", color="white", alpha=0.4, lw=0.6),
    )

    ax.set_xlabel("Goles a favor (Colo-Colo) con el jugador en cancha", labelpad=6)
    ax.set_ylabel("Goles en contra con el jugador en cancha", labelpad=6)
    ax.set_title("Goles a favor vs en contra  (tamaño = minutos)", fontsize=12, fontweight="bold", pad=8)
    ax.grid(zorder=1)

    pos_patch = mpatches.Patch(color=POS_COLOR, label="+/- positivo")
    neg_patch = mpatches.Patch(color=NEG_COLOR, label="+/- negativo")
    ax.legend(handles=[pos_patch, neg_patch], loc="upper left", fontsize=8,
              facecolor="#1a2635", edgecolor="#2e3f52")


def plot_pm_acumulado(detail, df, ax):
    top_players = df[df["minutos"] >= MIN_MINUTES].nlargest(6, "plus_minus")["player"].tolist()
    bottom_players = df[df["minutos"] >= MIN_MINUTES].nsmallest(3, "plus_minus")["player"].tolist()
    players_to_show = list(dict.fromkeys(top_players + bottom_players))

    cmap = plt.colormaps.get_cmap("tab10")
    colors_map = {p: cmap(i / max(len(players_to_show) - 1, 1)) for i, p in enumerate(players_to_show)}

    player_detail = detail[detail["player"].isin(players_to_show)].copy()

    lines = []
    for player in players_to_show:
        sub = player_detail[player_detail["player"] == player].sort_values("date").copy()
        sub["pm_cum"] = sub["plus_minus"].cumsum()
        x = list(range(len(sub)))
        y = sub["pm_cum"].tolist()
        line, = ax.plot(x, y, marker="o", markersize=4,
                        color=colors_map[player], linewidth=2)
        # Etiqueta al final de la línea
        if x:
            ax.text(x[-1] + 0.15, y[-1], player.split()[-1],
                    fontsize=8, color=colors_map[player], va="center")
        lines.append(line)

    ax.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Partido (orden cronológico)", labelpad=6)
    ax.set_ylabel("+/- acumulado", labelpad=6)
    ax.set_title("+/- Acumulado por partido (top/bottom jugadores)", fontsize=12, fontweight="bold", pad=8)
    ax.grid(zorder=1)

    # Ampliar eje x para que no se corten las etiquetas del final
    xmax = max(
        len(player_detail[player_detail["player"] == p]) for p in players_to_show
        if not player_detail[player_detail["player"] == p].empty
    )
    ax.set_xlim(-0.3, xmax + 1.5)


def main():
    df, detail = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle("Análisis +/- Colo-Colo · Primera División 2026",
                 fontsize=15, fontweight="bold", color="white", y=0.99)

    plot_pm_bruto(df, axes[0, 0])
    plot_pm_per90(df, axes[0, 1])
    plot_gf_ga_scatter(df, axes[1, 0])
    plot_pm_acumulado(detail, df, axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4, w_pad=3)

    out = REPORTS_DIR / "plus_minus_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Guardado en {out}")
    plt.show()


if __name__ == "__main__":
    main()
