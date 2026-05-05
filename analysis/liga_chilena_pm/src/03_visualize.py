"""
Visualizaciones del +/- de la Primera Division de Chile 2026.

Graficos:
  1. GF vs GA scatter — solo etiqueta outliers (top/bottom por pm absoluto)
  2. GF vs GA scatter coloreado por equipo
  3. Top/Bottom +/- bruto (ranking horizontal)
  4. Top/Bottom +/- por 90 min (ranking horizontal)

Uso:
    venv312\Scripts\python.exe analysis\liga_chilena_pm\src\03_visualize.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

BASE_DIR      = Path(__file__).resolve().parents[3]
REPORTS_DIR   = BASE_DIR / "analysis" / "liga_chilena_pm" / "reports"
PROCESSED_DIR = BASE_DIR / "analysis" / "liga_chilena_pm" / "data" / "processed"

MIN_MINUTES  = 270   # ~3 partidos completos
TOP_N        = 20    # jugadores en el ranking (10 top + 10 bottom)
LABEL_TOP_N  = 12   # cuantos outliers etiquetar en el scatter

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.facecolor": "#0f1923",
    "axes.facecolor":   "#1a2635",
    "axes.edgecolor":   "#2e3f52",
    "axes.labelcolor":  "white",
    "text.color":       "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.color":       "#2e3f52",
    "grid.linewidth":   0.6,
})

POS_COLOR = "#2ecc71"
NEG_COLOR = "#e74c3c"

TEAM_SHORT = {
    "Universidad de Chile":       "U. Chile",
    "Universidad Católica":       "U. Católica",
    "Universidad de Concepción":  "U. Concep.",
    "Deportes Concepción":        "D. Concep.",
    "Deportes La Serena":         "La Serena",
    "Deportes Limache":           "Limache",
    "Everton de Viña del Mar":    "Everton",
    "Unión La Calera":            "La Calera",
    "Coquimbo Unido":             "Coquimbo",
    "Audax Italiano":             "Audax",
    "Cobresal":                   "Cobresal",
    "Huachipato":                 "Huachipato",
    "Ñublense":                   "Ñublense",
    "O'Higgins":                  "O'Higgins",
    "Palestino":                  "Palestino",
    "Colo-Colo":                  "Colo-Colo",
}


def short_team(name):
    return TEAM_SHORT.get(name, name.split()[-1])


def bar_colors(values):
    return [POS_COLOR if v >= 0 else NEG_COLOR for v in values]


def load_data():
    df = pd.read_csv(REPORTS_DIR / "plus_minus_liga.csv")
    return df


# ── Scatter GF vs GA — solo etiqueta outliers ─────────────────────────────────

def plot_gf_ga_scatter(df, ax):
    filt   = df[df["minutos"] >= MIN_MINUTES].copy()
    sizes  = (filt["minutos"] / filt["minutos"].max() * 450).clip(lower=55)
    colors = bar_colors(filt["pm"])

    ax.scatter(filt["gf_on"], filt["ga_on"],
               s=sizes, c=colors,
               edgecolors="white", linewidths=0.45, zorder=3, alpha=0.85)

    lim = max(filt["gf_on"].max(), filt["ga_on"].max()) + 2.5
    ax.plot([0, lim], [0, lim], color="white", lw=0.8, ls="--", alpha=0.3, zorder=2)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)

    # Etiquetar solo outliers: top y bottom N por pm absoluto
    top_idx    = filt.nlargest(LABEL_TOP_N, "pm").index
    bottom_idx = filt.nsmallest(LABEL_TOP_N, "pm").index
    to_label   = filt.loc[top_idx.union(bottom_idx)]

    texts = []
    for _, row in to_label.iterrows():
        apellido = row["player"].split()[-1]
        equipo   = short_team(row["team"])
        label    = f"{apellido}\n{equipo}"
        t = ax.text(row["gf_on"], row["ga_on"], label,
                    fontsize=7, color="white", alpha=0.95, zorder=5,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#1a2635",
                              ec="none", alpha=0.6))
        texts.append(t)

    adjust_text(
        texts,
        x=filt["gf_on"].values,
        y=filt["ga_on"].values,
        ax=ax,
        expand=(1.6, 1.8),
        force_text=(0.4, 0.5),
        arrowprops=dict(arrowstyle="-", color="white", alpha=0.35, lw=0.55),
    )

    ax.set_xlabel("Goles a favor del equipo con el jugador en cancha", labelpad=6)
    ax.set_ylabel("Goles en contra con el jugador en cancha", labelpad=6)
    ax.set_title(
        f"GF vs GA por jugador  (tamaño = minutos, mín. {MIN_MINUTES} min)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.grid(zorder=1)

    pos_patch = mpatches.Patch(color=POS_COLOR, label="+/- positivo")
    neg_patch = mpatches.Patch(color=NEG_COLOR, label="+/- negativo")
    ax.legend(handles=[pos_patch, neg_patch], loc="upper left",
              fontsize=8.5, facecolor="#1a2635", edgecolor="#2e3f52")


# ── Scatter coloreado por equipo ──────────────────────────────────────────────

def plot_gf_ga_by_team(df, ax):
    filt    = df[df["minutos"] >= MIN_MINUTES].copy()
    teams   = sorted(filt["team"].unique())
    cmap    = plt.colormaps.get_cmap("tab20")
    t_color = {t: cmap(i / max(len(teams) - 1, 1)) for i, t in enumerate(teams)}

    for team, grp in filt.groupby("team"):
        sizes = (grp["minutos"] / filt["minutos"].max() * 450).clip(lower=55)
        ax.scatter(grp["gf_on"], grp["ga_on"],
                   s=sizes, color=t_color[team],
                   edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85,
                   label=short_team(team))

    lim = max(filt["gf_on"].max(), filt["ga_on"].max()) + 2.5
    ax.plot([0, lim], [0, lim], color="white", lw=0.8, ls="--", alpha=0.3, zorder=2)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_xlabel("Goles a favor del equipo con el jugador en cancha", labelpad=6)
    ax.set_ylabel("Goles en contra con el jugador en cancha", labelpad=6)
    ax.set_title("GF vs GA por equipo", fontsize=11, fontweight="bold", pad=8)
    ax.grid(zorder=1)
    ax.legend(fontsize=7, ncol=2,
              facecolor="#1a2635", edgecolor="#2e3f52",
              loc="upper left", markerscale=0.8,
              handlelength=1.2, handletextpad=0.5, borderpad=0.6)


# ── Ranking horizontal ────────────────────────────────────────────────────────

def plot_ranking(df, ax, metric="pm_per90", label="+/- por 90 min"):
    filt   = df[df["minutos"] >= MIN_MINUTES].copy()
    n_each = TOP_N // 2

    top    = filt.nlargest(n_each,  metric)
    bottom = filt.nsmallest(n_each, metric)
    sub    = pd.concat([bottom, top]).drop_duplicates().sort_values(metric, ascending=True)

    # Etiqueta: "Apellido · Equipo"
    ylabels = [
        f"{row['player'].split()[-1]}  ·  {short_team(row['team'])}"
        for _, row in sub.iterrows()
    ]

    colors = bar_colors(sub[metric])
    bars   = ax.barh(ylabels, sub[metric], color=colors, height=0.65, zorder=2)

    ax.axvline(0, color="white", lw=0.8, zorder=3)
    ax.grid(axis="x", zorder=1)
    ax.set_xlabel(label, labelpad=6)
    ax.set_title(
        f"Top/Bottom {n_each} — {label}  (mín. {MIN_MINUTES} min)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.tick_params(axis="y", labelsize=8)

    xmax = sub[metric].abs().max() if not sub.empty else 1
    margin = xmax * 0.18
    ax.set_xlim(-xmax - margin, xmax + margin)

    for bar, val in zip(bars, sub[metric]):
        of = xmax * 0.02 if val >= 0 else -xmax * 0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + of, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=8,
                color="white", fontweight="bold")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    fig = plt.figure(figsize=(26, 20))
    fig.suptitle(
        "Análisis +/- · Primera División de Chile 2026",
        fontsize=15, fontweight="bold", color="white", y=0.997,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.42,
        left=0.14, right=0.97, top=0.96, bottom=0.05,
    )

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_by_team = fig.add_subplot(gs[0, 1])
    ax_rank_pm = fig.add_subplot(gs[1, 0])
    ax_rank_90 = fig.add_subplot(gs[1, 1])

    plot_gf_ga_scatter(df, ax_scatter)
    plot_gf_ga_by_team(df, ax_by_team)
    plot_ranking(df, ax_rank_pm, metric="pm",       label="+/- bruto")
    plot_ranking(df, ax_rank_90, metric="pm_per90", label="+/- por 90 min")

    out = REPORTS_DIR / "plus_minus_liga.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Guardado en {out}")
    plt.show()


if __name__ == "__main__":
    main()
