"""
Visualizaciones ofensivas — Primera Division de Chile 2026.

1. Goles vs Asistencias  (estilo coordenadas, tamaño = minutos)
2. Goles vs Tiros        (conversion rate, linea diagonal = promedio liga)
3. Tiros al Arco vs Tiros (precision, linea diagonal = promedio liga)

Uso:
    venv312/Scripts/python.exe apps\\liga_chilena\\src\05_visualize_stats.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

BASE_DIR    = Path(__file__).resolve().parents[3]
REPORTS_DIR = BASE_DIR / "apps" / "liga_chilena" / "reports"

MIN_MIN_GA   = 270   # minutos minimos para scatter G+A
MIN_SHOTS    = 8     # tiros minimos para scatter de conversion y precision
LABEL_N      = 10   # outliers a etiquetar en cada scatter

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


def short(name, team):
    return f"{name.split()[-1]}\n{TEAM_SHORT.get(team, team.split()[-1])}"


def color_gradient(values, cmap_name="RdYlGn"):
    cmap  = plt.colormaps.get_cmap(cmap_name)
    vmin, vmax = values.min(), values.max()
    norm  = (values - vmin) / (vmax - vmin + 1e-9)
    return [cmap(v) for v in norm]


def add_labels(ax, df, xcol, ycol, label_col, n=LABEL_N, extra_x=None, extra_y=None):
    """Etiqueta los n mayores outliers por distancia a la diagonal promedio."""
    texts = []
    for _, row in df.iterrows():
        t = ax.text(
            row[xcol], row[ycol], row[label_col],
            fontsize=7, color="white", alpha=0.93, zorder=5,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#1a2635", ec="none", alpha=0.55),
        )
        texts.append(t)

    all_x = extra_x if extra_x is not None else df[xcol].values
    all_y = extra_y if extra_y is not None else df[ycol].values

    adjust_text(
        texts,
        x=all_x, y=all_y,
        ax=ax,
        expand=(1.55, 1.75),
        force_text=(0.35, 0.45),
        arrowprops=dict(arrowstyle="-", color="white", alpha=0.3, lw=0.5),
    )


def diag_line(ax, x_max, slope, label, color="white"):
    xs = np.array([0, x_max])
    ax.plot(xs, xs * slope, color=color, lw=0.9, ls="--", alpha=0.45, zorder=2)
    ax.text(
        x_max * 0.92, x_max * slope * 0.92, label,
        fontsize=7.5, color=color, alpha=0.65,
        rotation=np.degrees(np.arctan(slope)),
        ha="center", va="bottom",
    )


# ── 1. Goles vs Asistencias ───────────────────────────────────────────────────

def plot_goals_assists(df, ax):
    filt  = df[df["minutes"] >= MIN_MIN_GA].copy()
    filt["ga"] = filt["goals"] + filt["assists"]
    sizes = (filt["minutes"] / filt["minutes"].max() * 420).clip(lower=50)
    cols  = color_gradient(filt["ga_per90"], "RdYlGn")

    ax.scatter(filt["goals"], filt["assists"],
               s=sizes, c=cols,
               edgecolors="white", linewidths=0.45, zorder=3, alpha=0.87)

    lim = max(filt["goals"].max(), filt["assists"].max()) + 1.5
    ax.plot([0, lim], [0, lim], color="white", lw=0.8, ls="--", alpha=0.25, zorder=2)
    ax.set_xlim(-0.4, filt["goals"].max() + 1.8)
    ax.set_ylim(-0.4, filt["assists"].max() + 1.5)

    # Etiquetar outliers: top N por G+A total
    top = filt.nlargest(LABEL_N, "ga")
    top["lbl"] = top.apply(lambda r: short(r["player"], r["team"]), axis=1)
    add_labels(ax, top, "goals", "assists", "lbl",
               extra_x=filt["goals"].values, extra_y=filt["assists"].values)

    ax.set_xlabel("Goles", labelpad=6)
    ax.set_ylabel("Asistencias", labelpad=6)
    ax.set_title(
        f"Goles vs Asistencias  (tamaño = minutos, color = G+A/90, mín. {MIN_MIN_GA} min)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.grid(zorder=1)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                norm=plt.Normalize(filt["ga_per90"].min(),
                                                   filt["ga_per90"].max()))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("G+A por 90 min", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7.5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


# ── 2. Goles vs Tiros (conversion) ───────────────────────────────────────────

def plot_goals_shots(df, ax):
    filt = df[df["shots"] >= MIN_SHOTS].copy()
    sizes = (filt["minutes"] / filt["minutes"].max() * 420).clip(lower=50)
    cols  = color_gradient(filt["conv_rate"].fillna(0), "RdYlGn")

    ax.scatter(filt["shots"], filt["goals"],
               s=sizes, c=cols,
               edgecolors="white", linewidths=0.45, zorder=3, alpha=0.87)

    x_max = filt["shots"].max() + 3
    # Linea promedio liga
    avg_rate = filt["goals"].sum() / filt["shots"].sum()
    diag_line(ax, x_max, avg_rate, f"prom. {avg_rate:.1%}")

    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-0.3, filt["goals"].max() + 1.5)

    # Outliers: mayor distancia vertical a la linea diagonal
    filt["dist"] = filt["goals"] - filt["shots"] * avg_rate
    top    = filt.nlargest(LABEL_N // 2,  "dist")
    bottom = filt.nsmallest(LABEL_N // 2, "dist")
    to_lbl = pd.concat([top, bottom]).drop_duplicates()
    to_lbl["lbl"] = to_lbl.apply(lambda r: short(r["player"], r["team"]), axis=1)
    add_labels(ax, to_lbl, "shots", "goals", "lbl",
               extra_x=filt["shots"].values, extra_y=filt["goals"].values)

    ax.set_xlabel("Tiros totales", labelpad=6)
    ax.set_ylabel("Goles", labelpad=6)
    ax.set_title(
        f"Goles vs Tiros — tasa de conversión  (mín. {MIN_SHOTS} tiros, tamaño = minutos)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.grid(zorder=1)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                norm=plt.Normalize(0, filt["conv_rate"].max()))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("Conversión (goles/tiro)", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7.5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


# ── 3. Tiros al Arco vs Tiros (precision) ─────────────────────────────────────

def plot_sot_shots(df, ax):
    filt = df[df["shots"] >= MIN_SHOTS].copy()
    sizes = (filt["minutes"] / filt["minutes"].max() * 420).clip(lower=50)
    cols  = color_gradient(filt["sot_rate"].fillna(0), "RdYlGn")

    ax.scatter(filt["shots"], filt["sot"],
               s=sizes, c=cols,
               edgecolors="white", linewidths=0.45, zorder=3, alpha=0.87)

    x_max = filt["shots"].max() + 3
    # No puede haber más tiros al arco que tiros totales: linea limite = pendiente 1
    ax.plot([0, x_max], [0, x_max], color="white", lw=0.7, ls=":", alpha=0.3, zorder=2)
    # Linea promedio liga
    avg_sot = filt["sot"].sum() / filt["shots"].sum()
    diag_line(ax, x_max, avg_sot, f"prom. {avg_sot:.1%}")

    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-0.3, filt["sot"].max() + 2)

    # Outliers: mayor/menor precision relativa al promedio
    filt["dist"] = filt["sot"] - filt["shots"] * avg_sot
    top    = filt.nlargest(LABEL_N // 2,  "dist")
    bottom = filt.nsmallest(LABEL_N // 2, "dist")
    to_lbl = pd.concat([top, bottom]).drop_duplicates()
    to_lbl["lbl"] = to_lbl.apply(lambda r: short(r["player"], r["team"]), axis=1)
    add_labels(ax, to_lbl, "shots", "sot", "lbl",
               extra_x=filt["shots"].values, extra_y=filt["sot"].values)

    ax.set_xlabel("Tiros totales", labelpad=6)
    ax.set_ylabel("Tiros al arco", labelpad=6)
    ax.set_title(
        f"Tiros al Arco vs Tiros — precisión  (mín. {MIN_SHOTS} tiros, tamaño = minutos)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.grid(zorder=1)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                norm=plt.Normalize(0, filt["sot_rate"].max()))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("Precisión (tiros al arco / tiro)", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7.5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


# ── 4-5-6. Métrica vs Minutos ─────────────────────────────────────────────────

def plot_metric_vs_minutes(df, ax, ycol, rate_col, ylabel, title, min_min=MIN_MIN_GA, min_goals=1):
    """
    x = minutos jugados, y = metrica absoluta (goles / asist / G+A).
    Color = tasa por 90. Lineas de referencia por 90.
    """
    filt = df[(df["minutes"] >= min_min) & (df[ycol] >= min_goals)].copy()
    sizes = (filt["minutes"] / filt["minutes"].max() * 420).clip(lower=50)
    cols  = color_gradient(filt[rate_col].fillna(0), "RdYlGn")

    ax.scatter(filt["minutes"], filt[ycol],
               s=sizes, c=cols,
               edgecolors="white", linewidths=0.45, zorder=3, alpha=0.87)

    x_max = filt["minutes"].max() + 50
    # Líneas de referencia: 0.25, 0.5, 1.0 por 90 min
    for rate, ls in [(0.25, ":"), (0.5, "--"), (1.0, "-.")]:
        y_end = rate * x_max / 90
        if y_end > filt[ycol].max() * 1.5:
            continue
        ax.plot([0, x_max], [0, y_end], color="white", lw=0.7, ls=ls, alpha=0.25, zorder=2)
        ax.text(x_max * 0.97, y_end * 0.97, f"{rate}/90",
                fontsize=7, color="white", alpha=0.45, ha="right", va="top")

    ax.set_xlim(min_min * 0.85, x_max)
    ax.set_ylim(-0.3, filt[ycol].max() + 1.2)

    # Etiquetar outliers: mayor tasa y mayor volumen
    top_rate = filt.nlargest(LABEL_N // 2, rate_col)
    top_vol  = filt.nlargest(LABEL_N // 2, ycol)
    to_lbl   = pd.concat([top_rate, top_vol]).drop_duplicates()
    to_lbl["lbl"] = to_lbl.apply(lambda r: short(r["player"], r["team"]), axis=1)
    add_labels(ax, to_lbl, "minutes", ycol, "lbl",
               extra_x=filt["minutes"].values, extra_y=filt[ycol].values)

    ax.set_xlabel("Minutos jugados", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(zorder=1)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                norm=plt.Normalize(0, filt[rate_col].max()))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label(f"{ylabel}/90 min", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7.5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(REPORTS_DIR / "player_stats.csv")
    df["ga"]      = df["goals"] + df["assists"]
    df["g_per90"] = (df["goals"]   / df["minutes"] * 90).round(2)
    df["a_per90"] = (df["assists"] / df["minutes"] * 90).round(2)

    # ── Figura 1: Goles vs Asistencias + Conversión + Precisión ──────────────
    fig1 = plt.figure(figsize=(26, 22))
    fig1.suptitle(
        "Análisis ofensivo — Primera División de Chile 2026",
        fontsize=15, fontweight="bold", color="white", y=0.997,
    )
    gs1 = gridspec.GridSpec(2, 2, figure=fig1,
                            hspace=0.38, wspace=0.38,
                            left=0.07, right=0.95, top=0.96, bottom=0.05)

    ax1 = fig1.add_subplot(gs1[0, :])
    ax2 = fig1.add_subplot(gs1[1, 0])
    ax3 = fig1.add_subplot(gs1[1, 1])

    plot_goals_assists(df, ax1)
    plot_goals_shots(df, ax2)
    plot_sot_shots(df, ax3)

    out1 = REPORTS_DIR / "offensive_stats.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
    print(f"Guardado en {out1}")

    # ── Figura 2: Goles/Asist/G+A vs Minutos ─────────────────────────────────
    fig2 = plt.figure(figsize=(26, 26))
    fig2.suptitle(
        "Producción por minutos — Primera División de Chile 2026",
        fontsize=15, fontweight="bold", color="white", y=0.997,
    )
    gs2 = gridspec.GridSpec(3, 1, figure=fig2,
                            hspace=0.45,
                            left=0.07, right=0.95, top=0.96, bottom=0.04)

    ax4 = fig2.add_subplot(gs2[0])
    ax5 = fig2.add_subplot(gs2[1])
    ax6 = fig2.add_subplot(gs2[2])

    plot_metric_vs_minutes(df, ax4, "goals",   "g_per90",  "Goles",
                           f"Goles vs Minutos  (color = goles/90, mín. {MIN_MIN_GA} min, mín. 1 gol)")
    plot_metric_vs_minutes(df, ax5, "assists",  "a_per90",  "Asistencias",
                           f"Asistencias vs Minutos  (color = asist./90, mín. {MIN_MIN_GA} min, mín. 1 asist.)")
    plot_metric_vs_minutes(df, ax6, "ga",       "ga_per90", "Goles + Asistencias",
                           f"G+A vs Minutos  (color = G+A/90, mín. {MIN_MIN_GA} min, mín. 1 G+A)")

    out2 = REPORTS_DIR / "production_vs_minutes.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"Guardado en {out2}")

    plt.show()


if __name__ == "__main__":
    main()
