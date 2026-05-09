"""
Análisis extendido de Colo-Colo:
  1. Duplas más efectivas (+/- compartido por par de jugadores)
  2. Impacto de sustituciones (score antes vs después)
  4. Distribución de goles por franja horaria
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[4]
RAW_DIR = BASE_DIR / "apps" / "liga_chilena" / "colo_colo_plus_minus" / "data" / "raw"
REPORTS_DIR = BASE_DIR / "apps" / "liga_chilena" / "colo_colo_plus_minus" / "reports"

TEAM_NAME = "Colo-Colo"
MATCH_DURATION_SECONDS = 5400  # 90 min

MIN_MIN_INDIVIDUAL = 360   # para aparecer en heatmap de duplas
MIN_MIN_SHARED = 90        # mínimo tiempo juntos para contar el par

# último bucket abierto para capturar tiempo extra
FRANJAS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 999)]
FRANJA_LABELS = ["0-15'", "15-30'", "30-45'", "45-60'", "60-75'", "75-90'+"]

POS_COLOR = "#2ecc71"
NEG_COLOR = "#e74c3c"
NEUTRAL_COLOR = "#f39c12"

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


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def get_time_sec(inc):
    """Tiempo en segundos. SofaScore omite timeSeconds en la liga chilena; usa 'time' (minutos) como fallback."""
    ts = inc.get("timeSeconds")
    if ts is not None:
        return int(ts)
    t = inc.get("time")
    if t is not None:
        return int(t) * 60
    return 0


def build_player_intervals(lineups, incidents, colo_is_home):
    side_key = "home" if colo_is_home else "away"
    starters = {}
    for entry in lineups.get(side_key, {}).get("players", []):
        p = entry.get("player", {})
        pid = p.get("id")
        name = p.get("name") or p.get("shortName", f"id_{pid}")
        if pid and not entry.get("substitute", True):
            starters[pid] = name

    subs = []
    for inc in (incidents.get("incidents") or []):
        if inc.get("incidentType") != "substitution":
            continue
        if inc.get("isHome", False) != colo_is_home:
            continue
        pi = inc.get("playerIn") or {}
        po = inc.get("playerOut") or {}
        subs.append({
            "t": min(get_time_sec(inc), MATCH_DURATION_SECONDS),
            "in_id": pi.get("id"),
            "in_name": pi.get("name") or pi.get("shortName", "?"),
            "out_id": po.get("id"),
        })
    subs.sort(key=lambda s: s["t"])

    info = {pid: {"name": name, "intervals": []} for pid, name in starters.items()}
    active = {pid: 0 for pid in starters}

    for sub in subs:
        t = sub["t"]
        if sub["out_id"] and sub["out_id"] in active:
            info[sub["out_id"]]["intervals"].append((active.pop(sub["out_id"]), t))
        if sub["in_id"]:
            active[sub["in_id"]] = t
            if sub["in_id"] not in info:
                info[sub["in_id"]] = {"name": sub["in_name"], "intervals": []}

    for pid, t0 in active.items():
        info[pid]["intervals"].append((t0, MATCH_DURATION_SECONDS))

    return info, subs


def extract_goals(incidents, colo_is_home):
    goals = []
    for inc in (incidents.get("incidents") or []):
        if inc.get("incidentType") != "goal":
            continue
        goals.append({
            "time_sec": min(get_time_sec(inc), MATCH_DURATION_SECONDS),
            "is_colo": inc.get("isHome", False) == colo_is_home,
        })
    return goals


def intervals_overlap(ia, ib):
    result = []
    for a0, a1 in ia:
        for b0, b1 in ib:
            s, e = max(a0, b0), min(a1, b1)
            if s < e:
                result.append((s, e))
    return result


def in_intervals(t, intervals):
    return any(s <= t < e for s, e in intervals)


def score_at(goals, t):
    colo = sum(1 for g in goals if g["time_sec"] <= t and g["is_colo"])
    rival = sum(1 for g in goals if g["time_sec"] <= t and not g["is_colo"])
    return colo, rival


# ── data loading ─────────────────────────────────────────────────────────────

def load_all_matches():
    matches = []
    for d in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        meta = load_json(d / "meta.json")
        lineups = load_json(d / "lineups.json")
        incidents = load_json(d / "incidents.json")
        if not meta or not lineups or not incidents:
            continue
        colo_is_home = meta["home_team"] == TEAM_NAME
        player_info, subs = build_player_intervals(lineups, incidents, colo_is_home)
        goals = extract_goals(incidents, colo_is_home)
        matches.append({
            "meta": meta,
            "colo_is_home": colo_is_home,
            "player_info": player_info,
            "subs": subs,
            "goals": goals,
        })
    return matches


# ── análisis 1: duplas ────────────────────────────────────────────────────────

def build_duplas(matches):
    pair_stats = {}
    name_map = {}

    for m in matches:
        pi = m["player_info"]
        goals = m["goals"]
        for pid, data in pi.items():
            name_map[pid] = data["name"]

        for pid1, pid2 in combinations(pi.keys(), 2):
            shared = intervals_overlap(pi[pid1]["intervals"], pi[pid2]["intervals"])
            min_shared = sum(e - s for s, e in shared) / 60
            if min_shared <= 0:
                continue
            gf = sum(1 for g in goals if g["is_colo"] and in_intervals(g["time_sec"], shared))
            ga = sum(1 for g in goals if not g["is_colo"] and in_intervals(g["time_sec"], shared))
            key = tuple(sorted([pid1, pid2]))
            if key not in pair_stats:
                pair_stats[key] = {"min": 0, "gf": 0, "ga": 0}
            pair_stats[key]["min"] += min_shared
            pair_stats[key]["gf"] += gf
            pair_stats[key]["ga"] += ga

    rows = []
    for (p1, p2), s in pair_stats.items():
        if s["min"] < MIN_MIN_SHARED:
            continue
        pm = s["gf"] - s["ga"]
        pm90 = round(pm / s["min"] * 90, 2) if s["min"] > 0 else 0
        rows.append({
            "pid1": p1, "pid2": p2,
            "name1": name_map.get(p1, str(p1)),
            "name2": name_map.get(p2, str(p2)),
            "min": round(s["min"], 1),
            "gf": s["gf"], "ga": s["ga"],
            "pm": pm, "pm90": pm90,
        })

    df = pd.DataFrame(rows)

    # Filtrar a jugadores con >= MIN_MIN_INDIVIDUAL de minutos totales (del análisis principal)
    pm_df = pd.read_csv(REPORTS_DIR / "plus_minus_colo_colo.csv")
    eligible_names = set(pm_df[pm_df["minutos"] >= MIN_MIN_INDIVIDUAL]["player"])
    df = df[df["name1"].isin(eligible_names) & df["name2"].isin(eligible_names)]
    return df


def plot_duplas_heatmap(df_duplas, ax):
    players = sorted(set(df_duplas["name1"]) | set(df_duplas["name2"]))
    apellidos = [p.split()[-1] for p in players]
    n = len(players)
    matrix = np.full((n, n), np.nan)
    idx = {p: i for i, p in enumerate(players)}

    for _, row in df_duplas.iterrows():
        i, j = idx[row["name1"]], idx[row["name2"]]
        matrix[i, j] = row["pm90"]
        matrix[j, i] = row["pm90"]

    # Colormap divergente centrado en 0
    vmax = np.nanmax(np.abs(matrix)) or 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdYlGn

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(apellidos, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(apellidos, fontsize=7.5)

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "black" if abs(val) < vmax * 0.5 else "white"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=6.5, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("+/- por 90 min compartidos", fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors="white")
    cbar.outline.set_edgecolor("#2e3f52")

    ax.set_title(f"Duplas más efectivas · +/- por 90 min compartidos\n(mín. {MIN_MIN_SHARED} min juntos, jugadores con ≥{MIN_MIN_INDIVIDUAL} min)", fontsize=10, fontweight="bold", pad=8)


# ── análisis 2: sustituciones ─────────────────────────────────────────────────

def build_subs_impact(matches):
    rows = []
    for m in matches:
        goals = m["goals"]
        meta = m["meta"]
        colo_is_home = m["colo_is_home"]

        colo_final = meta["home_score"] if colo_is_home else meta["away_score"]
        rival_final = meta["away_score"] if colo_is_home else meta["home_score"]
        if colo_final is None or rival_final is None:
            continue
        diff_final = colo_final - rival_final

        for sub in m["subs"]:
            t = sub["t"]
            c_at, r_at = score_at(goals, t)
            diff_at = c_at - r_at
            impact = diff_final - diff_at

            situation = "Ganando" if diff_at > 0 else ("Perdiendo" if diff_at < 0 else "Empatando")
            rows.append({
                "player_in": sub["in_name"],
                "minute": round(t / 60, 0),
                "diff_at_sub": diff_at,
                "diff_final": diff_final,
                "impact": impact,
                "situation": situation,
            })
    return pd.DataFrame(rows)


def plot_subs_impact(df_subs, ax):
    if df_subs.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        return

    agg = (
        df_subs.groupby("player_in")
        .agg(avg_impact=("impact", "mean"), n=("impact", "count"))
        .reset_index()
        .sort_values("avg_impact", ascending=True)
    )

    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in agg["avg_impact"]]
    bars = ax.barh(agg["player_in"], agg["avg_impact"], color=colors, height=0.6, zorder=2)
    ax.axvline(0, color="white", linewidth=0.8, zorder=3)
    ax.grid(axis="x", zorder=1)

    xmax = agg["avg_impact"].abs().max()
    ax.set_xlim(-xmax - 0.8, xmax + 0.8)

    for bar, (_, row) in zip(bars, agg.iterrows()):
        v = row["avg_impact"]
        offset = 0.08 if v >= 0 else -0.08
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}  (n={int(row['n'])})",
                va="center", ha=ha, fontsize=8.5, color="white", fontweight="bold")

    ax.set_xlabel("Impacto promedio en el marcador  (dif. final − dif. al ingresar)")
    ax.set_title("Impacto de sustituciones · ¿El equipo mejoró o empeoró tras el ingreso?",
                 fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="y", labelsize=9)


# ── análisis 4: franja horaria ────────────────────────────────────────────────

def build_franjas(matches):
    counts = {label: {"gf": 0, "ga": 0} for label in FRANJA_LABELS}
    for m in matches:
        for g in m["goals"]:
            t_min = g["time_sec"] / 60.0
            for (f_start, f_end), label in zip(FRANJAS, FRANJA_LABELS):
                if f_start <= t_min < f_end:
                    key = "gf" if g["is_colo"] else "ga"
                    counts[label][key] += 1
                    break
    return counts


def plot_franjas(counts, ax):
    labels = FRANJA_LABELS
    gf = [counts[l]["gf"] for l in labels]
    ga = [counts[l]["ga"] for l in labels]
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w / 2, gf, w, label="Goles a favor", color=POS_COLOR, zorder=2)
    ax.bar(x + w / 2, ga, w, label="Goles en contra", color=NEG_COLOR, zorder=2)

    for xi, v in zip(x - w / 2, gf):
        if v:
            ax.text(xi, v + 0.05, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold", color=POS_COLOR)
    for xi, v in zip(x + w / 2, ga):
        if v:
            ax.text(xi, v + 0.05, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold", color=NEG_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Goles (total 11 partidos)")
    ax.set_ylim(0, max(max(gf), max(ga)) + 1.5)
    ax.set_title("Distribución de goles por franja horaria",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9, facecolor="#1a2635", edgecolor="#2e3f52")
    ax.grid(axis="y", zorder=1)

    # Línea de diferencia
    diff = [g - a for g, a in zip(gf, ga)]
    ax2 = ax.twinx()
    ax2.plot(x, diff, color="white", marker="D", markersize=6, linewidth=1.5,
             linestyle="--", alpha=0.7, label="Diferencia (GF−GA)")
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("GF − GA", color="white")
    ax2.tick_params(axis="y", colors="white")
    ax2.set_facecolor("#1a2635")
    ax2.legend(loc="upper right", fontsize=8, facecolor="#1a2635", edgecolor="#2e3f52")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    matches = load_all_matches()
    print(f"Partidos cargados: {len(matches)}")

    df_duplas = build_duplas(matches)
    df_subs = build_subs_impact(matches)
    counts_franjas = build_franjas(matches)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("Análisis Extendido — Colo-Colo · Primera División 2026",
                 fontsize=15, fontweight="bold", color="white", y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.94, bottom=0.07)

    ax_duplas = fig.add_subplot(gs[0, 0])
    ax_franjas = fig.add_subplot(gs[0, 1])
    ax_subs = fig.add_subplot(gs[1, :])

    plot_duplas_heatmap(df_duplas, ax_duplas)
    plot_franjas(counts_franjas, ax_franjas)
    plot_subs_impact(df_subs, ax_subs)

    out = REPORTS_DIR / "extended_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Guardado en {out}")
    plt.show()


if __name__ == "__main__":
    main()
