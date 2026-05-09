"""
Análisis de combinaciones por línea de Colo-Colo.

Para cada segmento de partido (entre sustituciones) identifica qué jugadores
de cada línea (Arquero / Defensa / Mediocampo / Delantera) estaban en cancha
y acumula los goles a favor/en contra de esa combinación exacta.

Métricas por combinación:
  - minutos   : minutos totales con esa combinación en cancha
  - gf / ga   : goles con esa combinación en cancha
  - pm        : gf - ga
  - pm_per90  : +/- normalizado por 90 min
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[4]
RAW_DIR  = BASE_DIR / "apps" / "liga_chilena" / "colo_colo_plus_minus" / "data" / "raw"
REPORTS_DIR = BASE_DIR / "apps" / "liga_chilena" / "colo_colo_plus_minus" / "reports"

TEAM_NAME = "Colo-Colo"
MATCH_END = 90 * 60   # segundos

MIN_MINUTES = 45      # mínimo de minutos compartidos para mostrar una combo
MAX_COMBOS  = 10      # máximo de combinaciones a mostrar por línea

POS_LABEL = {"G": "Arquero", "D": "Defensa", "M": "Mediocampo", "F": "Delantera"}
POS_ORDER  = ["F", "M", "D", "G"]

# Correcciones manuales de posición (player_id → posición canónica).
# Dejar vacío para usar la posición modal derivada de SofaScore.
POSITION_OVERRIDE: dict[int, str] = {
    # Ejemplo: 123456: "M"
}

POS_COLOR = "#2ecc71"
NEG_COLOR = "#e74c3c"

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


def build_canonical_positions():
    """
    Recorre todos los lineups y devuelve {player_id: posicion_modal}.
    Aplica POSITION_OVERRIDE sobre el resultado.
    """
    from collections import defaultdict, Counter
    counts: dict[int, Counter] = defaultdict(Counter)

    for d in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        meta    = load_json(d / "meta.json")
        lineups = load_json(d / "lineups.json")
        if not meta or not lineups:
            continue
        side = "home" if meta["home_team"] == TEAM_NAME else "away"
        for entry in lineups.get(side, {}).get("players", []):
            p   = entry.get("player", {})
            pid = p.get("id")
            pos = entry.get("position") or p.get("position")
            if pid and pos:
                counts[pid][pos] += 1

    canonical = {pid: ctr.most_common(1)[0][0] for pid, ctr in counts.items()}
    canonical.update(POSITION_OVERRIDE)

    # Imprimir resumen para verificar
    print(f"\n{'Jugador (id)':<35} {'Pos canonica':<14} {'Detalle'}")
    print("-" * 70)
    # Necesitamos nombres — los recuperamos del primer lineup que aparezca
    pid_names: dict[int, str] = {}
    for d in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        meta    = load_json(d / "meta.json")
        lineups = load_json(d / "lineups.json")
        if not meta or not lineups:
            continue
        side = "home" if meta["home_team"] == TEAM_NAME else "away"
        for entry in lineups.get(side, {}).get("players", []):
            p   = entry.get("player", {})
            pid = p.get("id")
            if pid and pid not in pid_names:
                pid_names[pid] = p.get("name") or p.get("shortName", f"id_{pid}")

    for pid, pos in sorted(canonical.items(), key=lambda x: pid_names.get(x[0], "")):
        det = "  |  ".join(f"{p}:{n}" for p, n in counts[pid].most_common())
        override = " [OVERRIDE]" if pid in POSITION_OVERRIDE else ""
        print(f"{pid_names.get(pid, str(pid)):<35} {pos:<14}{det}{override}")
    print()

    return canonical


def get_time_sec(inc):
    ts = inc.get("timeSeconds")
    if ts is not None:
        return int(ts)
    t = inc.get("time")
    return int(t) * 60 if t is not None else 0


def in_intervals(t, intervals):
    return any(s <= t < e for s, e in intervals)


# ── construcción de intervalos con posición ───────────────────────────────────

def build_intervals(lineups, incidents, colo_is_home, canonical_pos: dict):
    """
    Devuelve {pid: {name, position, intervals}}.
    Incluye solo jugadores de Colo-Colo.
    Usa canonical_pos para asignar la posición consistente a cada jugador.
    """
    side = "home" if colo_is_home else "away"

    info = {}
    for entry in lineups.get(side, {}).get("players", []):
        p   = entry.get("player", {})
        pid = p.get("id")
        if pid is None:
            continue
        name = p.get("name") or p.get("shortName", f"id_{pid}")
        pos  = canonical_pos.get(pid) or entry.get("position") or p.get("position", "?")
        if not entry.get("substitute", True):         # titular
            info[pid] = {"name": name, "position": pos, "intervals": []}

    active = {pid: 0 for pid in info}

    subs = []
    for inc in incidents.get("incidents") or []:
        if inc.get("incidentType") != "substitution":
            continue
        if inc.get("isHome", False) != colo_is_home:
            continue
        pi = inc.get("playerIn")  or {}
        po = inc.get("playerOut") or {}
        in_id = pi.get("id")
        subs.append({
            "t":        min(get_time_sec(inc), MATCH_END),
            "in_id":    in_id,
            "in_name":  pi.get("name") or pi.get("shortName", "?"),
            "in_pos":   canonical_pos.get(in_id) or pi.get("position", "?"),
            "out_id":   po.get("id"),
        })
    subs.sort(key=lambda s: s["t"])

    for sub in subs:
        t = sub["t"]
        if sub["out_id"] and sub["out_id"] in active:
            info[sub["out_id"]]["intervals"].append((active.pop(sub["out_id"]), t))
        if sub["in_id"]:
            active[sub["in_id"]] = t
            if sub["in_id"] not in info:
                info[sub["in_id"]] = {
                    "name": sub["in_name"],
                    "position": sub["in_pos"],
                    "intervals": [],
                }

    for pid, t0 in active.items():
        info[pid]["intervals"].append((t0, MATCH_END))

    return info


def extract_goals(incidents, colo_is_home):
    goals = []
    for inc in incidents.get("incidents") or []:
        if inc.get("incidentType") != "goal":
            continue
        goals.append({
            "t":      min(get_time_sec(inc), MATCH_END),
            "is_colo": inc.get("isHome", False) == colo_is_home,
        })
    return goals


# ── análisis de combinaciones ─────────────────────────────────────────────────

def combo_key(names):
    """Apellidos ordenados, unidos con ' · '."""
    return " · ".join(sorted(n.split()[-1] for n in names))


def analyze_match(info, goals):
    """
    Segmenta el partido por tiempos de sustitución.
    Para cada segmento acumula minutos/GF/GA por (posición, combo).
    """
    # Breakpoints: inicio, fin y cada cambio de plantel
    bp = {0, MATCH_END}
    for data in info.values():
        for s, e in data["intervals"]:
            if 0 < s < MATCH_END:
                bp.add(s)
            if 0 < e < MATCH_END:
                bp.add(e)
    segments = sorted(bp)

    stats = {}   # (pos, combo_str) -> {min, gf, ga}

    for i in range(len(segments) - 1):
        t0, t1 = segments[i], segments[i + 1]
        if t1 <= t0:
            continue
        t_mid = (t0 + t1) / 2
        dur = (t1 - t0) / 60.0

        # Jugadores en cancha en este segmento, agrupados por posición
        by_pos = {}
        for pid, data in info.items():
            if in_intervals(t_mid, data["intervals"]):
                pos = data["position"]
                by_pos.setdefault(pos, []).append(data["name"])

        gf = sum(1 for g in goals if t0 <= g["t"] < t1 and     g["is_colo"])
        ga = sum(1 for g in goals if t0 <= g["t"] < t1 and not g["is_colo"])

        for pos, names in by_pos.items():
            if pos not in POS_LABEL:
                continue
            key = (pos, combo_key(names))
            if key not in stats:
                stats[key] = {"min": 0.0, "gf": 0, "ga": 0}
            stats[key]["min"] += dur
            stats[key]["gf"]  += gf
            stats[key]["ga"]  += ga

    return stats


def build_all_combos():
    canonical_pos = build_canonical_positions()
    all_stats = {}

    for d in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        meta      = load_json(d / "meta.json")
        lineups   = load_json(d / "lineups.json")
        incidents = load_json(d / "incidents.json")
        if not meta or not lineups or not incidents:
            continue

        colo_is_home = meta["home_team"] == TEAM_NAME
        info  = build_intervals(lineups, incidents, colo_is_home, canonical_pos)
        goals = extract_goals(incidents, colo_is_home)

        for key, vals in analyze_match(info, goals).items():
            if key not in all_stats:
                all_stats[key] = {"min": 0.0, "gf": 0, "ga": 0}
            all_stats[key]["min"] += vals["min"]
            all_stats[key]["gf"]  += vals["gf"]
            all_stats[key]["ga"]  += vals["ga"]

    rows = []
    for (pos, combo), s in all_stats.items():
        if s["min"] < MIN_MINUTES:
            continue
        pm = s["gf"] - s["ga"]
        rows.append({
            "pos":     pos,
            "combo":   combo,
            "min":     round(s["min"], 1),
            "gf":      s["gf"],
            "ga":      s["ga"],
            "pm":      pm,
            "pm_per90": round(pm / s["min"] * 90, 2),
        })

    return pd.DataFrame(rows)


# ── visualización ─────────────────────────────────────────────────────────────

def plot_pos(ax, df_pos, pos):
    label = POS_LABEL.get(pos, pos)

    # Top N por minutos, luego ordenar por pm_per90
    top = (df_pos.nlargest(MAX_COMBOS, "min")
                 .sort_values("pm_per90", ascending=True))

    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in top["pm_per90"]]
    bars = ax.barh(top["combo"], top["pm_per90"], color=colors, height=0.6, zorder=2)

    ax.axvline(0, color="white", linewidth=0.8, zorder=3)
    ax.grid(axis="x", zorder=1)

    xmax = top["pm_per90"].abs().max() if not top.empty else 1
    ax.set_xlim(-xmax - 0.9, xmax + 0.9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(label, fontsize=12, fontweight="bold", pad=7, color="white")
    ax.set_xlabel("+/- por 90 min", fontsize=8, labelpad=4)

    for bar, (_, row) in zip(bars, top.iterrows()):
        v  = row["pm_per90"]
        of = 0.06 if v >= 0 else -0.06
        ha = "left" if v >= 0 else "right"
        label_txt = f"{v:+.2f}  ({int(row['min'])}′)"
        ax.text(v + of, bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", ha=ha, fontsize=8,
                color="white", fontweight="bold")


def main():
    print("Calculando combinaciones…")
    df = build_all_combos()
    if df.empty:
        print("Sin datos suficientes.")
        return

    # Guardar CSV
    df.sort_values(["pos", "pm_per90"], ascending=[True, False]).to_csv(
        REPORTS_DIR / "lineup_combos.csv", index=False
    )
    print(f"Guardado en {REPORTS_DIR / 'lineup_combos.csv'}")

    # Figura: 2×2, orden F / M / D / G
    fig = plt.figure(figsize=(22, 17))
    fig.suptitle("Combinaciones por línea — Colo-Colo · Primera División 2026",
                 fontsize=14, fontweight="bold", color="white", y=0.99)

    # Calcular alturas proporcionales al nº de combos
    heights = []
    for pos in POS_ORDER:
        n = min(len(df[df["pos"] == pos]), MAX_COMBOS)
        heights.append(max(n, 1))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[max(heights[0], heights[1]),
                                          max(heights[2], heights[3])],
                           hspace=0.5, wspace=0.45,
                           left=0.13, right=0.97, top=0.95, bottom=0.05)

    axes = [
        fig.add_subplot(gs[0, 0]),  # F
        fig.add_subplot(gs[0, 1]),  # M
        fig.add_subplot(gs[1, 0]),  # D
        fig.add_subplot(gs[1, 1]),  # G
    ]

    for ax, pos in zip(axes, POS_ORDER):
        sub = df[df["pos"] == pos]
        if sub.empty:
            ax.set_visible(False)
            continue
        plot_pos(ax, sub, pos)

    out = REPORTS_DIR / "lineup_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figura guardada en {out}")
    plt.show()


if __name__ == "__main__":
    main()
