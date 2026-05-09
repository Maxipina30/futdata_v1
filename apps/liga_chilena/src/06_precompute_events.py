"""
Precomputa datos de eventos para el dashboard:
  1. goals_by_minute.csv  — goles por franja de tiempo (por equipo y rival)
  2. substitution_impact.csv — jugadores que entran: +/- del equipo mientras estuvieron en cancha

Uso:
    venv312/Scripts/python.exe apps\\liga_chilena\\src\06_precompute_events.py
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR      = Path(__file__).resolve().parents[3]
RAW_DIR       = BASE_DIR / "apps" / "liga_chilena" / "data" / "raw"
REPORTS_DIR   = BASE_DIR / "apps" / "liga_chilena" / "reports"

MATCH_END = 90 * 60
FRANJAS   = [(0,15,"0-15'"), (15,30,"16-30'"), (30,45,"31-45'"),
             (45,60,"46-60'"), (60,75,"61-75'"), (75,999,"76-90'+")]


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_time_sec(inc):
    ts = inc.get("timeSeconds")
    if ts is not None:
        return int(ts)
    t = inc.get("time")
    return int(t) * 60 if t is not None else 0


def franja(t_sec):
    m = t_sec / 60
    for lo, hi, label in FRANJAS:
        if lo <= m < hi:
            return label
    return "76-90'+"


# ── 1. Goles por minuto ───────────────────────────────────────────────────────

def extract_goals_by_minute(meta, incidents_data):
    rows = []
    for inc in (incidents_data.get("incidents") or []):
        if inc.get("incidentType") != "goal":
            continue
        t   = min(get_time_sec(inc), MATCH_END)
        bucket = franja(t)
        is_home_goal = inc.get("isHome", False)

        # Desde la perspectiva del equipo que anota
        scoring_team  = meta["home_team"] if is_home_goal else meta["away_team"]
        conceding_team = meta["away_team"] if is_home_goal else meta["home_team"]

        rows.append({
            "match_id":      meta["event_id"],
            "date":          meta.get("start_timestamp"),
            "scoring_team":  scoring_team,
            "conceding_team": conceding_team,
            "franja":        bucket,
            "minute":        t // 60,
        })
    return rows


# ── 2. Impacto de sustituciones ───────────────────────────────────────────────

def extract_sub_impact(meta, incidents_data):
    incidents = incidents_data.get("incidents") or []
    incidents_sorted = sorted(incidents, key=lambda i: get_time_sec(i))

    rows = []
    for is_home in (True, False):
        team = meta["home_team"] if is_home else meta["away_team"]

        final_gf = (meta.get("home_score") if is_home else meta.get("away_score")) or 0
        final_ga = (meta.get("away_score") if is_home else meta.get("home_score")) or 0
        final_pm = final_gf - final_ga

        running_gf = 0
        running_ga = 0

        for inc in incidents_sorted:
            t = min(get_time_sec(inc), MATCH_END)

            if inc.get("incidentType") == "goal":
                if inc.get("isHome", False) == is_home:
                    running_gf += 1
                else:
                    running_ga += 1

            elif inc.get("incidentType") == "substitution":
                if inc.get("isHome", False) != is_home:
                    continue
                pi = inc.get("playerIn")  or {}
                po = inc.get("playerOut") or {}
                pm_at_entry = running_gf - running_ga
                pm_after    = final_pm - pm_at_entry

                rows.append({
                    "match_id":     meta["event_id"],
                    "date":         meta.get("start_timestamp"),
                    "team":         team,
                    "player_in_id": pi.get("id"),
                    "player_in":    pi.get("name") or pi.get("shortName", "?"),
                    "player_out":   po.get("name") or po.get("shortName", "?"),
                    "minute":       t // 60,
                    "pm_at_entry":  pm_at_entry,
                    "pm_after":     pm_after,
                })
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    goal_rows = []
    sub_rows  = []

    match_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir()) if RAW_DIR.exists() else []
    print(f"Partidos: {len(match_dirs)}")

    for d in match_dirs:
        meta      = load_json(d / "meta.json")
        incidents = load_json(d / "incidents.json")
        if not meta or not incidents:
            continue
        goal_rows.extend(extract_goals_by_minute(meta, incidents))
        sub_rows.extend(extract_sub_impact(meta, incidents))

    # Goles por minuto
    gdf = pd.DataFrame(goal_rows)
    if not gdf.empty:
        gdf["date"] = pd.to_datetime(gdf["date"], unit="s").dt.date
    gdf.to_csv(REPORTS_DIR / "goals_by_minute.csv", index=False)
    print(f"Goles registrados: {len(gdf)}")

    # Impacto de sustituciones
    sdf = pd.DataFrame(sub_rows)
    if not sdf.empty:
        sdf["date"] = pd.to_datetime(sdf["date"], unit="s").dt.date
    sdf.to_csv(REPORTS_DIR / "substitution_impact.csv", index=False)
    print(f"Sustituciones registradas: {len(sdf)}")

    # Resumen impacto por jugador
    if not sdf.empty:
        agg = (
            sdf.groupby(["player_in_id", "player_in", "team"])
            .agg(entradas=("match_id", "count"),
                 pm_after_avg=("pm_after", "mean"),
                 pm_after_sum=("pm_after", "sum"))
            .reset_index()
            .sort_values("pm_after_avg", ascending=False)
        )
        agg["pm_after_avg"] = agg["pm_after_avg"].round(2)
        agg.to_csv(REPORTS_DIR / "sub_impact_by_player.csv", index=False)
        print(f"\nTop 10 impacto al ingresar:")
        print(agg[["player_in", "team", "entradas", "pm_after_avg"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
