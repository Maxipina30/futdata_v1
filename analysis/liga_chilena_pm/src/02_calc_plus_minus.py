"""
Calcula el +/- de TODOS los jugadores de la Primera Division de Chile 2026.

Para cada partido procesa las dos plantillas (home y away).
La referencia del +/- es siempre desde la perspectiva del equipo del jugador:
  gf_on  = goles de SU equipo con el jugador en cancha
  ga_on  = goles del RIVAL con el jugador en cancha
  pm     = gf_on - ga_on

Uso:
    venv312\Scripts\python.exe analysis\liga_chilena_pm\src\02_calc_plus_minus.py
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR      = Path(__file__).resolve().parents[3]
RAW_DIR       = BASE_DIR / "analysis" / "liga_chilena_pm" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "analysis" / "liga_chilena_pm" / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "analysis" / "liga_chilena_pm" / "reports"

MATCH_END = 90 * 60  # 5400 segundos


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


def build_player_intervals(lineups, incidents, is_home: bool):
    """
    Devuelve {pid: {name, team, intervals}} para UNA de las dos plantillas.
    is_home=True  → procesa el equipo local
    is_home=False → procesa el equipo visitante
    """
    side = "home" if is_home else "away"
    lineup_side = lineups.get(side, {})

    starters = {}
    for entry in lineup_side.get("players", []):
        p   = entry.get("player", {})
        pid = p.get("id")
        if pid is None:
            continue
        name = p.get("name") or p.get("shortName", f"id_{pid}")
        if not entry.get("substitute", True):
            starters[pid] = name

    subs = []
    for inc in (incidents.get("incidents") or []):
        if inc.get("incidentType") != "substitution":
            continue
        if inc.get("isHome", False) != is_home:
            continue
        pi = inc.get("playerIn")  or {}
        po = inc.get("playerOut") or {}
        subs.append({
            "t":       min(get_time_sec(inc), MATCH_END),
            "in_id":   pi.get("id"),
            "in_name": pi.get("name") or pi.get("shortName", "?"),
            "out_id":  po.get("id"),
        })
    subs.sort(key=lambda s: s["t"])

    active      = {pid: 0 for pid in starters}
    player_info = {pid: {"name": name, "intervals": []} for pid, name in starters.items()}

    for sub in subs:
        t = sub["t"]
        if sub["out_id"] and sub["out_id"] in active:
            entry_t = active.pop(sub["out_id"])
            player_info[sub["out_id"]]["intervals"].append((entry_t, t))
        if sub["in_id"]:
            active[sub["in_id"]] = t
            if sub["in_id"] not in player_info:
                player_info[sub["in_id"]] = {"name": sub["in_name"], "intervals": []}

    for pid, t0 in active.items():
        player_info[pid]["intervals"].append((t0, MATCH_END))

    return player_info


def extract_goals(incidents, is_home: bool):
    """
    Devuelve listas (team_goals, rival_goals) con tiempos en segundos
    desde la perspectiva de is_home.
    """
    team_goals  = []
    rival_goals = []
    for inc in (incidents.get("incidents") or []):
        if inc.get("incidentType") != "goal":
            continue
        t = min(get_time_sec(inc), MATCH_END)
        if inc.get("isHome", False) == is_home:
            team_goals.append(t)
        else:
            rival_goals.append(t)
    return team_goals, rival_goals


def on_field(intervals, t):
    return any(s <= t < e for s, e in intervals)


def process_side(meta, lineups, incidents, is_home: bool):
    team  = meta["home_team"] if is_home else meta["away_team"]
    rival = meta["away_team"] if is_home else meta["home_team"]

    player_info            = build_player_intervals(lineups, incidents, is_home)
    team_goals, rival_goals = extract_goals(incidents, is_home)

    rows = []
    for pid, info in player_info.items():
        ivs = info["intervals"]
        if not ivs:
            continue
        minutes = sum(e - s for s, e in ivs) / 60.0
        gf_on   = sum(1 for t in team_goals  if on_field(ivs, t))
        ga_on   = sum(1 for t in rival_goals if on_field(ivs, t))
        rows.append({
            "player_id": pid,
            "player":    info["name"],
            "team":      team,
            "rival":     rival,
            "match_id":  meta["event_id"],
            "date":      meta.get("start_timestamp"),
            "venue":     "Home" if is_home else "Away",
            "minutes":   round(minutes, 1),
            "gf_on":     gf_on,
            "ga_on":     ga_on,
            "pm":        gf_on - ga_on,
        })
    return rows


def process_match(match_dir):
    meta      = load_json(match_dir / "meta.json")
    lineups   = load_json(match_dir / "lineups.json")
    incidents = load_json(match_dir / "incidents.json")
    if not meta or not lineups or not incidents:
        return []

    rows = []
    rows.extend(process_side(meta, lineups, incidents, is_home=True))
    rows.extend(process_side(meta, lineups, incidents, is_home=False))
    return rows


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    match_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir()) if RAW_DIR.exists() else []
    print(f"Partidos encontrados: {len(match_dirs)}")

    all_rows = []
    for d in match_dirs:
        rows = process_match(d)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("Sin datos. Ejecuta primero 01_scrape.py")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], unit="s").dt.date

    # Detalle por partido-jugador
    df.to_csv(PROCESSED_DIR / "player_match_pm.csv", index=False)
    print(f"Detalle guardado en {PROCESSED_DIR / 'player_match_pm.csv'}")

    # Agrupado por jugador
    agg = (
        df.groupby(["player_id", "player", "team"])
        .agg(
            partidos=("match_id", "nunique"),
            minutos=("minutes", "sum"),
            gf_on=("gf_on", "sum"),
            ga_on=("ga_on", "sum"),
            pm=("pm", "sum"),
        )
        .reset_index()
    )
    agg["pm_per90"] = (agg["pm"] / agg["minutos"] * 90).round(2)
    agg["minutos"]  = agg["minutos"].round(1)
    agg = agg.sort_values("pm", ascending=False).reset_index(drop=True)

    out = REPORTS_DIR / "plus_minus_liga.csv"
    agg.to_csv(out, index=False)
    print(f"\nRanking guardado en {out}")
    print(f"Jugadores totales: {len(agg)}")
    print()
    print(agg[["player", "team", "partidos", "minutos", "gf_on", "ga_on", "pm", "pm_per90"]]
          .head(20).to_string(index=False))


if __name__ == "__main__":
    main()
