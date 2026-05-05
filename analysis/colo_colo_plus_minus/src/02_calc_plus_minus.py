"""
Calcula el +/- de los jugadores de Colo-Colo a partir de los datos scrapeados.

Para cada partido terminado:
  - Reconstruye qué jugadores estaban en cancha en cada momento
  - Asigna +1 por cada gol a favor y -1 por cada gol en contra
    mientras el jugador estaba en el campo

Métricas de salida por jugador:
  - partidos     : partidos en que apareció (titular o ingresó)
  - minutos      : minutos totales jugados
  - gf_on        : goles de Colo-Colo con el jugador en cancha
  - ga_on        : goles rivales con el jugador en cancha
  - plus_minus   : gf_on - ga_on (bruto)
  - pm_per90     : plus_minus por 90 minutos

Uso:
    venv312\Scripts\python.exe analysis\colo_colo_plus_minus\src\02_calc_plus_minus.py
"""

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "analysis" / "colo_colo_plus_minus" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "analysis" / "colo_colo_plus_minus" / "data" / "processed"
REPORTS_DIR = BASE_DIR / "analysis" / "colo_colo_plus_minus" / "reports"

TEAM_NAME = "Colo-Colo"
MATCH_DURATION_SECONDS = 90 * 60  # 5400


def get_time_sec(inc):
    """Tiempo en segundos. SofaScore omite timeSeconds en la liga chilena; en ese caso usa 'time' (minutos)."""
    ts = inc.get("timeSeconds")
    if ts is not None:
        return int(ts)
    t = inc.get("time")
    if t is not None:
        return int(t) * 60
    return 0


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_score(event_meta, side):
    return event_meta.get(f"{side}_score")


def colo_side(meta):
    """Devuelve 'home' o 'away' según dónde juega Colo-Colo."""
    if meta["home_team"] == TEAM_NAME:
        return "home"
    return "away"


def build_player_intervals(lineups, incidents, colo_is_home):
    """
    A partir del lineup y los incidents, devuelve una lista de:
      {player_id, name, intervals: [(start_sec, end_sec), ...]}
    Solo para jugadores de Colo-Colo.
    """
    side_key = "home" if colo_is_home else "away"
    lineup_side = lineups.get(side_key, {})

    # Jugadores del lineup (titulares + suplentes que entraron)
    # substitute=False → titular, substitute=True → en banca (puede entrar o no)
    starters = {}    # player_id → name
    for entry in lineup_side.get("players", []):
        player = entry.get("player", {})
        pid = player.get("id")
        name = player.get("name") or player.get("shortName", f"id_{pid}")
        if pid is None:
            continue
        if not entry.get("substitute", True):
            starters[pid] = name

    # Sustituciones que involucran a Colo-Colo
    # incidents viene ordenado de mayor a menor minuto; normalizamos
    subs = []
    for inc in (incidents.get("incidents") or []):
        if inc.get("incidentType") != "substitution":
            continue
        inc_is_home = inc.get("isHome", False)
        if inc_is_home != colo_is_home:
            continue
        player_in = (inc.get("playerIn") or {})
        player_out = (inc.get("playerOut") or {})
        subs.append({
            "time_sec": min(get_time_sec(inc), MATCH_DURATION_SECONDS),
            "in_id": player_in.get("id"),
            "in_name": player_in.get("name") or player_in.get("shortName", "?"),
            "out_id": player_out.get("id"),
        })
    subs.sort(key=lambda s: s["time_sec"])

    # Construir intervalos
    # Cada jugador: lista de (entrada_seg, salida_seg)
    active = {}  # player_id → entrada_seg
    for pid, name in starters.items():
        active[pid] = 0

    player_info = {pid: {"name": name, "intervals": []} for pid, name in starters.items()}

    for sub in subs:
        t = sub["time_sec"]
        out_id = sub["out_id"]
        in_id = sub["in_id"]
        in_name = sub["in_name"]

        if out_id and out_id in active:
            entry_t = active.pop(out_id)
            player_info[out_id]["intervals"].append((entry_t, t))

        if in_id:
            active[in_id] = t
            if in_id not in player_info:
                player_info[in_id] = {"name": in_name, "intervals": []}

    # Todos los que siguen activos terminan al final del partido
    for pid, entry_t in active.items():
        player_info[pid]["intervals"].append((entry_t, MATCH_DURATION_SECONDS))

    return player_info


def extract_goals(incidents, colo_is_home):
    """
    Devuelve lista de goles: {time_sec, is_colo_goal}
    is_colo_goal=True si Colo-Colo marcó, False si marcó el rival.
    """
    goals = []
    for inc in (incidents.get("incidents") or []):
        if inc.get("incidentType") != "goal":
            continue
        inc_is_home = inc.get("isHome", False)
        is_colo_goal = (inc_is_home == colo_is_home)
        goals.append({"time_sec": min(get_time_sec(inc), MATCH_DURATION_SECONDS), "is_colo_goal": is_colo_goal})
    return goals


def player_was_on_field(intervals, goal_time):
    return any(start <= goal_time < end for start, end in intervals)


def process_match(match_dir):
    meta = load_json(match_dir / "meta.json")
    lineups = load_json(match_dir / "lineups.json")
    incidents = load_json(match_dir / "incidents.json")

    if not meta or not lineups or not incidents:
        return []

    colo_is_home = (meta["home_team"] == TEAM_NAME)
    player_info = build_player_intervals(lineups, incidents, colo_is_home)
    goals = extract_goals(incidents, colo_is_home)

    rows = []
    for pid, info in player_info.items():
        intervals = info["intervals"]
        if not intervals:
            continue
        minutes = sum(end - start for start, end in intervals) / 60.0
        gf_on = sum(1 for g in goals if g["is_colo_goal"] and player_was_on_field(intervals, g["time_sec"]))
        ga_on = sum(1 for g in goals if not g["is_colo_goal"] and player_was_on_field(intervals, g["time_sec"]))
        rows.append({
            "player_id": pid,
            "player": info["name"],
            "match_id": meta["event_id"],
            "date": meta.get("start_timestamp"),
            "opponent": meta["away_team"] if colo_is_home else meta["home_team"],
            "venue": "Home" if colo_is_home else "Away",
            "minutes": round(minutes, 1),
            "gf_on": gf_on,
            "ga_on": ga_on,
            "plus_minus": gf_on - ga_on,
        })
    return rows


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    match_dirs = sorted(RAW_DIR.iterdir()) if RAW_DIR.exists() else []
    match_dirs = [d for d in match_dirs if d.is_dir()]
    print(f"Partidos encontrados: {len(match_dirs)}")

    all_rows = []
    for match_dir in match_dirs:
        rows = process_match(match_dir)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("Sin datos. Ejecuta primero 01_scrape.py")
        return

    df = pd.DataFrame(all_rows)

    # Convertir timestamp a fecha legible
    df["date"] = pd.to_datetime(df["date"], unit="s").dt.date

    # Guardar detalle por partido-jugador
    df.to_csv(PROCESSED_DIR / "player_match_pm.csv", index=False)
    print(f"Detalle guardado en {PROCESSED_DIR / 'player_match_pm.csv'}")

    # Agregar por jugador
    agg = (
        df.groupby(["player_id", "player"])
        .agg(
            partidos=("match_id", "nunique"),
            minutos=("minutes", "sum"),
            gf_on=("gf_on", "sum"),
            ga_on=("ga_on", "sum"),
            plus_minus=("plus_minus", "sum"),
        )
        .reset_index()
    )
    agg["pm_per90"] = (agg["plus_minus"] / agg["minutos"] * 90).round(2)
    agg["minutos"] = agg["minutos"].round(1)
    agg = agg.sort_values("plus_minus", ascending=False).reset_index(drop=True)

    out_path = REPORTS_DIR / "plus_minus_colo_colo.csv"
    agg.to_csv(out_path, index=False)
    print(f"\nRanking +/- guardado en {out_path}")
    print()
    print(agg[["player", "partidos", "minutos", "gf_on", "ga_on", "plus_minus", "pm_per90"]].to_string(index=False))


if __name__ == "__main__":
    main()
