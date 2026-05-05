"""
Construye un indice de jugadores con posicion canonica (modal entre todos los partidos).
Salida: reports/player_index.csv  con columnas:
  player_id, player, team, position, partidos

Uso:
    venv312\Scripts\python.exe analysis\liga_chilena_pm\src\00_build_player_index.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[3]
RAW_DIR     = BASE_DIR / "analysis" / "liga_chilena_pm" / "data" / "raw"
REPORTS_DIR = BASE_DIR / "analysis" / "liga_chilena_pm" / "reports"

POS_VALID = {"G", "D", "M", "F"}
POS_LABEL = {"G": "Arquero", "D": "Defensa", "M": "Mediocampo", "F": "Delantera"}


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # pid → {name, team, pos_counter, matches}
    players: dict[int, dict] = {}

    for d in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        meta    = load_json(d / "meta.json")
        lineups = load_json(d / "lineups.json")
        if not meta or not lineups:
            continue

        for is_home in (True, False):
            side = "home" if is_home else "away"
            team = meta["home_team"] if is_home else meta["away_team"]
            for entry in lineups.get(side, {}).get("players", []):
                p   = entry.get("player", {})
                pid = p.get("id")
                if pid is None:
                    continue
                name = p.get("name") or p.get("shortName", f"id_{pid}")
                pos  = entry.get("position") or p.get("position", "?")

                if pid not in players:
                    players[pid] = {"name": name, "team": team,
                                    "pos_counter": Counter(), "matches": set()}
                players[pid]["pos_counter"][pos] += 1
                players[pid]["matches"].add(meta["event_id"])
                # Actualizar equipo con el más reciente (por si hubo traspaso)
                players[pid]["team"] = team

    rows = []
    for pid, info in players.items():
        pos_modal = info["pos_counter"].most_common(1)[0][0]
        if pos_modal not in POS_VALID:
            pos_modal = "?"
        rows.append({
            "player_id": pid,
            "player":    info["name"],
            "team":      info["team"],
            "position":  pos_modal,
            "pos_label": POS_LABEL.get(pos_modal, pos_modal),
            "partidos":  len(info["matches"]),
        })

    df = pd.DataFrame(rows).sort_values(["team", "player"])
    out = REPORTS_DIR / "player_index.csv"
    df.to_csv(out, index=False)
    print(f"Jugadores indexados: {len(df)}")
    print(df["position"].value_counts().to_string())
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
