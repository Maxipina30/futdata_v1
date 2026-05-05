import json
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path("analysis/colo_colo_plus_minus/data/raw")
TEAM = "Colo-Colo"

# pid -> {name, pos_count: {pos: count}}
players = {}

for d in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
    meta      = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    lineups   = json.loads((d / "lineups.json").read_text(encoding="utf-8"))
    side = "home" if meta["home_team"] == TEAM else "away"

    for entry in lineups.get(side, {}).get("players", []):
        p    = entry.get("player", {})
        pid  = p.get("id")
        name = p.get("name") or p.get("shortName", "?")
        pos  = entry.get("position") or p.get("position", "?")
        if pid not in players:
            players[pid] = {"name": name, "pos_count": defaultdict(int)}
        players[pid]["pos_count"][pos] += 1

print(f"{'Jugador':<30} {'Pos por partido (pos: n veces)'}")
print("-" * 65)
for pid, d in sorted(players.items(), key=lambda x: x[1]["name"]):
    counts = dict(sorted(d["pos_count"].items(), key=lambda x: -x[1]))
    counts_str = "  |  ".join(f"{pos}: {n}" for pos, n in counts.items())
    main_pos = max(counts, key=counts.get)
    print(f"{d['name']:<30} {counts_str}   canon: {main_pos}")
