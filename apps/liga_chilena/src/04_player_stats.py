"""
Extrae estadisticas de todos los jugadores de la liga desde lineups.json.
Incluye metricas ofensivas, defensivas y de portero para el radar del dashboard.

Uso:
    venv312/Scripts/python.exe apps\\liga_chilena\\src\04_player_stats.py
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR      = Path(__file__).resolve().parents[3]
RAW_DIR       = BASE_DIR / "apps" / "liga_chilena" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "apps" / "liga_chilena" / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "apps" / "liga_chilena" / "reports"


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_side(lineups, meta, is_home: bool):
    side = "home" if is_home else "away"
    team = meta["home_team"] if is_home else meta["away_team"]

    rows = []
    for entry in lineups.get(side, {}).get("players", []):
        p   = entry.get("player", {})
        pid = p.get("id")
        if pid is None:
            continue
        s = entry.get("statistics") or {}
        minutes = s.get("minutesPlayed") or 0
        if minutes == 0:
            continue

        goals   = s.get("goals")                    or 0
        assists = s.get("goalAssist")               or 0
        shots   = s.get("totalShots")               or 0
        sot     = goals + (s.get("onTargetScoringAttempt") or 0)

        rows.append({
            "player_id":     pid,
            "player":        p.get("name") or p.get("shortName", f"id_{pid}"),
            "team":          team,
            "match_id":      meta["event_id"],
            "minutes":       minutes,
            # Ofensivo
            "goals":         goals,
            "assists":       assists,
            "shots":         shots,
            "sot":           sot,
            "key_passes":    s.get("keyPass")              or 0,
            "big_chances_created": s.get("bigChanceCreated") or 0,
            "big_chances_missed":  s.get("bigChanceMissed")  or 0,
            "touches":       s.get("touches")              or 0,
            # Pases
            "passes":        s.get("totalPass")            or 0,
            "passes_acc":    s.get("accuratePass")         or 0,
            "long_balls":    s.get("totalLongBalls")       or 0,
            "long_balls_acc":s.get("accurateLongBalls")    or 0,
            "crosses":       s.get("totalCross")           or 0,
            "crosses_acc":   s.get("accurateCross")        or 0,
            # Regates
            "dribbles":      s.get("totalContest")         or 0,
            "dribbles_won":  s.get("wonContest")           or 0,
            # Defensivo
            "interceptions": s.get("interceptionWon")      or 0,
            "clearances":    s.get("totalClearance")       or 0,
            "tackles_won":   s.get("wonTackle")            or 0,
            "tackles_total": s.get("totalTackle")          or 0,
            "aerial_won":    s.get("aerialWon")            or 0,
            "ball_recovery": s.get("ballRecovery")         or 0,
            "duel_won":      s.get("duelWon")              or 0,
            "duel_lost":     s.get("duelLost")             or 0,
            "blocked_shots": s.get("outfielderBlock")      or 0,
            "fouls":         s.get("fouls")                or 0,
            "was_fouled":    s.get("wasFouled")            or 0,
            "offsides":      s.get("totalOffside")         or 0,
            # Portero
            "saves":         s.get("saves")                or 0,
        })
    return rows


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    match_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir()) if RAW_DIR.exists() else []
    print(f"Partidos: {len(match_dirs)}")

    all_rows = []
    for d in match_dirs:
        meta    = load_json(d / "meta.json")
        lineups = load_json(d / "lineups.json")
        if not meta or not lineups:
            continue
        all_rows.extend(extract_side(lineups, meta, is_home=True))
        all_rows.extend(extract_side(lineups, meta, is_home=False))

    df = pd.DataFrame(all_rows)
    df.to_csv(PROCESSED_DIR / "player_match_stats.csv", index=False)

    agg = (
        df.groupby(["player_id", "player", "team"])
        .agg(
            partidos             = ("match_id",           "nunique"),
            minutes              = ("minutes",            "sum"),
            goals                = ("goals",              "sum"),
            assists              = ("assists",            "sum"),
            shots                = ("shots",              "sum"),
            sot                  = ("sot",               "sum"),
            key_passes           = ("key_passes",         "sum"),
            big_chances_created  = ("big_chances_created","sum"),
            big_chances_missed   = ("big_chances_missed", "sum"),
            touches              = ("touches",            "sum"),
            passes               = ("passes",             "sum"),
            passes_acc           = ("passes_acc",         "sum"),
            long_balls           = ("long_balls",         "sum"),
            long_balls_acc       = ("long_balls_acc",     "sum"),
            crosses              = ("crosses",            "sum"),
            crosses_acc          = ("crosses_acc",        "sum"),
            dribbles             = ("dribbles",           "sum"),
            dribbles_won         = ("dribbles_won",       "sum"),
            interceptions        = ("interceptions",      "sum"),
            clearances           = ("clearances",         "sum"),
            tackles_won          = ("tackles_won",        "sum"),
            tackles_total        = ("tackles_total",      "sum"),
            aerial_won           = ("aerial_won",         "sum"),
            ball_recovery        = ("ball_recovery",      "sum"),
            duel_won             = ("duel_won",           "sum"),
            duel_lost            = ("duel_lost",          "sum"),
            blocked_shots        = ("blocked_shots",      "sum"),
            fouls                = ("fouls",              "sum"),
            was_fouled           = ("was_fouled",         "sum"),
            offsides             = ("offsides",           "sum"),
            saves                = ("saves",              "sum"),
        )
        .reset_index()
    )

    nan = float("nan")
    m = agg["minutes"]

    # /90
    agg["g_per90"]      = (agg["goals"]               / m * 90).round(2)
    agg["a_per90"]      = (agg["assists"]              / m * 90).round(2)
    agg["ga_per90"]     = ((agg["goals"]+agg["assists"])/ m * 90).round(2)
    agg["sh_per90"]     = (agg["shots"]                / m * 90).round(2)
    agg["kp_per90"]     = (agg["key_passes"]           / m * 90).round(2)
    agg["bcc_per90"]    = (agg["big_chances_created"]  / m * 90).round(2)
    agg["crosses_per90"]= (agg["crosses"]              / m * 90).round(2)
    agg["int_per90"]    = (agg["interceptions"]        / m * 90).round(2)
    agg["clr_per90"]    = (agg["clearances"]           / m * 90).round(2)
    agg["rec_per90"]    = (agg["ball_recovery"]        / m * 90).round(2)
    agg["saves_per90"]  = (agg["saves"]                / m * 90).round(2)
    agg["fouls_per90"]  = (agg["fouls"]                / m * 90).round(2)
    agg["fouled_per90"] = (agg["was_fouled"]           / m * 90).round(2)

    # Porcentajes
    agg["pass_acc_pct"]    = (agg["passes_acc"]    / agg["passes"].replace(0, nan)        * 100).round(1)
    agg["long_ball_pct"]   = (agg["long_balls_acc"]/ agg["long_balls"].replace(0, nan)    * 100).round(1)
    agg["cross_acc_pct"]   = (agg["crosses_acc"]   / agg["crosses"].replace(0, nan)       * 100).round(1)
    agg["dribble_win_pct"] = (agg["dribbles_won"]  / agg["dribbles"].replace(0, nan)      * 100).round(1)
    agg["tackle_acc_pct"]  = (agg["tackles_won"]   / agg["tackles_total"].replace(0, nan) * 100).round(1)
    agg["duel_win_pct"]    = (agg["duel_won"]       / (agg["duel_won"]+agg["duel_lost"]).replace(0, nan) * 100).round(1)
    agg["aerial_win_pct"]  = (agg["aerial_won"]    / (agg["aerial_won"]+agg["duel_lost"]).replace(0, nan) * 100).round(1)
    agg["conv_rate"]       = (agg["goals"] / agg["shots"].replace(0, nan) * 100).round(1)
    agg["sot_rate"]        = (agg["sot"]   / agg["shots"].replace(0, nan) * 100).round(1)

    out = REPORTS_DIR / "player_stats.csv"
    agg.sort_values("ga_per90", ascending=False).to_csv(out, index=False)
    print(f"Stats guardadas en {out}  ({len(agg)} jugadores)")


if __name__ == "__main__":
    main()
