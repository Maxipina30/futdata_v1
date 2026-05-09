import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
RAW_DIR = BASE_DIR / "data" / "raw"

POSITION_LABELS = {
    "G": "Arquero",
    "D": "Defensas",
    "M": "Mediocampistas",
    "F": "Delanteros",
}
FORMATION_442 = {"G": 1, "D": 4, "M": 4, "F": 2}
LINE_ORDER = ["F", "M", "D", "G"]
TABLE_ORDER = {"G": 0, "D": 1, "M": 2, "F": 3}
COLO_XI_BY_COMBOS = {
    "G": ["Gabriel Maureira"],
    "D": ["Diego Ulloa", "Jonathan Villagra", "Joaquin Sosa", "Jeyson Rojas"],
    "M": ["Claudio Aquino", "Arturo Vidal", "Tomas Alarcon", "Victor Felipe Mendez"],
    "F": ["Lautaro Pastran", "Maximiliano Romero"],
}


def fix_text(value):
    if not isinstance(value, str):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def fix_text_columns(table):
    output = table.copy()
    for column in output.select_dtypes(include="object").columns:
        output[column] = output[column].map(fix_text)
    return output


def number(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def signed_number(value):
    if pd.isna(value):
        return ""
    return f"{float(value):+.2f}"


def last_name(name):
    return str(name).split()[-1]


def normalized_name(name):
    text = unicodedata.normalize("NFKD", str(fix_text(name)))
    return "".join(char for char in text if not unicodedata.combining(char)).lower().strip()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def incident_second(incident):
    seconds = incident.get("timeSeconds")
    if seconds is not None:
        return min(int(seconds), 5400)
    minute = incident.get("time")
    return min(int(minute) * 60, 5400) if minute is not None else 0


def in_interval(second, intervals):
    return any(start <= second < end for start, end in intervals)


def interval_label(start_second, end_second):
    start = int(round(start_second / 60))
    end = int(round(end_second / 60))
    return f"{start}-{end}"


def merge_intervals(intervals):
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def group_key(names):
    return " · ".join(sorted(last_name(name) for name in names))


def split_group_names(group):
    return [part.strip() for part in re.split(r"\s*(?:·|Â·|\|)\s*", str(group)) if part.strip()]


def normalized_group_key(names):
    return tuple(sorted(normalized_name(name) for name in names))


@st.cache_data
def load_player_pool():
    index = fix_text_columns(pd.read_csv(REPORTS_DIR / "player_index.csv"))
    stats = fix_text_columns(pd.read_csv(REPORTS_DIR / "player_stats.csv"))
    plus_minus = fix_text_columns(pd.read_csv(REPORTS_DIR / "plus_minus_liga.csv"))
    subs_path = REPORTS_DIR / "sub_impact_by_player.csv"
    subs = fix_text_columns(pd.read_csv(subs_path)) if subs_path.exists() else pd.DataFrame()

    players = stats.merge(
        index[["player_id", "position", "pos_label"]],
        on="player_id",
        how="left",
    )
    players = players.merge(
        plus_minus[["player_id", "partidos", "minutos", "gf_on", "ga_on", "pm", "pm_per90"]],
        on="player_id",
        how="left",
    )
    players = players.rename(columns={"minutos": "pm_minutes", "pm": "plus_minus"})
    players["team"] = players["team"].map(fix_text)
    players["player"] = players["player"].map(fix_text)
    players["score_rol"] = players.apply(score_player, axis=1).round(3)
    teams = sorted(players["team"].dropna().unique())
    return players, subs, teams


def score_player(row):
    minutes = min(float(row.get("minutes") or 0), 900) / 900
    pm = np.clip(float(row.get("pm_per90") or 0), -1.5, 1.8)
    duel = np.clip(float(row.get("duel_win_pct") or 45), 0, 75) / 100
    rec = np.clip(float(row.get("rec_per90") or 0), 0, 9)
    ga = np.clip(float(row.get("ga_per90") or 0), 0, 2.5)
    kp = np.clip(float(row.get("kp_per90") or 0), 0, 4)
    saves = np.clip(float(row.get("saves_per90") or 0), 0, 6)
    defensive = np.clip(float(row.get("int_per90") or 0) + float(row.get("clr_per90") or 0), 0, 12)
    goals = np.clip(float(row.get("g_per90") or 0), 0, 1.2)

    position = row.get("position")
    if position == "G":
        return pm * 0.28 + saves * 0.12 + minutes * 0.32 + duel * 0.08 - ga * 0.08 + rec * 0.04
    if position == "D":
        return pm * 0.27 + defensive * 0.08 + duel * 0.18 + rec * 0.04 + minutes * 0.25 + kp * 0.03
    if position == "M":
        return pm * 0.25 + kp * 0.08 + rec * 0.07 + duel * 0.14 + minutes * 0.24 + ga * 0.07
    if position == "F":
        return pm * 0.22 + goals * 0.22 + ga * 0.11 + kp * 0.07 + minutes * 0.20 + duel * 0.08
    return pm * 0.25 + minutes * 0.25 + duel * 0.10


def recommend_xi(team_players):
    selected = {}
    used_ids = set()
    for position, amount in FORMATION_442.items():
        candidates = team_players[
            team_players["position"].eq(position)
            & team_players["minutes"].fillna(0).gt(0)
            & ~team_players["player_id"].isin(used_ids)
        ].sort_values(["score_rol", "minutes"], ascending=False)
        chosen = candidates.head(amount).copy()
        selected[position] = chosen
        used_ids.update(chosen["player_id"].tolist())

    total = sum(len(frame) for frame in selected.values())
    if total < 11:
        extras = team_players[
            team_players["minutes"].fillna(0).gt(0)
            & ~team_players["player_id"].isin(used_ids)
        ].sort_values(["score_rol", "minutes"], ascending=False).head(11 - total)
        selected.setdefault("M", pd.DataFrame())
        selected["M"] = pd.concat([selected["M"], extras], ignore_index=True)

    return selected


def fixed_xi_from_names(team_players, names_by_position):
    selected = {}
    by_name = {
        normalized_name(row["player"]): row
        for _, row in team_players.iterrows()
    }
    used_ids = set()
    for position, names in names_by_position.items():
        rows = []
        for name in names:
            row = by_name.get(normalized_name(name))
            if row is not None:
                rows.append(row)
                used_ids.add(row["player_id"])
        selected[position] = pd.DataFrame(rows)

    for position, amount in FORMATION_442.items():
        current = len(selected.get(position, pd.DataFrame()))
        if current >= amount:
            continue
        fallback = team_players[
            team_players["position"].eq(position)
            & team_players["minutes"].fillna(0).gt(0)
            & ~team_players["player_id"].isin(used_ids)
        ].sort_values(["minutes", "score_rol"], ascending=False).head(amount - current)
        selected[position] = pd.concat([selected.get(position, pd.DataFrame()), fallback], ignore_index=True)
        used_ids.update(fallback["player_id"].tolist())

    return selected


def players_from_group(team_players, position, group, used_ids):
    rows = []
    position_players = team_players[
        team_players["position"].eq(position)
        & ~team_players["player_id"].isin(used_ids)
    ].copy()
    position_players["normalized_last_name"] = position_players["player"].map(
        lambda value: normalized_name(last_name(value))
    )
    for name in split_group_names(group):
        candidates = position_players[position_players["normalized_last_name"].eq(normalized_name(name))]
        if candidates.empty:
            continue
        rows.append(candidates.sort_values(["minutes", "score_rol"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def observed_combo_xi(team_players, team):
    groups = build_line_groups(team)
    if groups.empty:
        return recommend_xi(team_players)

    selected = {}
    used_ids = set()
    for position, amount in FORMATION_442.items():
        line_groups = groups[groups["position"].eq(position)].copy()
        line_groups["group_size"] = line_groups["Grupo"].map(lambda value: len(split_group_names(value)))
        line_groups = line_groups[line_groups["group_size"].eq(amount)]
        if line_groups.empty:
            selected[position] = pd.DataFrame()
            continue
        line_groups["combo_score"] = (
            line_groups["Min juntos"].fillna(0).clip(0, 450) / 450 * 0.65
            + line_groups["+/-90"].fillna(0).clip(-2, 2) / 2 * 0.25
            + line_groups["+/- juntos"].fillna(0).clip(-3, 6) / 6 * 0.10
        )
        best_group = line_groups.sort_values(
            ["combo_score", "Min juntos", "+/-90"],
            ascending=False,
        ).iloc[0]
        selected[position] = players_from_group(team_players, position, best_group["Grupo"], used_ids)
        if not selected[position].empty:
            used_ids.update(selected[position]["player_id"].tolist())

    for position, amount in FORMATION_442.items():
        current = len(selected.get(position, pd.DataFrame()))
        if current >= amount:
            continue
        fallback = team_players[
            team_players["position"].eq(position)
            & team_players["minutes"].fillna(0).gt(0)
            & ~team_players["player_id"].isin(used_ids)
        ].sort_values(["minutes", "score_rol"], ascending=False).head(amount - current)
        selected[position] = pd.concat([selected.get(position, pd.DataFrame()), fallback], ignore_index=True)
        used_ids.update(fallback["player_id"].tolist())

    return selected


def player_chip(name):
    return f'<div class="xi-player">{name}</div>'


def pitch_line(players):
    return '<div class="xi-line">' + "".join(player_chip(player) for player in players) + "</div>"


def show_pitch(selected):
    names = {
        position: selected.get(position, pd.DataFrame()).get("player", pd.Series(dtype=object)).tolist()
        for position in FORMATION_442
    }
    html = f"""
    <style>
    .xi-pitch {{
        background: linear-gradient(180deg, #1e7b45 0%, #146039 100%);
        border: 2px solid rgba(255,255,255,0.78);
        border-radius: 12px;
        padding: 20px 18px;
        min-height: 520px;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.22);
    }}
    .xi-pitch::before {{
        content: "";
        position: absolute;
        inset: 18px;
        border: 1px solid rgba(255,255,255,0.45);
        border-radius: 8px;
        pointer-events: none;
    }}
    .xi-pitch::after {{
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 150px;
        height: 150px;
        transform: translate(-50%, -50%);
        border: 1px solid rgba(255,255,255,0.40);
        border-radius: 50%;
        pointer-events: none;
    }}
    .xi-halfway {{
        position: absolute;
        left: 18px;
        right: 18px;
        top: 50%;
        border-top: 1px solid rgba(255,255,255,0.35);
    }}
    .xi-block {{
        position: relative;
        z-index: 1;
        margin-bottom: 26px;
    }}
    .xi-label {{
        color: rgba(255,255,255,0.82);
        font-size: 13px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
        text-transform: uppercase;
    }}
    .xi-line {{
        display: flex;
        gap: 12px;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
    }}
    .xi-player {{
        background: rgba(15, 25, 35, 0.88);
        border: 1px solid rgba(255,255,255,0.60);
        color: #ffffff;
        border-radius: 999px;
        padding: 8px 14px;
        min-width: 145px;
        text-align: center;
        font-weight: 650;
        box-shadow: 0 6px 16px rgba(0,0,0,0.22);
    }}
    </style>
    <div class="xi-pitch">
        <div class="xi-halfway"></div>
        <div class="xi-block"><div class="xi-label">Delanteros</div>{pitch_line(names["F"])}</div>
        <div class="xi-block"><div class="xi-label">Mediocampistas</div>{pitch_line(names["M"])}</div>
        <div class="xi-block"><div class="xi-label">Defensas</div>{pitch_line(names["D"])}</div>
        <div class="xi-block"><div class="xi-label">Arquero</div>{pitch_line(names["G"])}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.caption(
        "Formacion base 4-4-2. El dato identifica lineas G/D/M/F, pero no lateralidad fina como LD, LI o extremo."
    )


def selected_table(selected):
    table = pd.concat(
        [frame.assign(linea=POSITION_LABELS[position]) for position, frame in selected.items() if not frame.empty],
        ignore_index=True,
    )
    table["orden"] = table["position"].map(TABLE_ORDER)
    columns = [
        "linea", "player", "minutes", "plus_minus", "pm_per90", "goals", "assists",
        "ga_per90", "kp_per90", "rec_per90", "duel_win_pct", "score_rol",
    ]
    columns = [column for column in columns if column in table.columns]
    return table.sort_values(["orden", "score_rol"], ascending=[True, False])[columns].rename(columns={
        "linea": "Linea",
        "player": "Jugador",
        "minutes": "Min",
        "plus_minus": "+/-",
        "pm_per90": "+/-90",
        "goals": "Goles",
        "assists": "Asistencias",
        "ga_per90": "G+A90",
        "kp_per90": "Pases clave90",
        "rec_per90": "Recuperaciones90",
        "duel_win_pct": "% duelos",
        "score_rol": "Score",
    })


def selected_group_rows(team, selected):
    groups = build_line_groups(team)
    if groups.empty:
        return pd.DataFrame()
    rows = []
    for position, frame in selected.items():
        if frame.empty:
            continue
        selected_key = normalized_group_key(last_name(name) for name in frame["player"].tolist())
        line_groups = groups[groups["position"].eq(position)].copy()
        line_groups["group_key"] = line_groups["Grupo"].map(lambda value: normalized_group_key(split_group_names(value)))
        match = line_groups[line_groups["group_key"].map(lambda value: value == selected_key)]
        if match.empty:
            continue
        rows.append(match.sort_values(["Min juntos", "+/-90"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def show_xi_reasoning(team, selected, is_colo=False):
    if is_colo:
        st.write(
            "La estructura sale de juntar las mejores señales por línea: defensa de 4, "
            "mediocampistas de 4 y la dupla Pastrán-Romero en ataque."
        )
        st.markdown("**Conclusión principal**")
        st.write(
            "La recomendación es partir desde un 4-4-2 porque es la forma que mejor respeta las "
            "combinaciones positivas ya observadas: mantiene una línea defensiva reconocible, junta "
            "a los mediocampistas con mejor balance colectivo y aprovecha la dupla Pastrán-Romero, "
            "que es la asociación ofensiva con mejor impacto compartido."
        )
        st.write(
            "No significa que el 4-4-2 sea siempre superior: significa que, con esta muestra, es la "
            "formación menos forzada para convertir los datos en un XI inicial. Las alternativas deben "
            "evaluarse especialmente por roles de banda, donde el dato actual todavía no distingue LD/LI."
        )

    group_rows = selected_group_rows(team, selected)
    if group_rows.empty:
        st.info(
            "No encontré minutos exactos compartidos para todas las líneas del XI; la lectura se apoya en "
            "los grupos disponibles y en los jugadores que completan la estructura."
        )
        return

    st.markdown("**Lectura de la recomendación**")
    for _, row in group_rows.sort_values("position", key=lambda col: col.map(TABLE_ORDER)).iterrows():
        line = row["Linea"].lower()
        group = row["Grupo"]
        minutes = number(row["Min juntos"])
        pm90 = signed_number(row["+/-90"])
        gf = int(row["GF juntos"])
        ga = int(row["GC juntos"])
        if row["Min juntos"] >= 180 and row["+/-90"] >= 0:
            st.write(
                f"- {line}: {group} combina muestra útil ({minutes} min) y balance no negativo "
                f"({gf}-{ga}, {pm90} +/-90)."
            )
        elif row["+/-90"] > 0:
            st.write(
                f"- {line}: {group} deja buena señal de impacto ({gf}-{ga}, {pm90} +/-90), "
                f"pero con muestra más corta ({minutes} min)."
            )
        else:
            st.write(
                f"- {line}: {group} es la combinación más defendible por estructura/minutos disponibles "
                f"({minutes} min, {gf}-{ga})."
            )

    st.info(
        "Conclusión: el XI prioriza combinaciones que ya convivieron en cancha. Cuando el dato no alcanza "
        "para una línea completa, se completa con jugadores de mayor continuidad en esa zona."
    )


def show_colo_forward_note():
    st.markdown("**Ajuste importante en delantera**")
    st.write(
        "La tabla histórica favorece Pastrán-Romero porque tiene más minutos y buen balance, "
        "pero el partido ante Coquimbo muestra algo que esa tabla ocultaba: los goles de Correa llegaron "
        "después de la salida de Romero, en un tramo ofensivo corto."
    )
    st.info(
        "Conclusión: Correa no debería quedar fuera de la lectura de delantera. Con los datos actuales, "
        "Pastrán-Romero sigue siendo la dupla con mejor muestra positiva, pero Correa merece recomendación "
        "fuerte como alternativa ofensiva o primer cambio para buscar gol."
    )


@st.cache_data
def canonical_positions(team):
    counts = {}
    for match_dir in sorted(path for path in RAW_DIR.iterdir() if path.is_dir()):
        meta = read_json(match_dir / "meta.json")
        lineups = read_json(match_dir / "lineups.json")
        if not meta or not lineups or team not in {fix_text(meta["home_team"]), fix_text(meta["away_team"])}:
            continue
        side = "home" if fix_text(meta["home_team"]) == team else "away"
        for entry in lineups.get(side, {}).get("players", []):
            player = entry.get("player", {})
            player_id = player.get("id")
            position = entry.get("position") or player.get("position")
            if player_id is None or not position:
                continue
            counts.setdefault(player_id, {})
            counts[player_id][position] = counts[player_id].get(position, 0) + 1
    return {player_id: max(items.items(), key=lambda item: item[1])[0] for player_id, items in counts.items()}


def build_player_intervals(lineups, incidents, team_is_home, positions):
    side = "home" if team_is_home else "away"
    players = {}
    active = {}

    for entry in lineups.get(side, {}).get("players", []):
        player = entry.get("player", {})
        player_id = player.get("id")
        if player_id is None:
            continue
        name = fix_text(player.get("name") or player.get("shortName") or f"id_{player_id}")
        players[player_id] = {
            "name": name,
            "position": positions.get(player_id) or entry.get("position") or player.get("position"),
            "intervals": [],
        }
        if not entry.get("substitute", True):
            active[player_id] = 0

    substitutions = []
    for incident in incidents.get("incidents") or []:
        if incident.get("incidentType") != "substitution":
            continue
        if incident.get("isHome", False) != team_is_home:
            continue
        substitutions.append({
            "second": incident_second(incident),
            "in_id": (incident.get("playerIn") or {}).get("id"),
            "out_id": (incident.get("playerOut") or {}).get("id"),
        })

    for substitution in sorted(substitutions, key=lambda item: item["second"]):
        if substitution["out_id"] in active:
            players[substitution["out_id"]]["intervals"].append(
                (active.pop(substitution["out_id"]), substitution["second"])
            )
        if substitution["in_id"] in players:
            active[substitution["in_id"]] = substitution["second"]

    for player_id, start in active.items():
        players[player_id]["intervals"].append((start, 5400))
    return players


@st.cache_data
def build_line_groups(team):
    rows_by_group = {}
    positions = canonical_positions(team)
    if not RAW_DIR.exists():
        return pd.DataFrame()

    for match_dir in sorted(path for path in RAW_DIR.iterdir() if path.is_dir()):
        meta = read_json(match_dir / "meta.json")
        lineups = read_json(match_dir / "lineups.json")
        incidents = read_json(match_dir / "incidents.json")
        if not meta or not lineups or not incidents:
            continue

        home_team = fix_text(meta["home_team"])
        away_team = fix_text(meta["away_team"])
        if team not in {home_team, away_team}:
            continue
        team_is_home = home_team == team
        players = build_player_intervals(lineups, incidents, team_is_home, positions)

        goals = []
        for incident in incidents.get("incidents") or []:
            if incident.get("incidentType") == "goal":
                goals.append({
                    "second": incident_second(incident),
                    "for_team": incident.get("isHome", False) == team_is_home,
                })

        cuts = {0, 5400}
        for data in players.values():
            for start, end in data["intervals"]:
                if 0 < start < 5400:
                    cuts.add(start)
                if 0 < end < 5400:
                    cuts.add(end)
        cuts = sorted(cuts)

        for index in range(len(cuts) - 1):
            start, end = cuts[index], cuts[index + 1]
            if end <= start:
                continue
            midpoint = (start + end) / 2
            minutes = (end - start) / 60
            gf = sum(1 for goal in goals if start <= goal["second"] < end and goal["for_team"])
            ga = sum(1 for goal in goals if start <= goal["second"] < end and not goal["for_team"])
            for position in POSITION_LABELS:
                active_line = [
                    data for data in players.values()
                    if data.get("position") == position and in_interval(midpoint, data["intervals"])
                ]
                if not active_line:
                    continue
                combo = group_key([data["name"] for data in active_line])
                key = (position, combo)
                rows_by_group.setdefault(key, {"position": position, "combo": combo, "minutes": 0.0, "gf": 0, "ga": 0})
                rows_by_group[key]["minutes"] += minutes
                rows_by_group[key]["gf"] += gf
                rows_by_group[key]["ga"] += ga

    rows = []
    for data in rows_by_group.values():
        pm = data["gf"] - data["ga"]
        minutes = data["minutes"]
        rows.append({
            "position": data["position"],
            "Linea": POSITION_LABELS[data["position"]],
            "Grupo": data["combo"],
            "Min juntos": round(minutes, 1),
            "GF juntos": int(data["gf"]),
            "GC juntos": int(data["ga"]),
            "+/- juntos": int(pm),
            "+/-90": round(pm / minutes * 90, 2) if minutes else 0,
        })
    return pd.DataFrame(rows)


@st.cache_data
def breakdown_group_by_match(team, position, group):
    rows = []
    positions = canonical_positions(team)
    target_key = normalized_group_key(split_group_names(group))
    if not RAW_DIR.exists():
        return pd.DataFrame()

    for match_dir in sorted(path for path in RAW_DIR.iterdir() if path.is_dir()):
        meta = read_json(match_dir / "meta.json")
        lineups = read_json(match_dir / "lineups.json")
        incidents = read_json(match_dir / "incidents.json")
        if not meta or not lineups or not incidents:
            continue

        home_team = fix_text(meta["home_team"])
        away_team = fix_text(meta["away_team"])
        if team not in {home_team, away_team}:
            continue

        team_is_home = home_team == team
        players = build_player_intervals(lineups, incidents, team_is_home, positions)
        goals = []
        for incident in incidents.get("incidents") or []:
            if incident.get("incidentType") == "goal":
                goals.append({
                    "second": incident_second(incident),
                    "for_team": incident.get("isHome", False) == team_is_home,
                })

        cuts = {0, 5400}
        for data in players.values():
            for start, end in data["intervals"]:
                if 0 < start < 5400:
                    cuts.add(start)
                if 0 < end < 5400:
                    cuts.add(end)
        cuts = sorted(cuts)

        intervals = []
        gf = 0
        ga = 0
        for index in range(len(cuts) - 1):
            start, end = cuts[index], cuts[index + 1]
            if end <= start:
                continue
            midpoint = (start + end) / 2
            active_line = [
                data for data in players.values()
                if data.get("position") == position and in_interval(midpoint, data["intervals"])
            ]
            active_key = normalized_group_key(last_name(data["name"]) for data in active_line)
            if active_key != target_key:
                continue
            intervals.append((start, end))
            gf += sum(1 for goal in goals if start <= goal["second"] < end and goal["for_team"])
            ga += sum(1 for goal in goals if start <= goal["second"] < end and not goal["for_team"])

        minutes_together = sum(end - start for start, end in intervals) / 60
        if minutes_together <= 0:
            continue

        team_score = meta.get("home_score") if team_is_home else meta.get("away_score")
        opponent_score = meta.get("away_score") if team_is_home else meta.get("home_score")
        if team_score > opponent_score:
            result_state = "Ganó"
        elif team_score == opponent_score:
            result_state = "Empató"
        else:
            result_state = "Perdió"

        rows.append({
            "Fecha": pd.to_datetime(meta.get("start_timestamp"), unit="s").date(),
            "Rival": away_team if team_is_home else home_team,
            "Sede": "Local" if team_is_home else "Visita",
            "Resultado": f"{home_team} {meta.get('home_score')}-{meta.get('away_score')} {away_team}",
            "Estado": result_state,
            "Tramos juntos": ", ".join(interval_label(start, end) for start, end in merge_intervals(intervals)),
            "Min juntos": round(minutes_together, 1),
            "GF juntos": int(gf),
            "GC juntos": int(ga),
            "+/- juntos": int(gf - ga),
        })

    return pd.DataFrame(rows).sort_values("Fecha") if rows else pd.DataFrame()


def show_position_comparison(team_players):
    st.markdown("**Comparacion puesto a puesto**")
    position = st.selectbox(
        "Puesto",
        ["G", "D", "M", "F"],
        index=1,
        format_func=lambda value: POSITION_LABELS.get(value, value),
        key="xi_position",
    )
    min_minutes = st.slider("Minutos minimos", 0, 900, 120, step=30, key="xi_min_position")
    candidates = team_players[
        team_players["position"].eq(position) & team_players["minutes"].fillna(0).ge(min_minutes)
    ].sort_values(["score_rol", "minutes"], ascending=False)
    if candidates.empty:
        st.info("No hay jugadores para ese filtro.")
        return

    best = candidates.iloc[0]
    st.success(
        f"Recomendado: {best['player']} ({signed_number(best['pm_per90'])} +/-90, {number(best['minutes'])} min)."
    )
    columns = [
        "player", "minutes", "plus_minus", "pm_per90", "goals", "assists",
        "ga_per90", "kp_per90", "crosses_per90", "pass_acc_pct", "int_per90",
        "clr_per90", "rec_per90", "duel_win_pct", "score_rol",
    ]
    columns = [column for column in columns if column in candidates.columns]
    table = candidates[columns].rename(columns={
        "player": "Jugador",
        "minutes": "Min",
        "plus_minus": "+/-",
        "pm_per90": "+/-90",
        "goals": "Goles",
        "assists": "Asistencias",
        "ga_per90": "G+A90",
        "kp_per90": "Pases clave90",
        "crosses_per90": "Centros90",
        "pass_acc_pct": "% pase",
        "int_per90": "Intercepciones90",
        "clr_per90": "Despejes90",
        "rec_per90": "Recuperaciones90",
        "duel_win_pct": "% duelos",
        "score_rol": "Score",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)


def show_line_groups(team):
    st.markdown("**Combinaciones por linea**")
    groups = build_line_groups(team)
    if groups.empty:
        st.info("No encontre combinaciones por linea para este equipo.")
        return
    line = st.selectbox(
        "Linea",
        ["G", "D", "M", "F"],
        index=1,
        format_func=lambda value: POSITION_LABELS.get(value, value),
        key="xi_line_groups",
    )
    min_group_minutes = st.slider("Minutos minimos del grupo", 0, 180, 45, step=5, key="xi_group_min")
    filtered = groups[
        groups["position"].eq(line) & groups["Min juntos"].fillna(0).ge(min_group_minutes)
    ].sort_values(["Min juntos", "+/-90"], ascending=False)
    if filtered.empty:
        st.info("No hay grupos para esa linea con el minimo elegido.")
        return
    st.dataframe(
        filtered[["Linea", "Grupo", "Min juntos", "GF juntos", "GC juntos", "+/- juntos", "+/-90"]],
        use_container_width=True,
        hide_index=True,
    )
    selected_group = st.selectbox("Grupo a desglosar por partido", filtered["Grupo"].tolist(), key="xi_group_breakdown")
    if not st.button("Analizar grupo", key="xi_group_breakdown_button"):
        st.info("Pulsa analizar para ver partidos, tramos juntos e impacto del grupo.")
        return

    breakdown = breakdown_group_by_match(team, line, selected_group)
    if breakdown.empty:
        st.info("No encontré minutos compartidos para ese grupo.")
        return

    total_minutes = breakdown["Min juntos"].sum()
    total_gf = breakdown["GF juntos"].sum()
    total_ga = breakdown["GC juntos"].sum()
    total_pm = breakdown["+/- juntos"].sum()
    pm90 = total_pm / total_minutes * 90 if total_minutes else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min juntos", number(total_minutes))
    c2.metric("GF/GC juntos", f"{int(total_gf)}/{int(total_ga)}")
    c3.metric("+/- juntos", f"{int(total_pm):+d}")
    c4.metric("+/-90 grupo", signed_number(pm90))

    if total_minutes < 120:
        st.warning(
            "Conclusión del grupo seleccionado: señal interesante, pero muestra baja. "
            "Sirve para detectar una variante, no para afirmar titularidad."
        )
    elif total_pm > 0:
        st.success(
            "Conclusión del grupo seleccionado: balance positivo con muestra razonable. "
            "Es una combinación candidata a sostenerse o repetirse."
        )
    elif total_pm == 0:
        st.info(
            "Conclusión del grupo seleccionado: balance neutro. Puede ser útil si aporta control, "
            "pero no aparece como ventaja clara en goles durante sus tramos."
        )
    else:
        st.error(
            "Conclusión del grupo seleccionado: balance negativo. Conviene revisar rivales, contexto "
            "y si el problema fue estructural o de un partido puntual."
        )

    st.dataframe(breakdown, use_container_width=True, hide_index=True)


def show_substitutes(team_players, subs, selected):
    st.markdown("**Alternativas desde la banca**")
    if subs.empty:
        st.info("No hay datos de sustituciones para este equipo.")
        return
    used = set(pd.concat(selected.values(), ignore_index=True)["player"].tolist())
    team_subs = subs[subs["team"].map(fix_text).eq(team_players["team"].iloc[0])].copy()
    bench = team_players[~team_players["player"].isin(used)].copy()
    if team_subs.empty or bench.empty:
        st.info("No hay datos suficientes de banca para este equipo.")
        return
    bench = bench.merge(
        team_subs[["player_in", "entradas", "pm_after_avg", "pm_after_sum"]],
        left_on="player",
        right_on="player_in",
        how="left",
    )
    bench["entradas"] = bench["entradas"].fillna(0)
    bench["pm_after_avg"] = bench["pm_after_avg"].fillna(0)
    bench["score_banca"] = (
        bench["score_rol"].fillna(0) * 0.45
        + bench["pm_after_avg"].clip(-1, 1.5) * 0.30
        + bench["entradas"].clip(0, 8) / 8 * 0.25
    )
    team = team_players["team"].iloc[0]
    bench = bench[bench["entradas"].gt(0)].copy()
    if team == "Colo-Colo":
        preferred = ["Yastin Cuevas", "Javier Correa", "Francisco Marchant"]
        preferred_rows = bench[bench["player"].isin(preferred)].copy()
        missing = 3 - len(preferred_rows)
        if missing > 0:
            extra = bench[~bench["player"].isin(preferred_rows["player"])].sort_values(
                ["score_banca", "pm_after_avg"],
                ascending=False,
            ).head(missing)
            bench = pd.concat([preferred_rows, extra], ignore_index=True)
        else:
            bench = preferred_rows
        bench["preferred_order"] = bench["player"].map({name: i for i, name in enumerate(preferred)}).fillna(99)
        bench = bench.sort_values(["preferred_order", "score_banca"], ascending=[True, False]).head(3)
    else:
        bench = bench.sort_values(["score_banca", "pm_after_avg"], ascending=False).head(3)
    if bench.empty:
        st.info("No hay suplentes con ingresos registrados para recomendar.")
        return
    st.dataframe(
        bench[["player", "position", "entradas", "pm_after_avg", "pm_per90", "score_banca"]].rename(columns={
            "player": "Jugador",
            "position": "Pos",
            "entradas": "Entradas",
            "pm_after_avg": "+/- tras entrar",
            "pm_per90": "+/-90",
            "score_banca": "Score banca",
        }),
        use_container_width=True,
        hide_index=True,
    )

    columns = st.columns(min(3, len(bench)))
    for column, (_, row) in zip(columns, bench.iterrows()):
        with column:
            st.metric(row["player"], f"{int(row['entradas'])} entradas")
            if team == "Colo-Colo" and row["player"] == "Yastin Cuevas":
                st.write(
                    f"Revulsivo directo: {number(row['ga_per90'])} G+A90, "
                    f"{number(row['sh_per90'])} tiros90 y buena señal de impacto al entrar."
                )
            elif team == "Colo-Colo" and row["player"] == "Javier Correa":
                st.write(
                    f"Plan de gol: {int(row['goals'])} goles, {number(row['sh_per90'])} tiros90. "
                    "Sirve si el partido pide área y volumen de remate."
                )
            elif team == "Colo-Colo" and row["player"] == "Francisco Marchant":
                st.write(
                    f"Plan creativo: {number(row['kp_per90'])} pases clave90 y "
                    f"{number(row['rec_per90'])} recuperaciones90. Cambia ritmo sin romper tanto el medio."
                )
            else:
                st.write(
                    f"Alternativa por impacto: {signed_number(row['pm_after_avg'])} +/- tras entrar, "
                    f"{signed_number(row['pm_per90'])} +/-90."
                )


def show_colo_xi_tab():
    st.subheader("Recomendar XI")
    players, subs, teams = load_player_pool()
    selected_team = st.selectbox("Equipo", teams, index=teams.index("Colo-Colo") if "Colo-Colo" in teams else 0)
    team_players = players[players["team"].eq(selected_team)].copy()
    if team_players.empty:
        st.info("No hay datos para ese equipo.")
        return

    uses_fixed_colo_logic = selected_team == "Colo-Colo"
    selected = fixed_xi_from_names(team_players, COLO_XI_BY_COMBOS) if uses_fixed_colo_logic else observed_combo_xi(team_players, selected_team)
    with st.container(border=True):
        st.markdown(f"**XI recomendado para {selected_team}: 4-4-2**")
        show_pitch(selected)
        st.dataframe(selected_table(selected), use_container_width=True, hide_index=True)
        if uses_fixed_colo_logic:
            st.caption(
                "Para Colo-Colo se conserva la recomendación original por combinaciones observadas."
            )
        else:
            st.caption("Recomendación por combinaciones reales observadas por línea.")
        show_xi_reasoning(selected_team, selected, uses_fixed_colo_logic)
        show_substitutes(team_players, subs, selected)
        if uses_fixed_colo_logic:
            show_colo_forward_note()

    st.divider()
    show_position_comparison(team_players)
    st.divider()
    show_line_groups(selected_team)
