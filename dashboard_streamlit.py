import importlib
import re
import sys
import unicodedata
from itertools import combinations
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
REPORT_DIR = BASE_DIR / "files" / "05_reports"
ODDS_DIR = BASE_DIR / "files" / "06_odds"
HISTORY_DIR = BASE_DIR / "files" / "07_recommendation_history"
DEFAULT_SOURCE_URL = (
    "https://www.cuotasahora.com/football/h2h/arsenal-hA1Zm19f/"
    "newcastle-p6ahwuwJ/#OQsq6PYa:over-under;2;"
)

LEAGUES = {
    "Premier League": {"key": "premier", "report_dir": REPORT_DIR},
    "La Liga": {"key": "la_liga", "report_dir": REPORT_DIR / "la_liga"},
    "Serie A": {"key": "serie_a", "report_dir": REPORT_DIR / "serie_a"},
}

DEFAULT_RULES = {
    "min_prob_safe": 0.85,
    "min_prob_high": 0.82,
    "min_prob_value": 0.55,
    "min_ev_value": 0.15,
    "min_ev_risky": 0.12,
    "allowed_negative_ev": -0.08,
    "low_legs": 4,
    "low_pick_odds": (1.05, 1.60),
    "low_min_prob": 0.68,
    "custom_legs": (3, 5),
    "custom_total_odds": (2.0, 10.0),
    "custom_pick_odds": (1.10, 1.50),
    "high_total_odds": (5.0, 10.0),
    "low_total_min_odds": 1.60,
    "low_total_max_odds": 2.00,
}

ODDS_MARKET_COLUMNS = [
    "decimal_home_win",
    "decimal_draw",
    "decimal_away_win",
    "decimal_home_or_draw",
    "decimal_home_or_away",
    "decimal_draw_or_away",
    "decimal_over_15",
    "decimal_under_15",
    "decimal_over_25",
    "decimal_under_25",
    "decimal_btts_yes",
    "decimal_btts_no",
]

KNOWN_CUOTASAHORA_URLS = {
    ("brighton", "chelsea"): "https://www.cuotasahora.com/football/h2h/brighton-2XrRecc3/chelsea-4fGZN2oK/",
    ("bournemouth", "leeds united"): "https://www.cuotasahora.com/football/h2h/bournemouth-OtpNdwrc/leeds-tUxUbLR2/",
    ("burnley", "manchester city"): "https://www.cuotasahora.com/football/h2h/burnley-z3dmTMMO/manchester-city-Wtn9Stg0/",
    ("sunderland", "nottingham forest"): "https://www.cuotasahora.com/football/h2h/sunderland-WSzc94ws/nottingham-UsushcZr/",
    ("arsenal", "newcastle united"): "https://www.cuotasahora.com/football/h2h/arsenal-hA1Zm19f/newcastle-p6ahwuwJ/",
    ("fulham", "aston villa"): "https://www.cuotasahora.com/football/h2h/fulham-69ZiU2Om/aston-villa-W00wmLO0/",
    ("liverpool", "crystal palace"): "https://www.cuotasahora.com/football/h2h/liverpool-lId4TMwf/crystal-palace-AovF1Mia/",
    ("west ham united", "everton"): "https://www.cuotasahora.com/football/h2h/west-ham-Cxq57r8g/everton-KluSTr9s/",
    ("wolves", "tottenham hotspur"): "https://www.cuotasahora.com/football/h2h/wolves-j3Azpf5d/tottenham-UDg08Ohm/",
    ("manchester utd", "brentford"): "https://www.cuotasahora.com/football/h2h/manchester-utd-ppjDR086/brentford-xYe7DwID/",
    ("everton", "manchester city"): "https://www.cuotasahora.com/football/h2h/everton-KluSTr9s/manchester-city-Wtn9Stg0/",
}

TEAM_ALIASES = {
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "manchester united": "manchester utd",
    "man united": "manchester utd",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "newcastle": "newcastle united",
    "newcastle united": "newcastle united",
    "nottingham": "nottingham forest",
    "tottenham": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "west ham": "west ham united",
    "west ham united": "west ham united",
    "wolves": "wolves",
    "wolverhampton wanderers": "wolves",
    "leeds utd": "leeds united",
    "leeds united": "leeds united",
    "athletic bilbao": "athletic club",
    "atletico de madrid": "atletico madrid",
    "celta de vigo": "celta vigo",
    "inter": "internazionale",
    "hellas verona": "hellas verona",
    "verona": "hellas verona",
    "napoles": "napoli",
    "nápoles": "napoli",
    "real oviedo": "oviedo",
    "real betis balompie": "real betis",
    "rcd espanyol": "espanyol",
    "espanyol barcelona": "espanyol",
    "levante ud": "levante",
    "valencia cf": "valencia",
    "villarreal cf": "villarreal",
    "fc barcelona": "barcelona",
    "bolonia": "bologna",
    "ac milan": "milan",
    "as roma": "roma",
    "ss lazio": "lazio",
    "deportes limache": "cd limache",
    "limache": "cd limache",
    "colo colo": "colo-colo",
    "colo colo": "colo-colo",
    "u de chile": "universidad de chile",
    "udechile": "universidad de chile",
    "u catolica": "universidad catolica",
    "universidad catolica": "universidad catolica",
    "union la calera": "union la calera",
    "nublense": "nublense",
    "ohiggins": "o higgins",
    "o higgins": "o higgins",
    "deportes concepcion": "deportes concepcion",
    "universidad de concepcion": "universidad de concepcion",
    "coquimbo": "coquimbo unido",
    "la serena": "la serena",
}


st.set_page_config(page_title="FutData apuestas", layout="wide")


def normalize_team(name):
    if not isinstance(name, str):
        return ""
    text = name.strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)


def pct(value):
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def dec(value):
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def signed_pct(value):
    if pd.isna(value):
        return ""
    return f"{value:+.1%}"


def league_label(league_key):
    for label, config in LEAGUES.items():
        if config["key"] == league_key:
            return label
    return league_key


@st.cache_data(show_spinner=False)
def load_predictions(matchweek):
    path_1x2 = REPORT_DIR / f"predicciones_matchweek{matchweek}.csv"
    if not path_1x2.exists():
        return pd.DataFrame(), f"No existe {path_1x2}"

    pred = pd.read_csv(path_1x2, parse_dates=["date"])
    goals_path = REPORT_DIR / "predicciones_goles_mw34_plus.csv"
    if goals_path.exists():
        goals = pd.read_csv(goals_path, parse_dates=["date"])
        goals = goals[goals["round_num"].eq(matchweek)].copy()
        goal_cols = [
            "date",
            "round_num",
            "local_team",
            "away_team",
            "p_over_15",
            "p_over_25",
            "p_btts",
        ]
        for col in ["target_over_15", "target_over_25", "target_btts"]:
            if col in goals.columns:
                goal_cols.append(col)
        pred = pred.merge(
            goals[goal_cols],
            on=["date", "round_num", "local_team", "away_team"],
            how="left",
        )
    pred["home_key"] = pred["local_team"].apply(normalize_team)
    pred["away_key"] = pred["away_team"].apply(normalize_team)
    return pred, ""


@st.cache_data(show_spinner=False)
def load_future_predictions_all():
    frames = []
    for label, config in LEAGUES.items():
        report_dir = config["report_dir"]
        one_x_two_paths = [
            report_dir / "predicciones_partidos_train_all.csv",
            report_dir / "predicciones_futuras_1x2.csv",
        ]
        league_frames = []
        for one_x_two_path in one_x_two_paths:
            if not one_x_two_path.exists():
                continue
            pred = pd.read_csv(one_x_two_path, parse_dates=["date"])
            if "p_home_win" not in pred.columns:
                rename_probs = {
                    "P_-1": "p_away_win",
                    "P_0": "p_draw",
                    "P_1": "p_home_win",
                    "Prediccion_Final": "prediccion",
                }
                pred = pred.rename(columns={k: v for k, v in rename_probs.items() if k in pred.columns})
                if {"p_home_win", "p_draw", "p_away_win"}.issubset(pred.columns):
                    pred["confianza"] = pred[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
            league_frames.append(pred)
        if not league_frames:
            continue
        pred = pd.concat(league_frames, ignore_index=True)
        pred = pred.drop_duplicates(["date", "round_num", "local_team", "away_team"], keep="last")
        pred["league"] = config["key"]
        pred["Liga"] = label

        goal_frames = []
        for goals_path in [
            report_dir / "predicciones_goles_train_all.csv",
            report_dir / "predicciones_goles_mw34_plus.csv",
        ]:
            if goals_path.exists():
                goal_frames.append(pd.read_csv(goals_path, parse_dates=["date"]))
        if goal_frames:
            goals = pd.concat(goal_frames, ignore_index=True)
            goals = goals.drop_duplicates(["date", "round_num", "local_team", "away_team"], keep="last")
            goal_cols = [
                "date",
                "round_num",
                "local_team",
                "away_team",
                "p_over_15",
                "p_over_25",
                "p_btts",
            ]
            for col in ["target_over_15", "target_over_25", "target_btts"]:
                if col in goals.columns:
                    goal_cols.append(col)
            pred = pred.merge(
                goals[[col for col in goal_cols if col in goals.columns]],
                on=["date", "round_num", "local_team", "away_team"],
                how="left",
            )
        frames.append(pred)

    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["home_key"] = data["local_team"].apply(normalize_team)
    data["away_key"] = data["away_team"].apply(normalize_team)
    return data.sort_values(["date", "Liga", "local_team"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_odds(path):
    odds_path = Path(path)
    if not odds_path.exists():
        return pd.DataFrame(columns=["home_key", "away_key"])
    odds = pd.read_csv(odds_path)
    if odds.empty:
        return pd.DataFrame(columns=["home_key", "away_key"])
    odds["home_key"] = odds["local_team"].apply(normalize_team)
    odds["away_key"] = odds["away_team"].apply(normalize_team)
    return odds


@st.cache_data(show_spinner=False)
def load_all_odds():
    frames = []
    for odds_path in ODDS_DIR.glob("*.csv"):
        try:
            odds = pd.read_csv(odds_path)
        except Exception:
            continue
        if odds.empty or "decimal_home_win" not in odds.columns:
            continue
        if not {"local_team", "away_team"}.issubset(odds.columns):
            continue
        odds["odds_local_team"] = odds["local_team"]
        odds["odds_away_team"] = odds["away_team"]
        odds["home_key"] = odds["local_team"].apply(normalize_team)
        odds["away_key"] = odds["away_team"].apply(normalize_team)
        odds["odds_source_file"] = odds_path.name
        frames.append(odds)

    if not frames:
        return pd.DataFrame(columns=["home_key", "away_key"])

    all_odds = pd.concat(frames, ignore_index=True)
    odds_cols = [
        "home_key",
        "away_key",
        "odds_local_team",
        "odds_away_team",
        "decimal_home_win",
        "decimal_draw",
        "decimal_away_win",
        "decimal_home_or_draw",
        "decimal_home_or_away",
        "decimal_draw_or_away",
        "decimal_over_15",
        "decimal_under_15",
        "decimal_over_25",
        "decimal_under_25",
        "decimal_btts_yes",
        "decimal_btts_no",
        "odds_source_file",
    ]
    all_odds = all_odds[[col for col in odds_cols if col in all_odds.columns]]
    return all_odds.drop_duplicates(["home_key", "away_key"], keep="last")


def odds_path_for_matchweek(matchweek):
    path = ODDS_DIR / f"cuotasahora_matchweek{matchweek}_consolidated.csv"
    return path if path.exists() else ODDS_DIR / "cuotasahora_matchweek35_consolidated.csv"


def scrape_odds(source_url, limit, matchweek):
    sys.path.insert(0, str(SRC_DIR))
    scraper = importlib.import_module("08_scrape_cuotasahora")
    pred, _ = load_predictions(matchweek)
    wanted = set(zip(pred["home_key"], pred["away_key"])) if not pred.empty else set()

    _, related, _ = scraper.scrape_urls([source_url])
    if not related.empty and wanted:
        related["home_key"] = related["home"].apply(normalize_team)
        related["away_key"] = related["away"].apply(normalize_team)
        related = related[related.apply(lambda row: (row["home_key"], row["away_key"]) in wanted, axis=1)]

    urls = related["url"].head(limit).tolist() if not related.empty else [source_url]
    for pair, url in KNOWN_CUOTASAHORA_URLS.items():
        if not wanted or pair in wanted:
            urls.append(url)
    urls = list(dict.fromkeys(urls))

    odds, _, failures = scraper.scrape_urls(urls)
    if not odds.empty and wanted:
        odds["home_key"] = odds["local_team"].apply(normalize_team)
        odds["away_key"] = odds["away_team"].apply(normalize_team)
        odds = odds[odds.apply(lambda row: (row["home_key"], row["away_key"]) in wanted, axis=1)]
        odds = odds.drop(columns=["home_key", "away_key"])
    output_path = ODDS_DIR / f"cuotasahora_matchweek{matchweek}_consolidated.csv"
    odds.to_csv(output_path, index=False)
    if not failures.empty:
        failures.to_csv(ODDS_DIR / "cuotasahora_scrape_failures.csv", index=False)
    st.cache_data.clear()
    return odds, failures


def explain_pick(row):
    probability = pct(row["Prob. modelo"])
    fair_odds = dec(row["Cuota justa"])
    real_odds = dec(row["Cuota real"])
    edge = signed_pct(row["Edge"])
    if row["Entra por"] == "probabilidad":
        return (
            f"Entra por probabilidad: modelo {probability}, cuota justa {fair_odds}, "
            f"cuota real {real_odds}, edge {edge}."
        )
    if row["Entra por"] == "valor":
        return (
            f"Entra por valor: cuota real {real_odds} sobre cuota justa {fair_odds}; "
            f"modelo {probability}, edge {edge}."
        )
    return f"Modelo {probability}, cuota justa {fair_odds}, cuota real {real_odds}, edge {edge}."


def add_market(candidates, row, market, probability, odds_col, label, rules):
    if pd.isna(probability):
        return
    odds = row.get(odds_col)
    fair_odds = 1 / probability if probability and probability > 0 else None
    implied = 1 / odds if pd.notna(odds) and odds and odds > 1 else None
    edge = probability - implied if implied is not None else None
    expected_value = probability * odds - 1 if pd.notna(odds) and odds and odds > 1 else None

    recommendation_rank = 0
    if market.startswith("Doble oportunidad") and probability >= 0.80 and (
        expected_value is None or expected_value >= rules["allowed_negative_ev"]
    ):
        recommendation = "Bajo riesgo"
        recommendation_rank = 4
    elif probability >= rules["min_prob_safe"]:
        recommendation = "Alta probabilidad"
        recommendation_rank = 3
    elif (label == "Over 1.5" or market == "Ambos anotan") and probability >= 0.75 and (
        expected_value is None or expected_value >= rules["allowed_negative_ev"]
    ):
        recommendation = "Bajo riesgo"
        recommendation_rank = 4
    elif probability >= rules["min_prob_high"] and (
        expected_value is None or expected_value >= rules["allowed_negative_ev"]
    ):
        recommendation = "Alta probabilidad"
        recommendation_rank = 3
    elif expected_value is not None and expected_value >= rules["min_ev_value"] and probability >= rules["min_prob_value"]:
        recommendation = "Valor"
        recommendation_rank = 2
    elif expected_value is not None and expected_value >= rules["min_ev_risky"] and probability >= 0.40:
        recommendation = "Valor con riesgo"
        recommendation_rank = 1
    else:
        recommendation = ""

    if recommendation in {"Bajo riesgo", "Alta probabilidad"}:
        entry_reason = "probabilidad"
    elif recommendation:
        entry_reason = "valor"
    else:
        entry_reason = ""

    candidates.append(
        {
            "Fecha": int(row["round_num"]),
            "Jornada": int(row["round_num"]),
            "Liga": row.get("Liga", ""),
            "Fecha partido": row.get("date"),
            "Partido": f"{row['local_team']} vs {row['away_team']}",
            "Mercado": market,
            "Pick": label,
            "Prob. modelo": probability,
            "Cuota real": odds,
            "Cuota justa": fair_odds,
            "Edge": edge,
            "EV": expected_value,
            "Recomendacion": recommendation,
            "Entra por": entry_reason,
            "Prioridad": recommendation_rank,
            "Tiene cuota": pd.notna(odds),
        }
    )


def pick_result(row, data):
    if data.empty:
        return "Pendiente", None
    match = data[data.apply(lambda item: f"{item['local_team']} vs {item['away_team']}" == row["Partido"], axis=1)]
    if match.empty:
        return "Pendiente", None
    match_row = match.iloc[0]

    outcome = None
    if row["Mercado"] == "1X2":
        target = match_row.get("Target")
        if pd.isna(target):
            return "Pendiente", None
        if row["Pick"] == match_row["local_team"]:
            outcome = int(target) == 1
        elif row["Pick"] == match_row["away_team"]:
            outcome = int(target) == -1
    elif row["Mercado"] == "Doble oportunidad":
        target = match_row.get("Target")
        if pd.isna(target):
            return "Pendiente", None
        if row["Pick"].startswith(f"{match_row['local_team']} o empate"):
            outcome = int(target) in {0, 1}
        elif row["Pick"].startswith("Empate o "):
            outcome = int(target) in {0, -1}
    elif row["Mercado"] == "Goles" and row["Pick"] == "Over 1.5":
        target = match_row.get("target_over_15")
        if pd.isna(target):
            return "Pendiente", None
        outcome = int(target) == 1
    elif row["Mercado"] == "Goles" and row["Pick"] == "Over 2.5":
        target = match_row.get("target_over_25")
        if pd.isna(target):
            return "Pendiente", None
        outcome = int(target) == 1
    elif row["Mercado"] == "Ambos anotan" and row["Pick"] == "Si":
        target = match_row.get("target_btts")
        if pd.isna(target):
            return "Pendiente", None
        outcome = int(target) == 1

    if outcome is None:
        return "Pendiente", None
    if outcome:
        return "✅", row["Cuota real"] - 1 if pd.notna(row["Cuota real"]) else None
    return "❌", -1


def add_results(recs, data):
    if recs.empty:
        return recs
    settled = recs.copy()
    results = settled.apply(lambda row: pick_result(row, data), axis=1)
    settled["Resultado"] = [item[0] for item in results]
    settled["Retorno 1u"] = [item[1] for item in results]
    return settled


def settle_parlay(parlay):
    if parlay.empty or "Resultado" not in parlay.columns:
        return "Pendiente", None
    results = parlay["Resultado"].tolist()
    if any(result == "❌" for result in results):
        return "❌", -1
    if all(result == "✅" for result in results):
        return "✅", parlay["Cuota real"].prod() - 1
    return "Pendiente", None


def save_history(matchweek, low_parlay, high_parlay):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for label, frame, stake in [
        ("combinada_bajo_riesgo", low_parlay, 3.0),
        ("combinada_riesgo_moderado", high_parlay, 0.5),
    ]:
        if frame.empty:
            continue
        status, return_1u = settle_parlay(frame)
        summary = summarize_parlay(frame)
        return_units = return_1u * stake if return_1u is not None else None
        rows.append(
            pd.DataFrame(
                [
                    {
                        "tipo": label,
                        "Stake u": stake,
                        "Fecha": matchweek,
                        "Partido": "COMBINADA",
                        "Mercado": "Combinada",
                        "Pick": " + ".join(frame["Pick"].astype(str).tolist()),
                        "Prob. modelo": summary["probabilidad_modelo"],
                        "Cuota real": summary["cuota"],
                        "Cuota justa": 1 / summary["probabilidad_modelo"]
                        if summary["probabilidad_modelo"] > 0
                        else pd.NA,
                        "Edge": pd.NA,
                        "EV": summary["ev"],
                        "Recomendacion": label.replace("_", " "),
                        "Entra por": "reglas fijas",
                        "Resultado": status,
                        "Retorno 1u": return_1u,
                        "Retorno stake": return_units,
                        "Explicacion": "Combinada sugerida por las reglas fijas del dashboard.",
                    }
                ]
            )
        )
    if not rows:
        return None
    history = pd.concat(rows, ignore_index=True)
    history.insert(0, "fecha_snapshot", timestamp)
    history.insert(0, "matchweek", matchweek)
    output_path = HISTORY_DIR / f"recomendaciones_mw{matchweek}_{timestamp}.csv"
    history.to_csv(output_path, index=False)
    return output_path


def load_history():
    if not HISTORY_DIR.exists():
        return pd.DataFrame()
    files = sorted(HISTORY_DIR.glob("recomendaciones_mw*.csv"))
    frames = []
    for file in files:
        try:
            frame = pd.read_csv(file)
        except Exception:
            continue
        for col in ["Resultado", "Retorno 1u"]:
            if col not in frame.columns:
                frame[col] = pd.NA
        if "Stake u" not in frame.columns:
            frame["Stake u"] = frame["tipo"].map(
                {"combinada_bajo_riesgo": 3.0, "combinada_riesgo_moderado": 0.5}
            ).fillna(1.0)
        if "Retorno stake" not in frame.columns:
            frame["Retorno stake"] = frame["Retorno 1u"] * frame["Stake u"]
        frame["archivo"] = file.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    if "tipo" in history.columns:
        history = history[history["tipo"].astype(str).str.startswith("combinada")].copy()
    return history


def roi_summary(frame):
    settled = frame[frame["Retorno stake"].notna()].copy() if "Retorno stake" in frame else pd.DataFrame()
    if settled.empty:
        return {"apuestas": 0, "aciertos": 0, "roi": None, "retorno": None, "stake": None}
    stake = settled["Stake u"].sum() if "Stake u" in settled else len(settled)
    return {
        "apuestas": len(settled),
        "aciertos": int(settled["Resultado"].eq("✅").sum()),
        "roi": settled["Retorno stake"].sum() / stake if stake else None,
        "retorno": settled["Retorno stake"].sum(),
        "stake": stake,
    }


def build_recommendations(data, rules):
    candidates = []
    for _, row in data.iterrows():
        add_market(candidates, row, "1X2", row["p_home_win"], "decimal_home_win", row["local_team"], rules)
        add_market(candidates, row, "1X2", row["p_away_win"], "decimal_away_win", row["away_team"], rules)
        add_market(
            candidates,
            row,
            "Doble oportunidad",
            row["p_home_win"] + row["p_draw"],
            "decimal_home_or_draw",
            f"{row['local_team']} o empate",
            rules,
        )
        add_market(
            candidates,
            row,
            "Doble oportunidad",
            row["p_draw"] + row["p_away_win"],
            "decimal_draw_or_away",
            f"Empate o {row['away_team']}",
            rules,
        )
        if "p_over_15" in row and pd.notna(row["p_over_15"]):
            add_market(candidates, row, "Goles", row["p_over_15"], "decimal_over_15", "Over 1.5", rules)
        if "p_over_25" in row and pd.notna(row["p_over_25"]):
            add_market(candidates, row, "Goles", row["p_over_25"], "decimal_over_25", "Over 2.5", rules)
        if "p_btts" in row and pd.notna(row["p_btts"]):
            add_market(candidates, row, "Ambos anotan", row["p_btts"], "decimal_btts_yes", "Si", rules)

    recs = pd.DataFrame(candidates)
    if recs.empty:
        return recs
    recs["score"] = (
        recs["Prob. modelo"].fillna(0) * 0.65
        + recs["EV"].fillna(-0.05).clip(-0.2, 0.3) * 0.35
    )
    recs["Explicacion"] = recs.apply(explain_pick, axis=1)
    return recs.sort_values(["Prioridad", "score"], ascending=[False, False])


def add_unique_picks(selected, candidates, limit, label):
    used = {(item["Partido"], item["Mercado"], item["Pick"]) for item in selected}
    for item in candidates.to_dict("records"):
        key = (item["Partido"], item["Mercado"], item["Pick"])
        if key in used:
            continue
        item["Tipo"] = label
        selected.append(item)
        used.add(key)
        if sum(row["Tipo"] == label for row in selected) >= limit:
            break


def select_general_recommendations(recs):
    if recs.empty:
        return recs

    selected = []
    safe = recs[
        recs["Recomendacion"].isin(["Bajo riesgo", "Alta probabilidad"])
        & recs["Cuota real"].notna()
    ].sort_values(["Prob. modelo", "EV"], ascending=False)
    value = recs[
        recs["Recomendacion"].eq("Valor")
        & recs["Cuota real"].notna()
    ].sort_values(["EV", "Prob. modelo"], ascending=False)
    aggressive = recs[
        recs["Recomendacion"].eq("Valor con riesgo")
        & recs["Cuota real"].notna()
    ].sort_values(["EV", "Prob. modelo"], ascending=False)

    add_unique_picks(selected, safe, 5, "Top seguridad")
    add_unique_picks(selected, value, 5, "Top valor")
    add_unique_picks(selected, aggressive, 5, "Oportunidad agresiva")

    if not selected:
        return pd.DataFrame(columns=[*recs.columns, "Tipo"])
    return pd.DataFrame(selected)


def future_recommendations(future_data):
    if future_data.empty:
        return pd.DataFrame()
    future = future_data[pd.to_datetime(future_data["date"]).dt.date >= datetime.now().date()].copy()
    if future.empty:
        future = future_data[future_data["Target"].isna()].copy()
    recs = build_recommendations(future, DEFAULT_RULES)
    recs = add_results(recs, future)
    return select_general_recommendations(recs)


def estimated_odds(probability):
    if pd.isna(probability) or probability <= 0:
        return pd.NA
    # Estimacion cercana a cuota justa, solo para construir combinadas cuando aun no hay cuotas reales.
    return max(1.05, min(1.85, (1 / probability) * 0.98))


def prepare_recs_for_parlays(recs):
    if recs.empty:
        return recs
    prepared = recs.copy()
    prepared["Tipo cuota"] = prepared["Cuota real"].apply(lambda value: "Real" if pd.notna(value) else "Estimada")
    prepared["Cuota real"] = prepared["Cuota real"].where(
        prepared["Cuota real"].notna(),
        prepared["Prob. modelo"].apply(estimated_odds),
    )
    prepared["EV"] = prepared["Prob. modelo"] * prepared["Cuota real"] - 1
    prepared["Cuota justa"] = prepared["Prob. modelo"].apply(lambda value: 1 / value if value else pd.NA)
    prepared["Tiene cuota"] = True
    return prepared


def weekend_parlay(future_data, max_legs=5):
    if future_data.empty:
        return pd.DataFrame(), {}
    future = future_data[pd.to_datetime(future_data["date"]).dt.date >= datetime.now().date()].copy()
    if future.empty:
        future = future_data[future_data["Target"].isna()].copy()
    if future.empty:
        return pd.DataFrame(), {}

    min_date = pd.to_datetime(future["date"]).min()
    weekend = future[pd.to_datetime(future["date"]).between(min_date, min_date + pd.Timedelta(days=3))].copy()
    if weekend.empty:
        weekend = future.copy()

    recs = build_recommendations(weekend, DEFAULT_RULES)
    recs = recs[
        recs["Recomendacion"].isin(["Bajo riesgo", "Alta probabilidad"])
        & recs["Prob. modelo"].ge(0.72)
    ].copy()
    if recs.empty:
        return pd.DataFrame(), {}

    recs["Cuota usada"] = recs["Cuota real"].where(recs["Cuota real"].notna(), recs["Prob. modelo"].apply(estimated_odds))
    recs["Tipo cuota"] = recs["Cuota real"].apply(lambda value: "Real" if pd.notna(value) else "Estimada")
    recs = recs.sort_values(["Prob. modelo", "EV"], ascending=False)

    legs = []
    used_market_pairs = set()
    for _, row in recs.iterrows():
        market_key = "Resultado" if row["Mercado"] in {"1X2", "Doble oportunidad"} else row["Mercado"]
        pair = (row["Partido"], market_key)
        if pair in used_market_pairs:
            continue
        legs.append(row)
        used_market_pairs.add(pair)
        if len(legs) >= max_legs:
            break

    if len(legs) < 2:
        return pd.DataFrame(), {}
    parlay = pd.DataFrame(legs)
    return parlay, {
        "cuota": parlay["Cuota usada"].prod(),
        "probabilidad_modelo": parlay["Prob. modelo"].prod(),
        "ev": pd.NA,
    }


def low_risk_parlay(recs, rules):
    return find_best_parlay(
        recs,
        min_legs=2,
        max_legs=rules["low_legs"],
        min_total_odds=rules["low_total_min_odds"],
        max_total_odds=rules["low_total_max_odds"],
        min_pick_odds=rules["low_pick_odds"][0],
        max_pick_odds=rules["low_pick_odds"][1],
        min_probability=rules["low_min_prob"],
        min_ev=rules["allowed_negative_ev"],
        allow_empty_recommendation=False,
        allow_same_match=True,
    )


def high_risk_parlay(recs, rules):
    return find_best_parlay(
        recs,
        min_legs=5,
        max_legs=5,
        min_total_odds=rules["high_total_odds"][0],
        max_total_odds=rules["high_total_odds"][1],
        min_pick_odds=1.30,
        max_pick_odds=2.00,
        min_probability=0.30,
        min_ev=-0.20,
        allow_empty_recommendation=True,
        allow_same_match=True,
    )


def summarize_parlay(parlay):
    combined_odds = parlay["Cuota real"].prod()
    combined_prob = parlay["Prob. modelo"].prod()
    return {
        "cuota": combined_odds,
        "probabilidad_modelo": combined_prob,
        "ev": combined_prob * combined_odds - 1,
    }


def find_best_parlay(
    recs,
    min_legs,
    max_legs,
    min_total_odds,
    max_total_odds,
    min_pick_odds,
    max_pick_odds,
    min_probability,
    min_ev,
    allow_empty_recommendation=True,
    allow_same_match=True,
):
    if recs.empty:
        return pd.DataFrame(), {}

    valid = recs[
        recs["Cuota real"].notna()
        & recs["Cuota real"].between(min_pick_odds, max_pick_odds)
        & recs["Prob. modelo"].ge(min_probability)
        & recs["EV"].fillna(-99).ge(min_ev)
    ].copy()
    if not allow_empty_recommendation:
        valid = valid[valid["Recomendacion"].ne("")]
    if valid.empty:
        return pd.DataFrame(), {}

    valid = valid.sort_values(["EV", "Prob. modelo"], ascending=False).head(18)
    records = valid.to_dict("records")
    best_combo = None
    best_score = None
    target_midpoint = (min_total_odds + max_total_odds) / 2

    for size in range(int(min_legs), int(max_legs) + 1):
        for combo in combinations(records, size):
            if not allow_same_match:
                matches = [item["Partido"] for item in combo]
                if len(set(matches)) != len(matches):
                    continue
            market_pairs = [
                (
                    item["Partido"],
                    "Resultado" if item["Mercado"] in {"1X2", "Doble oportunidad"} else item["Mercado"],
                )
                for item in combo
            ]
            if len(set(market_pairs)) != len(market_pairs):
                continue
            combo_df = pd.DataFrame(combo)
            summary = summarize_parlay(combo_df)
            if summary["cuota"] > max_total_odds:
                continue
            if not (min_total_odds <= summary["cuota"] <= max_total_odds):
                continue
            odds_distance = abs(summary["cuota"] - target_midpoint) / max(target_midpoint, 1)
            score = (
                summary["probabilidad_modelo"] * 0.55
                + max(summary["ev"], -0.5) * 0.35
                - odds_distance * 0.10
            )
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo_df

    if best_combo is None:
        return pd.DataFrame(), {}
    return best_combo, summarize_parlay(best_combo)


def show_parlay(parlay, summary):
    if parlay.empty:
        st.info("No encontré una combinada que cumpla esos filtros.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Cuota combinada", dec(summary["cuota"]))
    c2.metric("Prob. modelo", pct(summary["probabilidad_modelo"]))
    c3.metric("EV combinada", pct(summary["ev"]))
    status, return_1u = settle_parlay(parlay)
    if status != "Pendiente":
        st.metric("Resultado real", status, f"{return_1u:+.2f}u")
    elif "Resultado" in parlay.columns and parlay["Resultado"].ne("Pendiente").any():
        st.metric("Resultado real", "Pendiente", "hay picks sin resultado")
    display_cols = [
        "Liga",
        "Fecha partido",
        "Partido",
        "Mercado",
        "Pick",
        "Prob. modelo",
        "Cuota real",
        "Tipo cuota",
        "EV",
        "Resultado",
        "Explicacion",
    ]
    display_cols = [col for col in display_cols if col in parlay.columns]
    st.dataframe(
        parlay[display_cols].style.format(
            {
                "Fecha partido": lambda value: pd.to_datetime(value).strftime("%Y-%m-%d")
                if pd.notna(value)
                else "",
                "Prob. modelo": pct,
                "Cuota real": dec,
                "EV": pct,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


st.title("FutData: recomendaciones de apuestas")

with st.sidebar:
    st.header("Filtros")
    selected_leagues = st.multiselect(
        "Ligas",
        options=list(LEAGUES.keys()),
        default=list(LEAGUES.keys()),
    )
    future_only = st.checkbox("Solo partidos futuros/sin resultado", value=True)
    st.divider()
    st.header("Scraper beta")
    source_url = st.text_area("URL semilla CuotasAhora", value=DEFAULT_SOURCE_URL, height=90)
    scrape_limit = st.slider("Partidos a probar", min_value=1, max_value=10, value=10)
    if st.button("Actualizar cuotas reales"):
        with st.spinner("Scrapeando y decodificando cuotas..."):
            odds_scraped, failures = scrape_odds(source_url, scrape_limit, 35)
        st.success(f"Cuotas actualizadas: {len(odds_scraped)} partidos")
        if not failures.empty:
            st.warning(f"{len(failures)} URLs fallaron; quedaron guardadas en failures.")

all_data = load_future_predictions_all()
if all_data.empty:
    st.error("No hay predicciones multi-liga generadas. Ejecuta los pipelines/predicciones primero.")
    st.stop()

data = all_data[all_data["Liga"].isin(selected_leagues)].copy()
if future_only:
    today = datetime.now().date()
    data = data[(pd.to_datetime(data["date"]).dt.date >= today) | data["Target"].isna()].copy()

odds = load_all_odds()
if not odds.empty:
    data = data.merge(odds, on=["home_key", "away_key"], how="left")

recs = add_results(build_recommendations(data, DEFAULT_RULES), data)
parlay_recs = prepare_recs_for_parlays(recs)
parlay, parlay_summary = low_risk_parlay(parlay_recs, DEFAULT_RULES)
high_parlay, high_parlay_summary = high_risk_parlay(parlay_recs, DEFAULT_RULES)

col_a, col_b, col_c = st.columns(3)
available_odds_cols = [col for col in ODDS_MARKET_COLUMNS if col in data.columns]
rows_with_odds = int(data[available_odds_cols].notna().any(axis=1).sum()) if available_odds_cols else 0
col_a.metric("Partidos considerados", len(data))
col_b.metric("Con cuotas cruzadas", rows_with_odds)
col_c.metric("Recomendaciones", int(recs["Recomendacion"].ne("").sum()) if not recs.empty else 0)
if rows_with_odds == 0:
    st.warning(
        "No hay cuotas reales cruzadas para los partidos filtrados. "
        "Las combinadas pueden usar cuotas estimadas, pero las recomendaciones de valor necesitan scraping real."
    )
if not odds.empty:
    visible_pairs = set(zip(data["home_key"], data["away_key"])) if not data.empty else set()
    odds_pairs = odds.assign(pair=list(zip(odds["home_key"], odds["away_key"])))
    unmatched_odds = odds_pairs[~odds_pairs["pair"].isin(visible_pairs)].copy()
    if not unmatched_odds.empty:
        with st.expander("Cuotas sin cruce con prediccion"):
            display_unmatched = [
                "odds_local_team",
                "odds_away_team",
                "decimal_home_win",
                "decimal_draw",
                "decimal_away_win",
                "decimal_over_15",
                "decimal_over_25",
                "decimal_btts_yes",
                "odds_source_file",
            ]
            display_unmatched = [col for col in display_unmatched if col in unmatched_odds.columns]
            st.dataframe(
                unmatched_odds[display_unmatched].style.format(
                    {
                        "decimal_home_win": dec,
                        "decimal_draw": dec,
                        "decimal_away_win": dec,
                        "decimal_over_15": dec,
                        "decimal_over_25": dec,
                        "decimal_btts_yes": dec,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

with st.sidebar:
    if st.button("Guardar historial de recomendaciones"):
        saved_path = save_history(0, parlay, high_parlay)
        if saved_path:
            st.success(f"Historial guardado: {saved_path.name}")
        else:
            st.warning("No habia recomendaciones para guardar.")

tab_best, tab_match, tab_parlays, tab_builder, tab_history, tab_matches = st.tabs(
    [
        "Mejores recomendaciones",
        "Por partido",
        "Combinadas sugeridas",
        "Crea tu combinada",
        "Historial",
        "Todos los partidos",
    ]
)

with tab_best:
    st.subheader("Mejores recomendaciones")
    st.caption(
        "Criterio: 5 picks de seguridad por probabilidad, 5 picks de valor por edge/EV, "
        "y hasta 5 oportunidades agresivas cuando la cuota paga bien pero el riesgo es mayor."
    )
    top_recs = select_general_recommendations(recs)
    if top_recs.empty:
        st.info("Todavia no hay recomendaciones con las reglas actuales. Revisa cuotas o baja los umbrales.")
    else:
        st.dataframe(
            top_recs[
                [
                    "Liga",
                    "Fecha partido",
                    "Partido",
                    "Tipo",
                    "Mercado",
                    "Pick",
                    "Prob. modelo",
                    "Cuota real",
                    "Cuota justa",
                    "Edge",
                    "EV",
                    "Recomendacion",
                    "Resultado",
                ]
            ].style.format(
                {
                    "Fecha partido": lambda value: pd.to_datetime(value).strftime("%Y-%m-%d")
                    if pd.notna(value)
                    else "",
                    "Prob. modelo": pct,
                    "Cuota real": dec,
                    "Cuota justa": dec,
                    "Edge": pct,
                    "EV": pct,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_match:
    st.subheader("Recomendaciones por partido")
    match_order = st.radio(
        "Ordenar partidos por",
        ["Fecha real", "Jornada", "Liga"],
        horizontal=True,
    )
    match_rows = (
        recs[["Liga", "Jornada", "Fecha partido", "Partido"]]
        .drop_duplicates()
        .copy()
    )
    match_rows["Fecha partido"] = pd.to_datetime(match_rows["Fecha partido"], errors="coerce")
    if match_order == "Fecha real":
        match_rows = match_rows.sort_values(["Fecha partido", "Liga", "Jornada", "Partido"])
        section_key = lambda row: row["Fecha partido"].strftime("%Y-%m-%d") if pd.notna(row["Fecha partido"]) else "Sin fecha"
    elif match_order == "Jornada":
        match_rows = match_rows.sort_values(["Jornada", "Liga", "Fecha partido", "Partido"])
        section_key = lambda row: f"Fecha {int(row['Jornada'])}" if pd.notna(row["Jornada"]) else "Sin jornada"
    else:
        match_rows = match_rows.sort_values(["Liga", "Jornada", "Fecha partido", "Partido"])
        section_key = lambda row: row["Liga"] or "Sin liga"

    current_section = None
    for _, match_row in match_rows.iterrows():
        section = section_key(match_row)
        if section != current_section:
            st.markdown(f"**{section}**")
            current_section = section

        league_name = match_row["Liga"]
        match_name = match_row["Partido"]
        match_recs = recs[(recs["Liga"].eq(league_name)) & (recs["Partido"].eq(match_name))]
        shown = match_recs[
            match_recs["Tiene cuota"] | match_recs["Recomendacion"].ne("")
        ].sort_values(["Prioridad", "score"], ascending=[False, False]).head(8)
        match_date = match_row["Fecha partido"].strftime("%Y-%m-%d") if pd.notna(match_row["Fecha partido"]) else "Sin fecha"
        expander_label = f"{match_date} | Fecha {int(match_row['Jornada'])} | {league_name} | {match_name}"
        with st.expander(expander_label):
            if shown.empty:
                st.info("Sin cuotas cruzadas o sin recomendacion clara para este partido.")
            else:
                st.dataframe(
                    shown[
                        [
                            "Jornada",
                            "Fecha partido",
                            "Mercado",
                            "Pick",
                            "Prob. modelo",
                            "Cuota real",
                            "Cuota justa",
                            "Edge",
                            "EV",
                            "Recomendacion",
                            "Entra por",
                            "Resultado",
                            "Explicacion",
                        ]
                    ].style.format(
                        {
                            "Jornada": "{:.0f}",
                            "Fecha partido": lambda value: pd.to_datetime(value).strftime("%Y-%m-%d")
                            if pd.notna(value)
                            else "",
                            "Prob. modelo": pct,
                            "Cuota real": dec,
                            "Cuota justa": dec,
                            "Edge": pct,
                            "EV": pct,
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

with tab_parlays:
    st.subheader("Combinada razonable de bajo riesgo")
    st.caption(
        "Parte con picks de probabilidad muy alta y completa con Bajo riesgo/Alta probabilidad/Valor, "
        "considera todas las ligas seleccionadas y exige cuota combinada minima 1.60."
    )
    show_parlay(parlay, parlay_summary)

    st.subheader("Combinada de riesgo alto moderado")
    st.caption(
        "Usa 5 picks entre cuota 1.30 y 2.00, con cuota total entre 5 y 10, "
        "y permite combinar mercados distintos del mismo partido."
    )
    show_parlay(high_parlay, high_parlay_summary)

with tab_builder:
    st.subheader("Crea tu combinada")
    col1, col2 = st.columns(2)
    with col1:
        leg_range = st.slider(
            "Cantidad de picks",
            min_value=2,
            max_value=6,
            value=DEFAULT_RULES["custom_legs"],
        )
        total_odds_range = st.slider(
            "Cuota total buscada",
            min_value=1.2,
            max_value=20.0,
            value=DEFAULT_RULES["custom_total_odds"],
            step=0.1,
        )
    with col2:
        pick_odds_range = st.slider(
            "Cuota por pick",
            min_value=1.01,
            max_value=5.0,
            value=DEFAULT_RULES["custom_pick_odds"],
            step=0.01,
        )
        st.caption("Puede usar mas de un pick del mismo partido si son mercados distintos.")

    custom_parlay, custom_summary = find_best_parlay(
        parlay_recs,
        min_legs=leg_range[0],
        max_legs=leg_range[1],
        min_total_odds=total_odds_range[0],
        max_total_odds=total_odds_range[1],
        min_pick_odds=pick_odds_range[0],
        max_pick_odds=pick_odds_range[1],
        min_probability=0.30,
        min_ev=-0.20,
        allow_empty_recommendation=True,
        allow_same_match=True,
    )
    show_parlay(custom_parlay, custom_summary)

with tab_history:
    st.subheader("Historial y ROI simulado")
    st.caption(
        "El historial considera 3u para la combinada de bajo riesgo y 0.5u para la combinada "
        "de riesgo alto moderado."
    )
    current_rows = []
    for label, frame, stake in [
        ("combinada_bajo_riesgo", parlay, 3.0),
        ("combinada_riesgo_moderado", high_parlay, 0.5),
    ]:
        if frame.empty:
            continue
        status, return_1u = settle_parlay(frame)
        summary = summarize_parlay(frame)
        current_rows.append(
            {
                "tipo": label,
                "Stake u": stake,
                "Pick": " + ".join(frame["Pick"].astype(str).tolist()),
                "Cuota real": summary["cuota"],
                "Prob. modelo": summary["probabilidad_modelo"],
                "EV": summary["ev"],
                "Resultado": status,
                "Retorno 1u": return_1u,
                "Retorno stake": return_1u * stake if return_1u is not None else None,
            }
        )
    current = pd.DataFrame(current_rows)
    if current.empty:
        st.info("No hay combinadas evaluables para esta fecha.")
    else:
        current_summary = roi_summary(current)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Combinadas con resultado", current_summary["apuestas"])
        c2.metric("Aciertos", current_summary["aciertos"])
        c3.metric("Retorno total", dec(current_summary["retorno"]))
        c4.metric("ROI sobre stake", pct(current_summary["roi"]))
        st.dataframe(
            current.style.format(
                {
                    "Stake u": dec,
                    "Cuota real": dec,
                    "Prob. modelo": pct,
                    "EV": pct,
                    "Retorno 1u": dec,
                    "Retorno stake": dec,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    history = load_history()
    if history.empty:
        st.caption("Aun no hay snapshots guardados. Usa el boton de la barra lateral para guardar una fecha.")
    else:
        settled_history = history[history["Retorno 1u"].notna()].copy()
        if not settled_history.empty:
            summary = (
                settled_history.groupby(["matchweek", "tipo"], dropna=False)
                .agg(
                    apuestas=("Retorno stake", "count"),
                    aciertos=("Resultado", lambda values: int((values == "✅").sum())),
                    stake=("Stake u", "sum"),
                    retorno=("Retorno stake", "sum"),
                )
                .reset_index()
            )
            summary["roi"] = summary["retorno"] / summary["stake"]
            st.dataframe(
                summary.style.format({"stake": dec, "retorno": dec, "roi": pct}),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("Snapshots guardados"):
            st.dataframe(
                history.tail(200).style.format(
                    {
                        "Prob. modelo": pct,
                        "Cuota real": dec,
                        "Cuota justa": dec,
                        "Edge": pct,
                        "EV": pct,
                        "Retorno 1u": dec,
                        "Stake u": dec,
                        "Retorno stake": dec,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

with tab_matches:
    st.subheader("Todos los partidos")
    display_cols = [
        "Liga",
        "date",
        "local_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "p_over_15",
        "p_over_25",
        "p_btts",
        "Target",
        "target_over_15",
        "target_over_25",
        "target_btts",
        "decimal_home_win",
        "decimal_draw",
        "decimal_away_win",
        "decimal_over_15",
        "decimal_over_25",
        "decimal_btts_yes",
        "odds_source_file",
    ]
    available_cols = [col for col in display_cols if col in data.columns]
    st.dataframe(
        data[available_cols].style.format(
            {
                "p_home_win": pct,
                "p_draw": pct,
                "p_away_win": pct,
                "p_over_15": pct,
                "p_over_25": pct,
                "p_btts": pct,
                "decimal_home_win": dec,
                "decimal_draw": dec,
                "decimal_away_win": dec,
                "decimal_over_15": dec,
                "decimal_over_25": dec,
                "decimal_btts_yes": dec,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Beta: las cuotas vienen de CuotasAhora cuando el endpoint responde; "
    "la recomendacion compara probabilidad del modelo vs probabilidad implicita. "
    "No es consejo financiero."
)
