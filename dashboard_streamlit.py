import importlib
import re
import sys
import unicodedata
import __main__
from itertools import combinations
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
REPORT_DIR = BASE_DIR / "files" / "05_reports"
ODDS_DIR = BASE_DIR / "files" / "06_odds"
HISTORY_DIR = BASE_DIR / "files" / "07_recommendation_history"
SOFASCORE_REPORT_DIR = BASE_DIR / "files" / "sofascore_pipeline" / "reports"
SOFASCORE_ODDS_DIR = BASE_DIR / "files" / "sofascore_pipeline" / "odds"
SOFASCORE_PIPELINE_ROOT = BASE_DIR / "files" / "sofascore_pipeline"
SOFASCORE_RESULTS_DIR = BASE_DIR / "files" / "sofascore_pipeline" / "results"
WEEKEND_PREDICTIONS_PATH = SOFASCORE_REPORT_DIR / "weekend_value_predictions_2026-05-02_to_2026-05-04.csv"
DEFAULT_SOURCE_URL = (
    "https://www.cuotasahora.com/football/h2h/arsenal-hA1Zm19f/"
    "newcastle-p6ahwuwJ/#OQsq6PYa:over-under;2;"
)

LEAGUES = {
    "Premier League": {"key": "premier", "report_dir": SOFASCORE_REPORT_DIR / "premier"},
    "La Liga": {"key": "la_liga", "report_dir": SOFASCORE_REPORT_DIR / "la_liga"},
    "Serie A": {"key": "serie_a", "report_dir": SOFASCORE_REPORT_DIR / "serie_a"},
    "Bundesliga": {"key": "bundesliga", "report_dir": SOFASCORE_REPORT_DIR / "bundesliga"},
    "Ligue 1": {"key": "ligue_1", "report_dir": SOFASCORE_REPORT_DIR / "ligue_1"},
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
    "liverpool fc": "liverpool",
    "liverpool": "liverpool",
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
    "alaves": "deportivo alaves",
    "deportivo alaves": "deportivo alaves",
    "girona fc": "girona",
    "bolonia": "bologna",
    "ac milan": "milan",
    "as roma": "roma",
    "ss lazio": "lazio",
    "ssc napoli": "napoli",
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


def friendly_market(value):
    labels = {
        "1X2": "Ganador",
        "Doble oportunidad": "Doble chance",
        "Goles": "Goles",
        "Ambos anotan": "Ambos marcan",
        "Combinada": "Combinada",
    }
    return labels.get(value, value)


def friendly_recommendation(value):
    labels = {
        "Bajo riesgo": "Mas segura",
        "Alta probabilidad": "Buena confianza",
        "Valor": "Buena cuota",
        "Valor con riesgo": "Mas arriesgada",
        "combinada bajo riesgo": "Combinada prudente",
        "combinada riesgo moderado": "Combinada ambiciosa",
    }
    return labels.get(value, value)


def friendly_type(value):
    labels = {
        "Top seguridad": "Seguras",
        "Top valor": "Buena cuota",
        "Oportunidad agresiva": "Arriesgadas",
    }
    return labels.get(value, value)


def friendly_reason(row):
    confidence = pct(row.get("Prob. modelo"))
    odds = dec(row.get("Cuota real"))
    recommendation = row.get("Recomendacion", "")
    if recommendation in {"Bajo riesgo", "Alta probabilidad"}:
        return f"La opcion sale fuerte por confianza ({confidence})."
    if recommendation == "Valor":
        return f"La cuota {odds} paga mejor de lo que sugiere la estimacion."
    if recommendation == "Valor con riesgo":
        return f"Puede pagar bien, pero conviene tratarla como opcion arriesgada."
    if row.get("Tipo cuota") == "Estimada":
        return "Aun sin cuota real; se usa una estimacion para armar combinadas."
    return f"Confianza estimada: {confidence}."


def friendly_pick_frame(frame, include_group=False, include_result=True, include_reason=True):
    if frame.empty:
        return frame

    display = frame.copy()
    display["Mercado"] = display["Mercado"].apply(friendly_market)
    display["Recomendacion"] = display["Recomendacion"].apply(friendly_recommendation)
    if "Tipo" in display.columns:
        display["Tipo"] = display["Tipo"].apply(friendly_type)
    if "Explicacion" in display.columns:
        display["Motivo"] = display.apply(friendly_reason, axis=1)

    columns = ["Liga", "Fecha partido", "Partido"]
    if include_group and "Tipo" in display.columns:
        columns.append("Tipo")
    columns.extend(["Mercado", "Pick", "Prob. modelo", "Cuota real", "Recomendacion"])
    if include_result and "Resultado" in display.columns:
        columns.append("Resultado")
    if include_reason and "Motivo" in display.columns:
        columns.append("Motivo")
    if "Tipo cuota" in display.columns:
        columns.append("Tipo cuota")

    columns = [col for col in columns if col in display.columns]
    display = display[columns].rename(
        columns={
            "Fecha partido": "Fecha",
            "Tipo": "Grupo",
            "Prob. modelo": "Confianza",
            "Cuota real": "Cuota",
            "Recomendacion": "Lectura",
            "Tipo cuota": "Origen cuota",
        }
    )
    return display


def friendly_pick_style(frame):
    formatters = {
        "Fecha": lambda value: pd.to_datetime(value).strftime("%Y-%m-%d") if pd.notna(value) else "",
        "Confianza": pct,
        "Cuota": dec,
    }
    return frame.style.format({key: value for key, value in formatters.items() if key in frame.columns})


def league_label(league_key):
    for label, config in LEAGUES.items():
        if config["key"] == league_key:
            return label
    return league_key


def normalize_sofascore_predictions(pred):
    pred = pred.copy()
    pred = pred.rename(
        columns={
            "prob_1": "p_home_win",
            "prob_X": "p_draw",
            "prob_2": "p_away_win",
            "target": "Target_label",
            "pred": "prediccion",
        }
    )
    target_map = {"1": 1, "X": 0, "2": -1}
    if "Target_label" in pred.columns:
        pred["Target"] = pred["Target_label"].map(target_map)
    if {"p_home_win", "p_draw", "p_away_win"}.issubset(pred.columns):
        pred["confianza"] = pred[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
    return pred


def register_sofascore_model_classes(pipeline):
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_module.CorrelationPruner = pipeline.CorrelationPruner
        main_module.SafeSelectKBest = pipeline.SafeSelectKBest
    __main__.CorrelationPruner = pipeline.CorrelationPruner
    __main__.SafeSelectKBest = pipeline.SafeSelectKBest


@st.cache_data(show_spinner=False)
def load_sofascore_goal_probabilities():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    pipeline = importlib.import_module("run_sofascore_pipeline")
    register_sofascore_model_classes(pipeline)

    frames = []
    for label, config in LEAGUES.items():
        league = config["key"]
        model_path = SOFASCORE_PIPELINE_ROOT / "models" / league / "goal_market_models.joblib"
        if not model_path.exists():
            continue
        raw = pipeline.load_raw_matchlogs(SOFASCORE_PIPELINE_ROOT, league, include_unfinished=True)
        h2h = pipeline.load_h2h_features(SOFASCORE_PIPELINE_ROOT, league)
        dataset, _ = pipeline.build_match_dataset(raw, h2h)
        if dataset.empty:
            continue

        bundle = joblib.load(model_path)
        over_15 = bundle["models"].get("over_15")
        if not over_15:
            continue
        features = over_15["features"]
        probs = over_15["model"].predict_proba(dataset[features])[:, 1]
        goals_home = pd.to_numeric(dataset.get("home_goals"), errors="coerce")
        goals_away = pd.to_numeric(dataset.get("away_goals"), errors="coerce")
        total_goals = goals_home + goals_away
        frame = dataset[["date", "round_num", "local_team", "away_team"]].copy()
        frame["league"] = league
        frame["Liga"] = label
        frame["p_over_15_model"] = probs
        frame["target_over_15_model"] = pd.NA
        if "target" in dataset.columns:
            finished = dataset["target"].notna()
        else:
            finished = total_goals.notna()
        frame.loc[finished, "target_over_15_model"] = (total_goals[finished] >= 2).astype(int)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["date", "round_num", "local_team", "away_team"],
        keep="last",
    )


@st.cache_data(show_spinner=False)
def load_sofascore_1x2_probabilities():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    pipeline = importlib.import_module("run_sofascore_pipeline")
    register_sofascore_model_classes(pipeline)

    frames = []
    target_map = {"1": 1, "X": 0, "2": -1}
    for label, config in LEAGUES.items():
        league = config["key"]
        model_path = SOFASCORE_PIPELINE_ROOT / "models" / league / "logreg_1x2.joblib"
        if not model_path.exists():
            continue
        raw = pipeline.load_raw_matchlogs(SOFASCORE_PIPELINE_ROOT, league, include_unfinished=True)
        h2h = pipeline.load_h2h_features(SOFASCORE_PIPELINE_ROOT, league)
        dataset, _ = pipeline.build_match_dataset(raw, h2h)
        if dataset.empty:
            continue

        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]
        probabilities = model.predict_proba(dataset[features])
        classes = model.named_steps["logreg"].classes_.tolist()
        frame_cols = ["date", "time", "season_id", "season_name", "round_num", "local_team", "away_team", "target"]
        frame = dataset[[col for col in frame_cols if col in dataset.columns]].copy()
        frame = frame.rename(columns={"target": "Target_label"})
        frame["league"] = league
        frame["Liga"] = label
        frame["source"] = "model"
        for idx, class_name in enumerate(classes):
            if class_name == "1":
                frame["p_home_win"] = probabilities[:, idx]
            elif class_name == "X":
                frame["p_draw"] = probabilities[:, idx]
            elif class_name == "2":
                frame["p_away_win"] = probabilities[:, idx]
        frame["Target"] = frame.get("Target_label", pd.Series(pd.NA, index=frame.index)).map(target_map)
        if {"p_home_win", "p_draw", "p_away_win"}.issubset(frame.columns):
            frame["confianza"] = frame[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["date", "round_num", "local_team", "away_team"],
        keep="last",
    )


def load_weekend_value_predictions():
    if not WEEKEND_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    weekend = pd.read_csv(WEEKEND_PREDICTIONS_PATH, parse_dates=["date"])
    if weekend.empty or "match" not in weekend.columns:
        return pd.DataFrame()

    teams = weekend["match"].astype(str).str.split(" vs ", n=1, expand=True)
    if teams.shape[1] < 2:
        return pd.DataFrame()
    data = pd.DataFrame(
        {
            "date": weekend["date"],
            "time": weekend.get("time"),
            "round_num": weekend.get("round_num"),
            "local_team": teams[0],
            "away_team": teams[1],
            "p_home_win": pd.to_numeric(weekend.get("p_home_pct"), errors="coerce") / 100,
            "p_draw": pd.to_numeric(weekend.get("p_draw_pct"), errors="coerce") / 100,
            "p_away_win": pd.to_numeric(weekend.get("p_away_pct"), errors="coerce") / 100,
            "Target": pd.NA,
            "target_over_15": pd.NA,
            "source": "weekend",
        }
    )
    label_to_key = {label: config["key"] for label, config in LEAGUES.items()}
    data["Liga"] = weekend["league"]
    data["league"] = data["Liga"].map(label_to_key)
    data["confianza"] = data[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
    goal_probs = load_sofascore_goal_probabilities()
    if not goal_probs.empty:
        data = data.merge(
            goal_probs[
                [
                    "date",
                    "round_num",
                    "local_team",
                    "away_team",
                    "p_over_15_model",
                    "target_over_15_model",
                ]
            ],
            on=["date", "round_num", "local_team", "away_team"],
            how="left",
        )
        data["p_over_15"] = data["p_over_15_model"]
        data["target_over_15"] = data["target_over_15_model"]
        data = data.drop(columns=["p_over_15_model", "target_over_15_model"])
    return data


@st.cache_data(show_spinner=False)
def load_sofascore_results_overlay():
    if not SOFASCORE_RESULTS_DIR.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(SOFASCORE_RESULTS_DIR.glob("sofascore_weekend_results_*.csv"), key=lambda item: item.stat().st_mtime):
        try:
            frame = pd.read_csv(path, parse_dates=["date"])
        except Exception:
            continue
        if frame.empty:
            continue
        frame["results_source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    results = pd.concat(frames, ignore_index=True)
    results["home_key"] = results["local_team"].apply(normalize_team)
    results["away_key"] = results["away_team"].apply(normalize_team)
    return results.drop_duplicates(["date", "home_key", "away_key"], keep="last")


def apply_results_overlay(data):
    results = load_sofascore_results_overlay()
    if data.empty or results.empty:
        return data

    updated = data.copy()
    updated["date"] = pd.to_datetime(updated["date"])
    updated["home_key"] = updated["local_team"].apply(normalize_team)
    updated["away_key"] = updated["away_team"].apply(normalize_team)
    result_cols = [
        "date",
        "home_key",
        "away_key",
        "status_type",
        "home_goals",
        "away_goals",
        "Target",
        "target_over_15",
        "results_source_file",
    ]
    result_cols = [col for col in result_cols if col in results.columns]
    updated = updated.merge(
        results[result_cols],
        on=["date", "home_key", "away_key"],
        how="left",
        suffixes=("", "_result"),
    )

    finished = updated.get("status_type").eq("finished") if "status_type" in updated.columns else pd.Series(False, index=updated.index)
    if "Target_result" in updated.columns:
        updated.loc[finished, "Target"] = updated.loc[finished, "Target_result"]
    if "target_over_15_result" in updated.columns:
        updated.loc[finished, "target_over_15"] = updated.loc[finished, "target_over_15_result"]
    if {"home_goals", "away_goals"}.issubset(updated.columns):
        updated["Marcador"] = pd.NA
        scored = finished & updated["home_goals"].notna() & updated["away_goals"].notna()
        updated.loc[scored, "Marcador"] = (
            updated.loc[scored, "home_goals"].astype("Int64").astype(str)
            + "-"
            + updated.loc[scored, "away_goals"].astype("Int64").astype(str)
        )

    drop_cols = [
        "status_type",
        "home_goals",
        "away_goals",
        "Target_result",
        "target_over_15_result",
        "results_source_file",
    ]
    return updated.drop(columns=[col for col in drop_cols if col in updated.columns])


@st.cache_data(show_spinner=False)
def load_predictions(matchweek):
    path_1x2 = SOFASCORE_REPORT_DIR / "premier" / "premier_logreg_predictions.csv"
    if not path_1x2.exists():
        return pd.DataFrame(), f"No existe {path_1x2}"

    pred = normalize_sofascore_predictions(pd.read_csv(path_1x2, parse_dates=["date"]))
    pred = pred[pred["round_num"].eq(matchweek)].copy()
    goals_path = SOFASCORE_REPORT_DIR / "premier" / "premier_goal_market_predictions.csv"
    if goals_path.exists():
        goals = pd.read_csv(goals_path, parse_dates=["date"])
        goals = goals[goals["round_num"].eq(matchweek)].copy()
        goal_cols = [
            "date",
            "round_num",
            "local_team",
            "away_team",
            "p_over_15",
        ]
        for col in ["target_over_15"]:
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
        one_x_two_path = report_dir / f"{config['key']}_logreg_predictions.csv"
        if not one_x_two_path.exists():
            continue
        pred = normalize_sofascore_predictions(pd.read_csv(one_x_two_path, parse_dates=["date"]))
        pred = pred.drop_duplicates(["date", "round_num", "local_team", "away_team"], keep="last")
        pred["league"] = config["key"]
        pred["Liga"] = label

        goals_path = report_dir / f"{config['key']}_goal_market_predictions.csv"
        if goals_path.exists():
            goals = pd.read_csv(goals_path, parse_dates=["date"])
            goals = goals.drop_duplicates(["date", "round_num", "local_team", "away_team"], keep="last")
            goal_cols = [
                "date",
                "round_num",
                "local_team",
                "away_team",
                "p_over_15",
            ]
            for col in ["p_over_15_raw", "target_over_15", "monotonic_adjusted_over_15"]:
                if col in goals.columns:
                    goal_cols.append(col)
            pred = pred.merge(
                goals[[col for col in goal_cols if col in goals.columns]],
                on=["date", "round_num", "local_team", "away_team"],
                how="left",
            )
        frames.append(pred)

    live_1x2 = load_sofascore_1x2_probabilities()
    if not live_1x2.empty:
        frames.append(live_1x2)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["source"] = data.get("source", "historical").fillna("historical")
    data["_source_priority"] = data["source"].map({"historical": 3, "weekend": 2, "model": 1}).fillna(1)
    data = data.sort_values("_source_priority", ascending=False)
    data = data.drop_duplicates(["date", "round_num", "local_team", "away_team"], keep="first")
    data = data.drop(columns=["_source_priority"])
    goal_probs = load_sofascore_goal_probabilities()
    if not goal_probs.empty:
        data = data.merge(
            goal_probs[
                [
                    "date",
                    "round_num",
                    "local_team",
                    "away_team",
                    "p_over_15_model",
                    "target_over_15_model",
                ]
            ],
            on=["date", "round_num", "local_team", "away_team"],
            how="left",
        )
        if "p_over_15" not in data.columns:
            data["p_over_15"] = pd.NA
        if "target_over_15" not in data.columns:
            data["target_over_15"] = pd.NA
        data["p_over_15"] = data["p_over_15"].combine_first(data["p_over_15_model"])
        data["target_over_15"] = data["target_over_15"].combine_first(data["target_over_15_model"])
        data = data.drop(columns=["p_over_15_model", "target_over_15_model"])
    weekend = load_weekend_value_predictions()
    if not weekend.empty:
        data = pd.concat([data, weekend], ignore_index=True)
        data["source"] = data.get("source", "historical").fillna("historical")
        data["_source_priority"] = data["source"].map({"historical": 3, "weekend": 2, "model": 1}).fillna(1)
        data = data.sort_values("_source_priority", ascending=False)
        data = data.drop_duplicates(["date", "round_num", "local_team", "away_team"], keep="first")
        data = data.drop(columns=["_source_priority"])
    data = apply_results_overlay(data)
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
def load_all_odds(cache_key=None):
    del cache_key
    frames = []
    odds_paths = sorted(ODDS_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if SOFASCORE_ODDS_DIR.exists():
        odds_paths.extend(sorted(SOFASCORE_ODDS_DIR.glob("sofascore_*_odds.csv"), key=lambda path: path.stat().st_mtime))
    for odds_path in odds_paths:
        try:
            odds = pd.read_csv(odds_path, parse_dates=["date"] if odds_path.parent == SOFASCORE_ODDS_DIR else None)
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
        odds["odds_source_priority"] = 2 if odds_path.parent == SOFASCORE_ODDS_DIR else 1
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
        "odds_source_priority",
    ]
    all_odds = all_odds[[col for col in odds_cols if col in all_odds.columns]]
    all_odds = all_odds.sort_values(["odds_source_priority"], ascending=True)

    def last_valid(series):
        valid = series.dropna()
        return valid.iloc[-1] if not valid.empty else pd.NA

    merge_cols = [col for col in all_odds.columns if col not in {"home_key", "away_key"}]
    merged = (
        all_odds.groupby(["home_key", "away_key"], as_index=False, dropna=False)[merge_cols]
        .agg(last_valid)
    )
    return merged


def odds_cache_signature():
    paths = sorted(ODDS_DIR.glob("*.csv"))
    if SOFASCORE_ODDS_DIR.exists():
        paths.extend(sorted(SOFASCORE_ODDS_DIR.glob("sofascore_*_odds.csv")))
    return tuple(
        (str(path.relative_to(BASE_DIR)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
        if path.exists()
    )


def odds_path_for_matchweek(matchweek):
    path = ODDS_DIR / f"cuotasahora_matchweek{matchweek}_consolidated.csv"
    return path if path.exists() else ODDS_DIR / "cuotasahora_matchweek35_consolidated.csv"


def scrape_odds(source_url, limit, matchweek):
    sys.path.insert(0, str(SRC_DIR))
    scraper = importlib.import_module("07_scrape_cuotasahora_odds")
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
    real_odds = dec(row["Cuota real"])
    if row["Entra por"] == "probabilidad":
        return f"La opcion tiene una confianza alta ({probability}). Cuota disponible: {real_odds}."
    if row["Entra por"] == "valor":
        return f"La cuota {real_odds} parece interesante para una confianza de {probability}."
    return f"Confianza estimada: {probability}. Cuota disponible: {real_odds}."


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
    elif label == "Over 1.5" and probability >= 0.68 and (
        expected_value is None or expected_value >= -0.20
    ):
        recommendation = "Bajo riesgo"
        recommendation_rank = 4
    elif label == "Over 1.5" and probability >= 0.62 and (
        expected_value is None or expected_value >= -0.25
    ):
        recommendation = "Alta probabilidad"
        recommendation_rank = 3
    elif market == "Ambos anotan" and probability >= 0.75 and (
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


def result_market_group(row):
    if row["Mercado"] in {"1X2", "Doble oportunidad"}:
        return (row["Partido"], "Resultado")
    return (row["Partido"], row["Mercado"], row["Pick"])


def resolve_result_recommendation_conflicts(recs):
    if recs.empty:
        return recs

    resolved = recs.copy()
    result_mask = resolved["Mercado"].isin(["1X2", "Doble oportunidad"]) & resolved["Recomendacion"].ne("")
    if not result_mask.any():
        return resolved

    keep_indexes = set()
    for _, group in resolved[result_mask].groupby("Partido", dropna=False):
        best = group.sort_values(
            ["Prioridad", "score", "Prob. modelo", "EV"],
            ascending=[False, False, False, False],
        ).head(1)
        keep_indexes.add(best.index[0])

    suppress_mask = result_mask & ~resolved.index.isin(keep_indexes)
    resolved.loc[suppress_mask, ["Recomendacion", "Entra por"]] = ""
    resolved.loc[suppress_mask, "Prioridad"] = 0
    return resolved


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

    if outcome is None:
        return "Pendiente", None
    if outcome:
        return "✅", row["Cuota real"] - 1 if pd.notna(row["Cuota real"]) else None
    return "🔴", -1


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
    if any(result == "🔴" for result in results):
        return "🔴", -1
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

    recs = pd.DataFrame(candidates)
    if recs.empty:
        return recs
    recs["score"] = (
        recs["Prob. modelo"].fillna(0) * 0.65
        + recs["EV"].fillna(-0.05).clip(-0.2, 0.3) * 0.35
    )
    recs = resolve_result_recommendation_conflicts(recs)
    recs["Explicacion"] = recs.apply(explain_pick, axis=1)
    return recs.sort_values(["Prioridad", "score"], ascending=[False, False])


def add_unique_picks(selected, candidates, limit, label):
    used = {(item["Partido"], item["Mercado"], item["Pick"]) for item in selected}
    used_result_groups = {
        result_market_group(item)
        for item in selected
        if item["Mercado"] in {"1X2", "Doble oportunidad"}
    }
    for item in candidates.to_dict("records"):
        key = (item["Partido"], item["Mercado"], item["Pick"])
        if key in used:
            continue
        result_group = result_market_group(item)
        if item["Mercado"] in {"1X2", "Doble oportunidad"} and result_group in used_result_groups:
            continue
        item["Tipo"] = label
        selected.append(item)
        used.add(key)
        if item["Mercado"] in {"1X2", "Doble oportunidad"}:
            used_result_groups.add(result_group)
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
    goals = recs[
        recs["Mercado"].eq("Goles")
        & recs["Cuota real"].notna()
        & recs["Prob. modelo"].ge(0.62)
    ].sort_values(["Prob. modelo", "EV"], ascending=False)

    add_unique_picks(selected, safe, 5, "Top seguridad")
    add_unique_picks(selected, goals, 4, "Top +1.5")
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


def filter_recs_by_dates(recs, selected_dates):
    if recs.empty or not selected_dates:
        return recs
    filtered = recs.copy()
    filtered_dates = pd.to_datetime(filtered["Fecha partido"], errors="coerce").dt.date
    return filtered[filtered_dates.isin(selected_dates)].copy()


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


def target_return_parlay(recs, bankroll, target_return):
    if recs.empty or bankroll <= 0 or target_return <= bankroll:
        return pd.DataFrame(), {}

    target_odds = target_return / bankroll
    max_total_odds = max(target_odds * 1.8, target_odds + 4)
    valid = recs[
        recs["Cuota real"].notna()
        & recs["Cuota real"].between(1.05, 3.50)
        & recs["Prob. modelo"].ge(0.25)
        & recs["EV"].fillna(-99).ge(-0.25)
    ].copy()
    if valid.empty:
        return pd.DataFrame(), {}

    valid["target_score"] = (
        valid["Prob. modelo"].fillna(0) * 0.60
        + valid["EV"].fillna(-0.25).clip(-0.25, 0.50) * 0.30
        + valid["Prioridad"].fillna(0) * 0.025
    )
    valid = valid.sort_values(["target_score", "Prob. modelo"], ascending=False).head(20)

    best_combo = None
    best_score = None
    records = valid.to_dict("records")
    for size in range(2, min(8, len(records)) + 1):
        for combo in combinations(records, size):
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
            if summary["cuota"] < target_odds or summary["cuota"] > max_total_odds:
                continue

            excess = (summary["cuota"] - target_odds) / target_odds
            score = (
                summary["probabilidad_modelo"] * 0.65
                + max(summary["ev"], -0.5) * 0.25
                - excess * 0.10
            )
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo_df

    if best_combo is None:
        return pd.DataFrame(), {}

    summary = summarize_parlay(best_combo)
    summary["objetivo_cuota"] = target_odds
    summary["retorno_estimado"] = bankroll * summary["cuota"]
    return best_combo, summary


def show_parlay(parlay, summary):
    if parlay.empty:
        st.info("No encontre una combinada que cumpla esos filtros.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Cuota combinada", dec(summary["cuota"]))
    c2.metric("Confianza estimada", pct(summary["probabilidad_modelo"]))
    c3.metric("Balance esperado", "Positivo" if pd.notna(summary["ev"]) and summary["ev"] > 0 else "Ajustado")
    status, return_1u = settle_parlay(parlay)
    if status != "Pendiente":
        st.metric("Resultado real", status, f"{return_1u:+.2f}u")
    elif "Resultado" in parlay.columns and parlay["Resultado"].ne("Pendiente").any():
        st.metric("Resultado real", "Pendiente", "hay picks sin resultado")
    display = friendly_pick_frame(parlay, include_result=True, include_reason=True)
    st.dataframe(
        friendly_pick_style(display),
        width="stretch",
        hide_index=True,
    )


st.title("FutData")
st.caption("Una vista simple para elegir picks y combinadas sin perderse en datos tecnicos.")

with st.sidebar:
    st.header("Filtros")
    selected_leagues = st.multiselect(
        "Ligas",
        options=list(LEAGUES.keys()),
        default=list(LEAGUES.keys()),
    )
    date_range = st.date_input(
        "Fechas",
        value=(datetime(2026, 5, 1).date(), datetime(2026, 5, 4).date()),
    )
    future_only = st.checkbox("Mostrar solo partidos pendientes", value=False)
    st.divider()
    with st.expander("Actualizar cuotas", expanded=False):
        source_url = st.text_area("URL de referencia", value=DEFAULT_SOURCE_URL, height=90)
        scrape_limit = st.slider("Partidos a revisar", min_value=1, max_value=10, value=10)
        if st.button("Buscar cuotas nuevas"):
            with st.spinner("Actualizando cuotas..."):
                odds_scraped, failures = scrape_odds(source_url, scrape_limit, 35)
            st.success(f"Cuotas actualizadas: {len(odds_scraped)} partidos")
            if not failures.empty:
                st.warning(f"No se pudieron leer {len(failures)} paginas.")

all_data = load_future_predictions_all()
if all_data.empty:
    st.error("No hay predicciones multi-liga generadas. Ejecuta los pipelines/predicciones primero.")
    st.stop()

data = all_data[all_data["Liga"].isin(selected_leagues)].copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    match_dates = pd.to_datetime(data["date"], errors="coerce").dt.date
    data = data[match_dates.between(start_date, end_date)].copy()
if future_only:
    today = datetime.now().date()
    data = data[(pd.to_datetime(data["date"]).dt.date >= today) | data["Target"].isna()].copy()

odds = load_all_odds(odds_cache_signature())
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
col_b.metric("Partidos con cuota", rows_with_odds)
col_c.metric("Picks sugeridos", int(recs["Recomendacion"].ne("").sum()) if not recs.empty else 0)
if rows_with_odds == 0:
    st.warning(
        "No hay cuotas reales para los partidos filtrados. "
        "Las combinadas pueden usar cuotas aproximadas, pero conviene actualizar cuotas antes de apostar."
    )
if not odds.empty:
    visible_pairs = set(zip(data["home_key"], data["away_key"])) if not data.empty else set()
    odds_pairs = odds.assign(pair=list(zip(odds["home_key"], odds["away_key"])))
    unmatched_odds = odds_pairs[~odds_pairs["pair"].isin(visible_pairs)].copy()
    if not unmatched_odds.empty:
        with st.expander("Cuotas encontradas que no coinciden con un partido"):
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
                width="stretch",
                hide_index=True,
            )

with st.sidebar:
    if st.button("Guardar seleccion actual"):
        saved_path = save_history(0, parlay, high_parlay)
        if saved_path:
            st.success(f"Historial guardado: {saved_path.name}")
        else:
            st.warning("No habia picks para guardar.")

tab_best, tab_match, tab_parlays, tab_builder, tab_history, tab_matches = st.tabs(
    [
        "Picks sugeridos",
        "Por partido",
        "Combinadas",
        "Armar combinada",
        "Historial",
        "Partidos",
    ]
)

with tab_best:
    st.subheader("Picks sugeridos")
    st.caption(
        "Separados entre opciones mas seguras, cuotas interesantes y picks mas arriesgados."
    )
    top_recs = select_general_recommendations(recs)
    if top_recs.empty:
        st.info("Todavia no hay picks claros con los filtros actuales. Prueba actualizar cuotas o ampliar ligas/fechas.")
    else:
        display = friendly_pick_frame(top_recs, include_group=True, include_result=True, include_reason=True)
        st.dataframe(
            friendly_pick_style(display),
            width="stretch",
            hide_index=True,
        )

with tab_match:
    st.subheader("Mirar partido por partido")
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
                st.info("Sin cuota o sin pick claro para este partido.")
            else:
                display = friendly_pick_frame(shown, include_result=True, include_reason=True)
                st.dataframe(
                    friendly_pick_style(display),
                    width="stretch",
                    hide_index=True,
                )

with tab_parlays:
    if parlay_recs.empty or "Fecha partido" not in parlay_recs.columns:
        parlay_dates = []
    else:
        parlay_dates = sorted(pd.to_datetime(parlay_recs["Fecha partido"], errors="coerce").dt.date.dropna().unique())
    default_parlay_dates = []
    if parlay_dates:
        today = datetime.now().date()
        default_parlay_dates = [today] if today in parlay_dates else [parlay_dates[0]]
    selected_parlay_dates = st.multiselect(
        "Fechas para jugar",
        options=parlay_dates,
        default=default_parlay_dates,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
    )
    suggested_recs = filter_recs_by_dates(parlay_recs, selected_parlay_dates)
    suggested_low_parlay, suggested_low_summary = low_risk_parlay(suggested_recs, DEFAULT_RULES)
    suggested_high_parlay, suggested_high_summary = high_risk_parlay(suggested_recs, DEFAULT_RULES)

    st.subheader("Combinada prudente")
    st.caption(
        "Busca una cuota moderada con picks de buena confianza."
    )
    show_parlay(suggested_low_parlay, suggested_low_summary)

    st.subheader("Combinada mas ambiciosa")
    st.caption(
        "Apunta a pagar mas, aceptando que la probabilidad baja."
    )
    show_parlay(suggested_high_parlay, suggested_high_summary)

with tab_builder:
    st.subheader("Armar combinada")
    col1, col2 = st.columns(2)
    with col1:
        leg_range = st.slider(
            "Cuantos picks incluir",
            min_value=2,
            max_value=6,
            value=DEFAULT_RULES["custom_legs"],
        )
        total_odds_range = st.slider(
            "Cuota total que quieres",
            min_value=1.2,
            max_value=20.0,
            value=DEFAULT_RULES["custom_total_odds"],
            step=0.1,
        )
    with col2:
        pick_odds_range = st.slider(
            "Cuota de cada pick",
            min_value=1.01,
            max_value=5.0,
            value=DEFAULT_RULES["custom_pick_odds"],
            step=0.01,
        )
        st.caption("Puede mezclar mercados distintos del mismo partido, pero no resultados contradictorios.")

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

    st.divider()
    st.subheader("Objetivo de dinero")
    st.caption("Ejemplo: tengo 5.000 y quiero llegar a 50.000. La app busca una combinada que alcance esa cuota.")
    target_col1, target_col2 = st.columns(2)
    with target_col1:
        bankroll = st.number_input(
            "Tengo",
            min_value=1.0,
            value=5000.0,
            step=500.0,
        )
        target_dates = st.multiselect(
            "Fechas objetivo",
            options=parlay_dates,
            default=default_parlay_dates,
            format_func=lambda value: value.strftime("%Y-%m-%d"),
            key="target_dates",
        )
    with target_col2:
        target_amount = st.number_input(
            "Quiero recibir",
            min_value=1.0,
            value=50000.0,
            step=1000.0,
        )
        target_multiple = target_amount / bankroll if bankroll else pd.NA
        st.metric("Necesitas multiplicar por", dec(target_multiple))

    target_recs = filter_recs_by_dates(parlay_recs, target_dates)
    target_key = (
        float(bankroll),
        float(target_amount),
        tuple(str(date) for date in target_dates),
        tuple(selected_leagues),
    )
    if st.button("Buscar combinada objetivo"):
        with st.spinner("Buscando una combinada para tu objetivo..."):
            target_parlay, target_summary = target_return_parlay(target_recs, bankroll, target_amount)
        st.session_state["target_return_result"] = {
            "key": target_key,
            "parlay": target_parlay,
            "summary": target_summary,
        }

    target_result = st.session_state.get("target_return_result")
    if target_result and target_result.get("key") == target_key:
        target_parlay = target_result["parlay"]
        target_summary = target_result["summary"]
        if target_summary:
            st.caption(
                f"Cuota minima necesaria: {dec(target_summary['objetivo_cuota'])}. "
                f"Pago aproximado con esta combinada: {dec(target_summary['retorno_estimado'])}."
            )
        show_parlay(target_parlay, target_summary)
    else:
        st.info("Elige el monto, la meta y las fechas. Luego pulsa buscar.")

with tab_history:
    st.subheader("Historial")
    st.caption(
        "Resumen simple de combinadas guardadas y su resultado cuando el partido ya termino."
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
        c1.metric("Combinadas cerradas", current_summary["apuestas"])
        c2.metric("Ganadas", current_summary["aciertos"])
        c3.metric("Pago total", dec(current_summary["retorno"]))
        c4.metric("Rendimiento", pct(current_summary["roi"]))
        current_display = current.rename(
            columns={
                "tipo": "Tipo",
                "Stake u": "Monto base",
                "Cuota real": "Cuota",
                "Prob. modelo": "Confianza",
                "Retorno 1u": "Pago por 1u",
                "Retorno stake": "Pago total",
            }
        )
        st.dataframe(
            current_display[
                ["Tipo", "Monto base", "Pick", "Cuota", "Confianza", "Resultado", "Pago por 1u", "Pago total"]
            ].style.format(
                {
                    "Monto base": dec,
                    "Cuota": dec,
                    "Confianza": pct,
                    "Pago por 1u": dec,
                    "Pago total": dec,
                }
            ),
            width="stretch",
            hide_index=True,
        )

    history = load_history()
    if history.empty:
        st.caption("Aun no hay guardados. Usa el boton de la barra lateral para guardar la seleccion actual.")
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
            summary_display = summary.rename(
                columns={
                    "matchweek": "Fecha guardada",
                    "tipo": "Tipo",
                    "apuestas": "Combinadas",
                    "aciertos": "Ganadas",
                    "stake": "Monto base",
                    "retorno": "Pago total",
                    "roi": "Rendimiento",
                }
            )
            st.dataframe(
                summary_display.style.format({"Monto base": dec, "Pago total": dec, "Rendimiento": pct}),
                width="stretch",
                hide_index=True,
            )
        with st.expander("Detalle guardado"):
            history_display = history.tail(200).copy()
            history_display["Mercado"] = history_display["Mercado"].apply(friendly_market)
            history_display["Recomendacion"] = history_display["Recomendacion"].apply(friendly_recommendation)
            history_display = history_display.rename(
                columns={
                    "matchweek": "Fecha guardada",
                    "tipo": "Tipo",
                    "Fecha partido": "Fecha",
                    "Prob. modelo": "Confianza",
                    "Cuota real": "Cuota",
                    "Recomendacion": "Lectura",
                    "Retorno 1u": "Pago por 1u",
                    "Stake u": "Monto base",
                    "Retorno stake": "Pago total",
                }
            )
            history_cols = [
                "Fecha guardada",
                "Tipo",
                "Fecha",
                "Partido",
                "Mercado",
                "Pick",
                "Confianza",
                "Cuota",
                "Lectura",
                "Resultado",
                "Pago por 1u",
                "Monto base",
                "Pago total",
            ]
            history_cols = [col for col in history_cols if col in history_display.columns]
            st.dataframe(
                history_display[history_cols].style.format(
                    {
                        "Fecha": lambda value: pd.to_datetime(value).strftime("%Y-%m-%d") if pd.notna(value) else "",
                        "Confianza": pct,
                        "Cuota": dec,
                        "Pago por 1u": dec,
                        "Monto base": dec,
                        "Pago total": dec,
                    }
                ),
                width="stretch",
                hide_index=True,
            )

with tab_matches:
    st.subheader("Partidos")
    st.caption("Vista general de los partidos cargados. Las probabilidades estan expresadas como confianza estimada.")
    display_cols = [
        "Liga",
        "date",
        "local_team",
        "away_team",
        "Marcador",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "p_over_15",
        "Target",
        "target_over_15",
        "decimal_home_win",
        "decimal_draw",
        "decimal_away_win",
        "decimal_over_15",
        "odds_source_file",
    ]
    available_cols = [col for col in display_cols if col in data.columns]
    matches_display = data[available_cols].rename(
        columns={
            "date": "Fecha",
            "local_team": "Local",
            "away_team": "Visita",
            "Marcador": "Marcador",
            "p_home_win": "Gana local",
            "p_draw": "Empate",
            "p_away_win": "Gana visita",
            "p_over_15": "Mas de 1.5 goles",
            "decimal_home_win": "Cuota local",
            "decimal_draw": "Cuota empate",
            "decimal_away_win": "Cuota visita",
            "decimal_over_15": "Cuota +1.5",
            "odds_source_file": "Fuente cuotas",
        }
    )
    simple_cols = [
        "Liga",
        "Fecha",
        "Local",
        "Visita",
        "Marcador",
        "Gana local",
        "Empate",
        "Gana visita",
        "Mas de 1.5 goles",
        "Cuota local",
        "Cuota empate",
        "Cuota visita",
        "Cuota +1.5",
    ]
    simple_cols = [col for col in simple_cols if col in matches_display.columns]
    st.dataframe(
        matches_display[simple_cols].style.format(
            {
                "Fecha": lambda value: pd.to_datetime(value).strftime("%Y-%m-%d") if pd.notna(value) else "",
                "Gana local": pct,
                "Empate": pct,
                "Gana visita": pct,
                "Mas de 1.5 goles": pct,
                "Mas de 2.5 goles": pct,
                "Ambos marcan": pct,
                "Cuota local": dec,
                "Cuota empate": dec,
                "Cuota visita": dec,
                "Cuota +1.5": dec,
                "Cuota +2.5": dec,
                "Cuota ambos": dec,
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.caption(
    "Las cuotas pueden cambiar. Usa esto como apoyo para decidir, no como garantia de resultado."
)
