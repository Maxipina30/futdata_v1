import os
import re
from itertools import combinations

import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
ODDS_DIR = os.path.join(BASE_DIR, "files", "06_odds")
os.makedirs(ODDS_DIR, exist_ok=True)

MATCHWEEK_OBJETIVO = int(os.getenv("FUTDATA_MATCHWEEK_OBJETIVO", "34"))
PREDICTIONS_PATH = os.path.join(REPORT_DIR, f"predicciones_matchweek{MATCHWEEK_OBJETIVO}.csv")
ODDS_INPUT_PATH = os.getenv(
    "FUTDATA_ODDS_CSV",
    os.path.join(ODDS_DIR, f"cuotas_matchweek{MATCHWEEK_OBJETIVO}.csv"),
)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
REGIONS = os.getenv("ODDS_REGIONS", "eu,uk")

OUT_RECOMMENDATIONS = os.path.join(
    ODDS_DIR,
    f"recomendaciones_apuestas_matchweek{MATCHWEEK_OBJETIVO}.csv",
)
OUT_PARLAY = os.path.join(
    ODDS_DIR,
    f"recomendacion_combinada_matchweek{MATCHWEEK_OBJETIVO}.csv",
)

TEAM_ALIASES = {
    "manchester united": "manchester utd",
    "man united": "manchester utd",
    "man utd": "manchester utd",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "tottenham hotspur": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolves",
    "wolverhampton wanderers": "wolves",
    "west ham": "west ham united",
    "west ham united": "west ham united",
    "newcastle": "newcastle united",
    "newcastle united": "newcastle united",
    "nottingham forest": "nottingham forest",
    "leeds": "leeds united",
    "leeds united": "leeds united",
    "brighton": "brighton",
    "brighton and hove albion": "brighton",
    "brighton & hove albion": "brighton",
}


def normalize_team(name):
    if not isinstance(name, str):
        return ""
    text = name.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)


def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"No existe {PREDICTIONS_PATH}. Ejecuta primero src/05_predict_upcoming.py"
        )
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    predictions["home_key"] = predictions["local_team"].apply(normalize_team)
    predictions["away_key"] = predictions["away_team"].apply(normalize_team)
    return predictions


def create_odds_template(predictions):
    template = predictions[["date", "round_num", "local_team", "away_team"]].copy()
    template["bookmaker"] = ""
    template["decimal_home_win"] = ""
    template["decimal_draw"] = ""
    template["decimal_away_win"] = ""
    template["decimal_home_or_draw"] = ""
    template["decimal_home_or_away"] = ""
    template["decimal_draw_or_away"] = ""
    template.to_csv(ODDS_INPUT_PATH, index=False)
    print(f"No encontré cuotas. Plantilla creada en: {ODDS_INPUT_PATH}")
    print("Rellena cuotas decimales y vuelve a ejecutar este script.")


def fetch_odds_from_api(predictions):
    if not ODDS_API_KEY:
        return pd.DataFrame()

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    response = requests.get(ODDS_API_URL, params=params, timeout=30)
    response.raise_for_status()

    rows = []
    prediction_pairs = {
        tuple(sorted((row.home_key, row.away_key)))
        for row in predictions.itertuples(index=False)
    }

    for event in response.json():
        home_key = normalize_team(event.get("home_team"))
        away_key = normalize_team(event.get("away_team"))
        if tuple(sorted((home_key, away_key))) not in prediction_pairs:
            continue

        for bookmaker in event.get("bookmakers", []):
            market = next(
                (
                    item for item in bookmaker.get("markets", [])
                    if item.get("key") == "h2h"
                ),
                None,
            )
            if not market:
                continue
            prices = {
                normalize_team(outcome.get("name")): outcome.get("price")
                for outcome in market.get("outcomes", [])
            }
            rows.append(
                {
                    "date": event.get("commence_time"),
                    "local_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": bookmaker.get("title"),
                    "decimal_home_win": prices.get(home_key),
                    "decimal_draw": prices.get("draw"),
                    "decimal_away_win": prices.get(away_key),
                    "decimal_home_or_draw": None,
                    "decimal_home_or_away": None,
                    "decimal_draw_or_away": None,
                }
            )

    odds = pd.DataFrame(rows)
    if not odds.empty:
        odds.to_csv(ODDS_INPUT_PATH, index=False)
        print(f"Cuotas API guardadas en: {ODDS_INPUT_PATH}")
    return odds


def load_or_fetch_odds(predictions):
    if os.path.exists(ODDS_INPUT_PATH):
        return pd.read_csv(ODDS_INPUT_PATH)

    odds = fetch_odds_from_api(predictions)
    if odds.empty:
        create_odds_template(predictions)
    return odds


def best_price(row_group, col):
    prices = pd.to_numeric(row_group[col], errors="coerce").dropna()
    return prices.max() if not prices.empty else None


def collapse_best_odds(odds):
    odds = odds.copy()
    odds["home_key"] = odds["local_team"].apply(normalize_team)
    odds["away_key"] = odds["away_team"].apply(normalize_team)
    rows = []
    for (home_key, away_key), group in odds.groupby(["home_key", "away_key"]):
        rows.append(
            {
                "home_key": home_key,
                "away_key": away_key,
                "best_home_win": best_price(group, "decimal_home_win"),
                "best_draw": best_price(group, "decimal_draw"),
                "best_away_win": best_price(group, "decimal_away_win"),
                "best_home_or_draw": best_price(group, "decimal_home_or_draw"),
                "best_home_or_away": best_price(group, "decimal_home_or_away"),
                "best_draw_or_away": best_price(group, "decimal_draw_or_away"),
            }
        )
    return pd.DataFrame(rows)


def add_value_metrics(row, outcome, probability, odds):
    if pd.isna(odds) or odds <= 1:
        return None
    implied = 1 / odds
    edge = probability - implied
    expected_value = probability * odds - 1
    b = odds - 1
    kelly = ((b * probability) - (1 - probability)) / b if b > 0 else 0
    conservative_stake_units = max(0, min(kelly * 0.25, 0.03)) * 100
    return {
        "date": row["date"],
        "round_num": row["round_num"],
        "local_team": row["local_team"],
        "away_team": row["away_team"],
        "market": outcome,
        "model_probability": probability,
        "decimal_odds": odds,
        "implied_probability": implied,
        "edge": edge,
        "expected_value": expected_value,
        "kelly_fraction": kelly,
        "stake_units_100_bankroll": conservative_stake_units,
    }


def recommendation_type(probability, expected_value, market):
    if probability >= 0.70 and expected_value >= -0.03:
        return "fuerte_por_probabilidad"
    if expected_value >= 0.08 and probability >= 0.35:
        return "valor_simple"
    if "doble oportunidad" in market and probability >= 0.68 and expected_value >= -0.02:
        return "doble_oportunidad_riesgo_bajo"
    return ""


def build_recommendations(predictions, odds):
    best_odds = collapse_best_odds(odds)
    data = predictions.merge(best_odds, on=["home_key", "away_key"], how="left")

    candidates = []
    for _, row in data.iterrows():
        outcomes = [
            ("gana_local", row["p_home_win"], row.get("best_home_win")),
            ("empate", row["p_draw"], row.get("best_draw")),
            ("gana_visita", row["p_away_win"], row.get("best_away_win")),
            (
                "doble oportunidad local/empate",
                row["p_home_win"] + row["p_draw"],
                row.get("best_home_or_draw"),
            ),
            (
                "doble oportunidad local/visita",
                row["p_home_win"] + row["p_away_win"],
                row.get("best_home_or_away"),
            ),
            (
                "doble oportunidad empate/visita",
                row["p_draw"] + row["p_away_win"],
                row.get("best_draw_or_away"),
            ),
        ]
        for market, probability, odds_value in outcomes:
            metrics = add_value_metrics(row, market, probability, odds_value)
            if metrics:
                metrics["recommendation"] = recommendation_type(
                    probability,
                    metrics["expected_value"],
                    market,
                )
                candidates.append(metrics)

    recommendations = pd.DataFrame(candidates)
    if recommendations.empty:
        return recommendations

    recommendations = recommendations.sort_values(
        ["recommendation", "expected_value", "model_probability"],
        ascending=[False, False, False],
    )
    return recommendations


def build_parlay(recommendations):
    valid = recommendations[recommendations["recommendation"].ne("")].copy()
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values(["model_probability", "expected_value"], ascending=False)
    best_combo = None
    for size in [2, 3]:
        for combo in combinations(valid.to_dict("records"), size):
            matches = {(item["local_team"], item["away_team"]) for item in combo}
            if len(matches) != size:
                continue
            probability = 1.0
            odds = 1.0
            for item in combo:
                probability *= item["model_probability"]
                odds *= item["decimal_odds"]
            expected_value = probability * odds - 1
            score = expected_value + probability
            if best_combo is None or score > best_combo["score"]:
                best_combo = {
                    "score": score,
                    "legs": combo,
                    "combined_probability": probability,
                    "combined_odds": odds,
                    "combined_expected_value": expected_value,
                }

    if not best_combo:
        return pd.DataFrame()

    rows = []
    for leg_num, item in enumerate(best_combo["legs"], start=1):
        rows.append(
            {
                "leg": leg_num,
                "local_team": item["local_team"],
                "away_team": item["away_team"],
                "market": item["market"],
                "model_probability": item["model_probability"],
                "decimal_odds": item["decimal_odds"],
                "combined_probability": best_combo["combined_probability"],
                "combined_odds": best_combo["combined_odds"],
                "combined_expected_value": best_combo["combined_expected_value"],
            }
        )
    return pd.DataFrame(rows)


def main():
    predictions = load_predictions()
    odds = load_or_fetch_odds(predictions)
    if odds.empty:
        return

    recommendations = build_recommendations(predictions, odds)
    if recommendations.empty:
        print("No pude cruzar cuotas con predicciones. Revisa nombres de equipos/cuotas.")
        return

    recommendations.to_csv(OUT_RECOMMENDATIONS, index=False)
    parlay = build_parlay(recommendations)
    if not parlay.empty:
        parlay.to_csv(OUT_PARLAY, index=False)

    print(f"Recomendaciones guardadas en: {OUT_RECOMMENDATIONS}")
    print("\nMejores apuestas con valor:")
    top = recommendations[recommendations["recommendation"].ne("")].head(12)
    if top.empty:
        print("No hay apuestas con edge positivo bajo las reglas actuales.")
    else:
        print(
            top[
                [
                    "local_team", "away_team", "market", "model_probability",
                    "decimal_odds", "expected_value", "stake_units_100_bankroll",
                    "recommendation",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )

    if not parlay.empty:
        print("\nCombinada sugerida:")
        print(
            parlay[
                [
                    "local_team", "away_team", "market", "model_probability",
                    "decimal_odds", "combined_probability", "combined_odds",
                    "combined_expected_value",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )

    print("\nNota: esto no es consejo financiero; usa stake bajo y valida cuotas reales antes de apostar.")


if __name__ == "__main__":
    main()
