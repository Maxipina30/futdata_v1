import importlib
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
REPORT_DIR = BASE_DIR / "files" / "05_reports"
ODDS_DIR = BASE_DIR / "files" / "06_odds"
DEFAULT_ODDS_PATH = ODDS_DIR / "cuotasahora_matchweek35_consolidated.csv"
DEFAULT_SOURCE_URL = (
    "https://www.cuotasahora.com/football/h2h/arsenal-hA1Zm19f/"
    "newcastle-p6ahwuwJ/#OQsq6PYa:over-under;2;"
)

KNOWN_CUOTASAHORA_URLS = {
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
}


st.set_page_config(page_title="FutData Premier League", layout="wide")


def normalize_team(name):
    if not isinstance(name, str):
        return ""
    text = name.lower().replace("&", " and ")
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
        pred = pred.merge(
            goals[
                [
                    "date",
                    "round_num",
                    "local_team",
                    "away_team",
                    "p_over_15",
                    "p_over_25",
                    "p_btts",
                ]
            ],
            on=["date", "round_num", "local_team", "away_team"],
            how="left",
        )
    pred["home_key"] = pred["local_team"].apply(normalize_team)
    pred["away_key"] = pred["away_team"].apply(normalize_team)
    return pred, ""


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
    odds.to_csv(DEFAULT_ODDS_PATH, index=False)
    if not failures.empty:
        failures.to_csv(ODDS_DIR / "cuotasahora_scrape_failures.csv", index=False)
    st.cache_data.clear()
    return odds, failures


def add_market(candidates, row, market, probability, odds_col, label):
    odds = row.get(odds_col)
    fair_odds = 1 / probability if probability and probability > 0 else None
    implied = 1 / odds if pd.notna(odds) and odds and odds > 1 else None
    edge = probability - implied if implied is not None else None
    expected_value = probability * odds - 1 if pd.notna(odds) and odds and odds > 1 else None

    recommendation_rank = 0
    if market.startswith("Doble oportunidad") and probability >= 0.80 and (
        expected_value is None or expected_value >= -0.08
    ):
        recommendation = "Bajo riesgo"
        recommendation_rank = 4
    elif probability >= 0.85:
        recommendation = "Alta probabilidad"
        recommendation_rank = 3
    elif (label == "Over 1.5" or market == "Ambos anotan") and probability >= 0.75 and (
        expected_value is None or expected_value >= -0.08
    ):
        recommendation = "Bajo riesgo"
        recommendation_rank = 4
    elif probability >= 0.72 and (expected_value is None or expected_value >= -0.05):
        recommendation = "Alta probabilidad"
        recommendation_rank = 3
    elif expected_value is not None and expected_value >= 0.15 and probability >= 0.55:
        recommendation = "Valor"
        recommendation_rank = 2
    elif expected_value is not None and expected_value >= 0.12 and probability >= 0.40:
        recommendation = "Valor con riesgo"
        recommendation_rank = 1
    else:
        recommendation = ""

    candidates.append(
        {
            "Fecha": int(row["round_num"]),
            "Partido": f"{row['local_team']} vs {row['away_team']}",
            "Mercado": market,
            "Pick": label,
            "Prob. modelo": probability,
            "Cuota real": odds,
            "Cuota justa": fair_odds,
            "Edge": edge,
            "EV": expected_value,
            "Recomendacion": recommendation,
            "Prioridad": recommendation_rank,
            "Tiene cuota": pd.notna(odds),
        }
    )


def build_recommendations(data):
    candidates = []
    for _, row in data.iterrows():
        add_market(candidates, row, "1X2", row["p_home_win"], "decimal_home_win", row["local_team"])
        add_market(candidates, row, "1X2", row["p_away_win"], "decimal_away_win", row["away_team"])
        add_market(
            candidates,
            row,
            "Doble oportunidad",
            row["p_home_win"] + row["p_draw"],
            "decimal_home_or_draw",
            f"{row['local_team']} o empate",
        )
        add_market(
            candidates,
            row,
            "Doble oportunidad",
            row["p_draw"] + row["p_away_win"],
            "decimal_draw_or_away",
            f"Empate o {row['away_team']}",
        )
        if "p_over_15" in row and pd.notna(row["p_over_15"]):
            add_market(candidates, row, "Goles", row["p_over_15"], "decimal_over_15", "Over 1.5")
        if "p_over_25" in row and pd.notna(row["p_over_25"]):
            add_market(candidates, row, "Goles", row["p_over_25"], "decimal_over_25", "Over 2.5")
        if "p_btts" in row and pd.notna(row["p_btts"]):
            add_market(candidates, row, "Ambos anotan", row["p_btts"], "decimal_btts_yes", "Si")

    recs = pd.DataFrame(candidates)
    if recs.empty:
        return recs
    recs["score"] = (
        recs["Prob. modelo"].fillna(0) * 0.65
        + recs["EV"].fillna(-0.05).clip(-0.2, 0.3) * 0.35
    )
    return recs.sort_values(["Prioridad", "score"], ascending=[False, False])


def low_risk_parlay(recs):
    if recs.empty:
        return pd.DataFrame(), {}

    locked = recs[
        recs["Cuota real"].notna()
        & recs["Prob. modelo"].ge(0.85)
        & recs["Cuota real"].between(1.01, 1.70)
    ].copy()
    locked = locked.sort_values(["Prob. modelo", "Cuota real"], ascending=[False, False])

    legs = []
    used_market_pairs = set()
    for _, row in locked.iterrows():
        market_key = "Resultado" if row["Mercado"] in {"1X2", "Doble oportunidad"} else row["Mercado"]
        pair = (row["Partido"], market_key)
        if pair in used_market_pairs:
            continue
        legs.append(row)
        used_market_pairs.add(pair)

    remaining = recs[
        recs["Recomendacion"].isin(["Bajo riesgo", "Alta probabilidad", "Valor"])
        & recs["Cuota real"].notna()
        & recs["Cuota real"].between(1.05, 1.60)
        & recs["Prob. modelo"].ge(0.68)
        & recs["EV"].fillna(-99).ge(-0.08)
    ].copy()
    remaining = remaining.sort_values(["Prob. modelo", "EV"], ascending=False)

    for _, row in remaining.iterrows():
        market_key = "Resultado" if row["Mercado"] in {"1X2", "Doble oportunidad"} else row["Mercado"]
        pair = (row["Partido"], market_key)
        if pair in used_market_pairs:
            continue
        legs.append(row)
        used_market_pairs.add(pair)
        if len(legs) >= 4:
            break

    if len(legs) < 2:
        return pd.DataFrame(), {}
    parlay = pd.DataFrame(legs)
    return parlay, summarize_parlay(parlay)


def high_risk_parlay(recs):
    return find_best_parlay(
        recs,
        min_legs=5,
        max_legs=5,
        min_total_odds=5.0,
        max_total_odds=10.0,
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

    valid = valid.sort_values(["EV", "Prob. modelo"], ascending=False).head(30)
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
    st.dataframe(
        parlay[["Partido", "Mercado", "Pick", "Prob. modelo", "Cuota real", "EV"]].style.format(
            {"Prob. modelo": pct, "Cuota real": dec, "EV": pct}
        ),
        use_container_width=True,
        hide_index=True,
    )


st.title("Premier League: recomendaciones por fecha")

with st.sidebar:
    st.header("Filtros")
    matchweek = st.number_input("Fecha", min_value=1, max_value=38, value=35, step=1)
    odds_path = st.text_input("CSV de cuotas", value=str(DEFAULT_ODDS_PATH))
    st.divider()
    st.header("Scraper beta")
    source_url = st.text_area("URL semilla CuotasAhora", value=DEFAULT_SOURCE_URL, height=90)
    scrape_limit = st.slider("Partidos a probar", min_value=1, max_value=10, value=10)
    if st.button("Actualizar cuotas reales"):
        with st.spinner("Scrapeando y decodificando cuotas..."):
            odds_scraped, failures = scrape_odds(source_url, scrape_limit, matchweek)
        st.success(f"Cuotas actualizadas: {len(odds_scraped)} partidos")
        if not failures.empty:
            st.warning(f"{len(failures)} URLs fallaron; quedaron guardadas en failures.")

pred, error = load_predictions(matchweek)
if error:
    st.error(error)
    st.stop()

odds = load_odds(odds_path)
data = pred.merge(
    odds.drop(columns=["local_team", "away_team"], errors="ignore"),
    on=["home_key", "away_key"],
    how="left",
)
recs = build_recommendations(data)
parlay, parlay_summary = low_risk_parlay(recs)
high_parlay, high_parlay_summary = high_risk_parlay(recs)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Partidos fecha", len(pred))
col_b.metric("Con cuotas cruzadas", int(data["decimal_home_win"].notna().sum()) if "decimal_home_win" in data else 0)
col_c.metric("Recomendaciones", int(recs["Recomendacion"].ne("").sum()) if not recs.empty else 0)

tab_best, tab_match, tab_parlays, tab_builder, tab_matches = st.tabs(
    [
        "Mejores recomendaciones",
        "Por partido",
        "Combinadas sugeridas",
        "Crea tu combinada",
        "Todos los partidos",
    ]
)

with tab_best:
    st.subheader("Mejores recomendaciones")
    st.caption(
        "Criterio: Bajo riesgo prioriza probabilidad alta; Valor exige cuota claramente superior "
        "a la cuota justa del modelo; Valor con riesgo paga bien, pero no es una jugada conservadora."
    )
    top_recs = recs[recs["Recomendacion"].ne("")].head(18).copy()
    if top_recs.empty:
        st.info("Todavia no hay recomendaciones con las reglas actuales. Revisa cuotas o baja los umbrales.")
    else:
        st.dataframe(
            top_recs[
                [
                    "Partido",
                    "Mercado",
                    "Pick",
                    "Prob. modelo",
                    "Cuota real",
                    "Cuota justa",
                    "Edge",
                    "EV",
                    "Recomendacion",
                ]
            ].style.format(
                {
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
    for match_name, match_recs in recs.groupby("Partido", sort=False):
        shown = match_recs[
            match_recs["Tiene cuota"] | match_recs["Recomendacion"].ne("")
        ].sort_values(["Prioridad", "score"], ascending=[False, False]).head(8)
        with st.expander(match_name):
            if shown.empty:
                st.info("Sin cuotas cruzadas o sin recomendacion clara para este partido.")
            else:
                st.dataframe(
                    shown[
                        [
                            "Mercado",
                            "Pick",
                            "Prob. modelo",
                            "Cuota real",
                            "Cuota justa",
                            "Edge",
                            "EV",
                            "Recomendacion",
                        ]
                    ].style.format(
                        {
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
    show_parlay(parlay, parlay_summary)

    st.subheader("Combinada de riesgo alto moderado")
    st.caption("Busca una cuota total entre 5 y 10, evitando combinadas exageradas.")
    show_parlay(high_parlay, high_parlay_summary)

with tab_builder:
    st.subheader("Crea tu combinada")
    col1, col2 = st.columns(2)
    with col1:
        leg_range = st.slider("Cantidad de picks", min_value=2, max_value=6, value=(3, 5))
        total_odds_range = st.slider("Cuota total buscada", min_value=1.2, max_value=20.0, value=(2.0, 10.0), step=0.1)
    with col2:
        pick_odds_range = st.slider("Cuota por pick", min_value=1.01, max_value=5.0, value=(1.10, 1.50), step=0.01)
        st.caption("Puede usar mas de un pick del mismo partido si son mercados distintos.")

    custom_parlay, custom_summary = find_best_parlay(
        recs,
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

with tab_matches:
    st.subheader("Todos los partidos")
    display_cols = [
        "date",
        "local_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "p_over_15",
        "p_over_25",
        "p_btts",
        "decimal_home_win",
        "decimal_draw",
        "decimal_away_win",
        "decimal_over_15",
        "decimal_over_25",
        "decimal_btts_yes",
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
