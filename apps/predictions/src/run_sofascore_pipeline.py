import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = BASE_DIR / "files" / "sofascore_pipeline"
ROLLING_WINDOWS = [3, 5]
MIN_HISTORY = 3
ROLLING_METRICS = [
    "gf",
    "ga",
    "poss",
    "sh",
    "sot",
    "sot_pct",
    "xg",
    "xga",
    "big_chances",
    "big_chances_allowed",
    "big_chances_scored",
    "big_chances_scored_allowed",
    "big_chances_missed",
    "big_chances_missed_allowed",
    "corner_kicks",
    "corner_kicks_allowed",
    "fouls",
    "fouls_allowed",
    "passes",
    "passes_allowed",
    "accurate_passes",
    "accurate_passes_allowed",
    "pass_accuracy_pct",
    "pass_accuracy_pct_allowed",
    "tackles",
    "tackles_allowed",
    "free_kicks",
    "free_kicks_allowed",
    "offsides",
    "offsides_allowed",
    "yellow_cards",
    "yellow_cards_allowed",
    "red_cards",
    "red_cards_allowed",
    "goalkeeper_saves",
    "goalkeeper_saves_allowed",
    "blocked_shots",
    "blocked_shots_allowed",
    "shots_off_target",
    "shots_off_target_allowed",
    "shots_inside_box",
    "shots_inside_box_allowed",
    "shots_outside_box",
    "shots_outside_box_allowed",
    "through_balls",
    "through_balls_allowed",
    "touches_in_box",
    "touches_in_box_allowed",
    "accurate_long_balls",
    "accurate_long_balls_allowed",
    "accurate_crosses",
    "accurate_crosses_allowed",
    "duel_win_pct",
    "duel_win_pct_allowed",
    "dispossessed",
    "dispossessed_allowed",
    "ground_duels_won",
    "ground_duels_won_allowed",
    "aerial_duels_won",
    "aerial_duels_won_allowed",
    "successful_dribbles",
    "successful_dribbles_allowed",
    "interceptions",
    "interceptions_allowed",
    "clearances",
    "clearances_allowed",
    "errors_lead_to_shot",
    "errors_lead_to_shot_allowed",
    "high_claims",
    "high_claims_allowed",
    "goal_kicks",
    "goal_kicks_allowed",
    "xg_per_sh",
    "xg_per_sh_allowed",
    "sot_per_sh",
    "sot_per_sh_allowed",
    "goals_minus_xg",
    "goals_minus_xg_allowed",
    "save_pct",
    "save_pct_allowed",
    "sh_allowed",
    "sot_allowed",
    "sot_allowed_pct",
]
TABLE_PRIOR_FEATURES = [
    "table_mp_prev",
    "table_ppg_prev",
    "table_gd_prev",
    "table_rank_pct_prev",
]
H2H_FEATURES = [
    "h2h_matches",
    "h2h_local_win_pct",
    "h2h_draw_pct",
    "h2h_away_win_pct",
    "h2h_local_goals_avg",
    "h2h_away_goals_avg",
    "h2h_total_goals_avg",
    "h2h_btts_pct",
    "h2h_over25_pct",
    "h2h_local_points_avg",
    "h2h_away_points_avg",
]

MODEL_SELECTION_FOLDS = [
    {"train_max_round": 14, "valid_min_round": 15, "valid_max_round": 20},
    {"train_max_round": 20, "valid_min_round": 21, "valid_max_round": 24},
    {"train_max_round": 24, "valid_min_round": 25, "valid_max_round": 28},
]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
K_GRID = [24, 36, 48, 72, 108, 160, "all"]
CORRELATION_THRESHOLDS = [0.85, 0.9, 0.95]
CLASS_WEIGHTS = ["balanced", None]
GOAL_C_GRID = [0.03, 0.1, 0.3]
GOAL_K_GRID = [36, 72, "all"]
GOAL_CORRELATION_THRESHOLDS = [0.9, 0.95]
GOAL_CLASS_WEIGHTS = ["balanced"]
TABLE_FEATURE_MIN_LOGLOSS_GAIN = 0.02
MODEL_SELECTION_LOGLOSS_TOLERANCE = 0.02
SELECTION_OBJECTIVES = ["calibrated_macro", "macro_f1", "balanced_accuracy", "log_loss"]
GOAL_MARKETS = {
    "over_15": "Over 1.5 goles",
    "over_25": "Over 2.5 goles",
    "btts": "Ambos anotan",
}


def normalize_cols(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("%", "pct", regex=False)
        .str.lower()
    )
    return df


class CorrelationPruner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, x_data, y=None):
        x_array = np.asarray(x_data, dtype=float)
        n_features = x_array.shape[1]
        if n_features <= 1:
            self.keep_indices_ = np.arange(n_features)
            return self

        corr = np.corrcoef(x_array, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.abs(corr), k=1)
        drop = set()
        for col_idx in range(n_features):
            if col_idx in drop:
                continue
            correlated = np.where(upper[col_idx] > self.threshold)[0]
            drop.update(int(idx) for idx in correlated)
        self.keep_indices_ = np.array(
            [idx for idx in range(n_features) if idx not in drop],
            dtype=int,
        )
        return self

    def transform(self, x_data):
        return np.asarray(x_data)[:, self.keep_indices_]


class SafeSelectKBest(SelectKBest):
    def fit(self, x_data, y):
        if self.k != "all" and self.k > x_data.shape[1]:
            self.k = "all"
        return super().fit(x_data, y)


def load_raw_matchlogs(root, league, include_unfinished=False):
    matchlogs_dir = root / "raw" / league / "team_matchlogs"
    schedule = normalize_cols(pd.read_csv(matchlogs_dir / f"{league}_team_schedule.csv"))
    shooting_for = normalize_cols(pd.read_csv(matchlogs_dir / f"{league}_team_shooting_for.csv"))
    shooting_against = normalize_cols(pd.read_csv(matchlogs_dir / f"{league}_team_shooting_against.csv"))

    schedule = schedule.rename(columns={"team": "equipo"})
    shooting_for = shooting_for[shooting_for["table_side"].eq("for")].copy()
    shooting_for = shooting_for.rename(
        columns={"team": "equipo", "sotpct": "sot_pct"}
    )
    shooting_against = shooting_against[shooting_against["table_side"].eq("against")].copy()
    shooting_against = shooting_against.rename(
        columns={
            "team": "equipo",
            "sh": "sh_allowed",
            "sot": "sot_allowed",
            "sotpct": "sot_allowed_pct",
        }
    )

    for frame in [schedule, shooting_for, shooting_against]:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    keep_for = [c for c in ["equipo", "date", "opponent", "sh", "sot", "sot_pct"] if c in shooting_for.columns]
    keep_against = [
        c
        for c in ["equipo", "date", "opponent", "sh_allowed", "sot_allowed", "sot_allowed_pct"]
        if c in shooting_against.columns
    ]
    df = schedule.merge(shooting_for[keep_for], on=["equipo", "date", "opponent"], how="left")
    df = df.merge(shooting_against[keep_against], on=["equipo", "date", "opponent"], how="left")

    if "round" in df.columns:
        df["round_num"] = pd.to_numeric(df["round"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    for col in ROLLING_METRICS + ["gf", "ga"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not include_unfinished:
        df = df[df["result"].isin(["W", "D", "L"])].copy()
    return df


def load_h2h_features(root, league):
    path = root / "raw" / league / f"{league}_h2h_features.csv"
    if not path.exists():
        return pd.DataFrame()
    features = pd.read_csv(path)
    if features.empty:
        return features
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["time"] = features["time"].astype(str).fillna("")
    return features


def add_prior_table_features(df):
    df = df.sort_values(["date", "time", "equipo"]).copy()
    teams = sorted(df["equipo"].dropna().unique())
    table = {
        team: {
            "mp": 0,
            "pts": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
        for team in teams
    }

    prior_rows = []
    for _, match_block in df.groupby(["date", "time"], sort=True, dropna=False):
        standings = pd.DataFrame(
            [
                {
                    "equipo": team,
                    "pts": values["pts"],
                    "gd": values["gd"],
                    "gf": values["gf"],
                    "wins": values["wins"],
                    "mp": values["mp"],
                }
                for team, values in table.items()
            ]
        ).sort_values(
            ["pts", "gd", "gf", "wins", "equipo"],
            ascending=[False, False, False, False, True],
        )
        ranks = {
            row.equipo: rank
            for rank, row in enumerate(standings.itertuples(index=False), start=1)
        }

        for idx, row in match_block.iterrows():
            team = row["equipo"]
            values = table[team]
            rank = ranks.get(team, np.nan)
            prior_rows.append(
                {
                    "index": idx,
                    "table_mp_prev": values["mp"],
                    "table_pts_prev": values["pts"],
                    "table_ppg_prev": values["pts"] / values["mp"] if values["mp"] else 0,
                    "table_gf_prev": values["gf"],
                    "table_ga_prev": values["ga"],
                    "table_gd_prev": values["gd"],
                    "table_rank_prev": rank,
                    "table_rank_pct_prev": rank / len(teams) if teams and not pd.isna(rank) else np.nan,
                    "table_wins_prev": values["wins"],
                    "table_draws_prev": values["draws"],
                    "table_losses_prev": values["losses"],
                }
            )

        for _, row in match_block.iterrows():
            if pd.isna(row.get("gf")) or pd.isna(row.get("ga")) or pd.isna(row.get("result")):
                continue
            team = row["equipo"]
            gf = int(row["gf"])
            ga = int(row["ga"])
            result = row["result"]
            table[team]["mp"] += 1
            table[team]["gf"] += gf
            table[team]["ga"] += ga
            table[team]["gd"] = table[team]["gf"] - table[team]["ga"]
            if result == "W":
                table[team]["pts"] += 3
                table[team]["wins"] += 1
            elif result == "D":
                table[team]["pts"] += 1
                table[team]["draws"] += 1
            elif result == "L":
                table[team]["losses"] += 1

    prior = pd.DataFrame(prior_rows).set_index("index")
    df = pd.concat([df, prior.reindex(df.index)], axis=1)
    return df.sort_index()


def add_h2h_features(match_dataset, h2h_features):
    if h2h_features is None or h2h_features.empty:
        return match_dataset
    merge_cols = ["date", "time", "local_team", "away_team"]
    h2h_cols = [col for col in h2h_features.columns if col.startswith("h2h_")]
    return match_dataset.merge(
        h2h_features[merge_cols + h2h_cols],
        on=merge_cols,
        how="left",
    )


def add_prior_rolling_features(df):
    df = df.sort_values(["equipo", "date", "time"]).copy()
    group = df.groupby("equipo", group_keys=False)
    new_cols = {}
    for window in ROLLING_WINDOWS:
        min_periods = min(window, MIN_HISTORY)
        for metric in ROLLING_METRICS:
            if metric not in df.columns:
                continue
            new_cols[f"{metric}_rolling{window}"] = group[metric].transform(
                lambda series: series.shift().rolling(window, min_periods=min_periods).mean()
            )
        new_cols[f"winrate_rolling{window}"] = group["result"].transform(
            lambda series: series.shift().eq("W").rolling(window, min_periods=min_periods).mean()
        )
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def build_match_dataset(df, h2h_features=None):
    df = add_prior_table_features(df)
    df = add_prior_rolling_features(df)
    feature_cols = [
        c
        for c in df.columns
        if c.endswith("_rolling3") or c.endswith("_rolling5") or c in TABLE_PRIOR_FEATURES
    ]
    home = df[df["venue"].str.lower().eq("home")].copy()
    away = df[df["venue"].str.lower().eq("away")].copy()

    home = home.rename(columns={c: f"{c}_home" for c in feature_cols})
    away = away.rename(columns={c: f"{c}_away" for c in feature_cols})
    merged = home.merge(
        away,
        left_on=["date", "time", "opponent"],
        right_on=["date", "time", "equipo"],
        how="inner",
        suffixes=("_home", "_away"),
    )
    merged["target"] = merged["result_home"].map({"W": "1", "D": "X", "L": "2"})

    diff_cols = {}
    model_features = []
    for col in feature_cols:
        home_col = f"{col}_home"
        away_col = f"{col}_away"
        if home_col in merged.columns and away_col in merged.columns:
            diff_col = f"diff_{col}"
            diff_cols[diff_col] = merged[home_col] - merged[away_col]
            model_features.extend([home_col, away_col, diff_col])
    if diff_cols:
        merged = pd.concat([merged, pd.DataFrame(diff_cols, index=merged.index)], axis=1)

    id_cols = [
        "date",
        "time",
        "season_id_home",
        "season_name_home",
        "season_year_home",
        "round_num_home",
        "equipo_home",
        "opponent_home",
        "gf_home",
        "ga_home",
        "target",
    ]
    dataset = merged[[c for c in id_cols + model_features if c in merged.columns]].rename(
        columns={
            "round_num_home": "round_num",
            "season_id_home": "season_id",
            "season_name_home": "season_name",
            "season_year_home": "season_year",
            "equipo_home": "local_team",
            "opponent_home": "away_team",
            "gf_home": "home_goals",
            "ga_home": "away_goals",
        }
    )
    dataset = add_h2h_features(dataset, h2h_features)
    h2h_feature_cols = [col for col in dataset.columns if col.startswith("h2h_")]
    model_features.extend(h2h_feature_cols)
    return dataset, model_features


def brier_multiclass(y_true, probabilities, classes):
    class_to_index = {klass: idx for idx, klass in enumerate(classes)}
    encoded = np.zeros_like(probabilities)
    for row_idx, klass in enumerate(y_true):
        encoded[row_idx, class_to_index[klass]] = 1
    return float(np.mean(np.sum((encoded - probabilities) ** 2, axis=1)))


def evaluate_model(model, data, feature_cols):
    y_true = data["target"]
    preds = model.predict(data[feature_cols])
    probabilities = model.predict_proba(data[feature_cols])
    classes = model.classes_.tolist() if hasattr(model, "classes_") else model.named_steps["logreg"].classes_.tolist()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "brier_multiclass": brier_multiclass(y_true.to_numpy(), probabilities, classes),
    }


def safe_binary_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, probabilities))


def evaluate_binary_model(model, data, feature_cols, target_col):
    y_true = data[target_col]
    preds = model.predict(data[feature_cols])
    probabilities = model.predict_proba(data[feature_cols])[:, 1]
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": safe_binary_auc(y_true, probabilities),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def build_model(c_value, k_value, corr_threshold, class_weight):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=1e-9)),
            ("corr", CorrelationPruner(threshold=corr_threshold)),
            ("scaler", StandardScaler()),
            ("select", SafeSelectKBest(score_func=f_classif, k=k_value)),
            (
                "logreg",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def model_selection(train, feature_cols, feature_set_name="all", selection_objective="calibrated_macro"):
    rows = []
    for corr_threshold in CORRELATION_THRESHOLDS:
        for k_value in K_GRID:
            for c_value in C_GRID:
                for class_weight in CLASS_WEIGHTS:
                    fold_scores = []
                    for fold in MODEL_SELECTION_FOLDS:
                        train_core = train[train["round_num"] <= fold["train_max_round"]].copy()
                        valid = train[
                            train["round_num"].between(
                                fold["valid_min_round"],
                                fold["valid_max_round"],
                            )
                        ].copy()
                        if train_core.empty or valid.empty or train_core["target"].nunique() < 3:
                            continue

                        model = build_model(c_value, k_value, corr_threshold, class_weight)
                        model.fit(train_core[feature_cols], train_core["target"])
                        fold_scores.append(evaluate_model(model, valid, feature_cols))

                    if not fold_scores:
                        continue

                    row = {
                        "feature_set": feature_set_name,
                        "corr_threshold": corr_threshold,
                        "k": k_value,
                        "C": c_value,
                        "class_weight": "balanced" if class_weight == "balanced" else "none",
                        "folds": len(fold_scores),
                    }
                    for metric in fold_scores[0]:
                        row[metric] = float(np.mean([score[metric] for score in fold_scores]))
                        row[f"{metric}_std"] = float(np.std([score[metric] for score in fold_scores]))
                    rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No se pudo seleccionar modelo: no hubo folds válidos.")
    return rank_model_selection(results, selection_objective)


def binary_model_selection(train, feature_cols, target_col, feature_set_name="all"):
    rows = []
    for corr_threshold in GOAL_CORRELATION_THRESHOLDS:
        for k_value in GOAL_K_GRID:
            for c_value in GOAL_C_GRID:
                for class_weight in GOAL_CLASS_WEIGHTS:
                    fold_scores = []
                    for fold in MODEL_SELECTION_FOLDS:
                        train_core = train[train["round_num"] <= fold["train_max_round"]].copy()
                        valid = train[
                            train["round_num"].between(
                                fold["valid_min_round"],
                                fold["valid_max_round"],
                            )
                        ].copy()
                        if train_core.empty or valid.empty:
                            continue
                        if train_core[target_col].nunique() < 2 or valid[target_col].nunique() < 2:
                            continue

                        model = build_model(c_value, k_value, corr_threshold, class_weight)
                        model.fit(train_core[feature_cols], train_core[target_col])
                        fold_scores.append(
                            evaluate_binary_model(model, valid, feature_cols, target_col)
                        )

                    if not fold_scores:
                        continue

                    row = {
                        "target": target_col,
                        "feature_set": feature_set_name,
                        "corr_threshold": corr_threshold,
                        "k": k_value,
                        "C": c_value,
                        "class_weight": "balanced" if class_weight == "balanced" else "none",
                        "folds": len(fold_scores),
                    }
                    for metric in fold_scores[0]:
                        row[metric] = float(np.nanmean([score[metric] for score in fold_scores]))
                        row[f"{metric}_std"] = float(np.nanstd([score[metric] for score in fold_scores]))
                    rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    return results.sort_values(
        ["log_loss", "brier", "roc_auc", "average_precision"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def rank_model_selection(results, selection_objective="calibrated_macro"):
    ranked = results.copy()
    best_log_loss = ranked["log_loss"].min()
    ranked["within_logloss_tolerance"] = ranked["log_loss"] <= best_log_loss + MODEL_SELECTION_LOGLOSS_TOLERANCE
    if selection_objective == "macro_f1":
        sort_cols = ["macro_f1", "balanced_accuracy", "accuracy", "log_loss", "brier_multiclass"]
        ascending = [False, False, False, True, True]
    elif selection_objective == "balanced_accuracy":
        sort_cols = ["balanced_accuracy", "macro_f1", "accuracy", "log_loss", "brier_multiclass"]
        ascending = [False, False, False, True, True]
    elif selection_objective == "log_loss":
        sort_cols = ["log_loss", "brier_multiclass", "macro_f1", "balanced_accuracy"]
        ascending = [True, True, False, False]
    else:
        sort_cols = ["within_logloss_tolerance", "macro_f1", "balanced_accuracy", "log_loss", "brier_multiclass"]
        ascending = [False, False, False, True, True]
    return ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def choose_feature_set(train, feature_cols, selection_objective="calibrated_macro"):
    table_cols = [col for col in feature_cols if any(marker in col for marker in TABLE_PRIOR_FEATURES)]
    h2h_cols = [col for col in feature_cols if col.startswith("h2h_")]
    rolling_cols = [col for col in feature_cols if col not in table_cols and col not in h2h_cols]

    rolling_results = model_selection(train, rolling_cols, "rolling_only", selection_objective)
    candidates = [("rolling_only", rolling_cols, rolling_results)]
    if h2h_cols:
        h2h_pool = rolling_cols + h2h_cols
        candidates.append(
            ("rolling_plus_h2h", h2h_pool, model_selection(train, h2h_pool, "rolling_plus_h2h", selection_objective))
        )
    if table_cols:
        table_pool = rolling_cols + table_cols
        candidates.append(
            ("rolling_plus_table", table_pool, model_selection(train, table_pool, "rolling_plus_table", selection_objective))
        )
    if table_cols and h2h_cols:
        full_pool = rolling_cols + table_cols + h2h_cols
        candidates.append(
            (
                "rolling_plus_table_h2h",
                full_pool,
                model_selection(train, full_pool, "rolling_plus_table_h2h", selection_objective),
            )
        )

    best_rolling = rolling_results.iloc[0]
    pool_by_name = {name: pool for name, pool, _ in candidates}
    combined = rank_model_selection(
        pd.concat([results for _, _, results in candidates], ignore_index=True),
        selection_objective,
    )
    if selection_objective == "calibrated_macro":
        eligible_names = {
            name
            for name, _, results in candidates
            if name == "rolling_only"
            or results.iloc[0]["log_loss"] <= best_rolling["log_loss"] - TABLE_FEATURE_MIN_LOGLOSS_GAIN
        }
        selected_name = combined[combined["feature_set"].isin(eligible_names)].iloc[0]["feature_set"]
    else:
        selected_name = combined.iloc[0]["feature_set"]
    selected_pool = pool_by_name[selected_name]
    return combined, selected_pool, selected_name


def choose_binary_feature_set(train, feature_cols, target_col):
    table_cols = [col for col in feature_cols if any(marker in col for marker in TABLE_PRIOR_FEATURES)]
    h2h_cols = [col for col in feature_cols if col.startswith("h2h_")]
    rolling_cols = [col for col in feature_cols if col not in table_cols and col not in h2h_cols]

    candidates = []
    pools = [("rolling_only", rolling_cols)]
    if h2h_cols:
        pools.append(("rolling_plus_h2h", rolling_cols + h2h_cols))
    if table_cols:
        pools.append(("rolling_plus_table", rolling_cols + table_cols))
    if table_cols and h2h_cols:
        pools.append(("rolling_plus_table_h2h", rolling_cols + table_cols + h2h_cols))

    for name, pool in pools:
        if not pool:
            continue
        results = binary_model_selection(train, pool, target_col, name)
        if not results.empty:
            candidates.append((name, pool, results))

    if not candidates:
        raise RuntimeError(f"No se pudo seleccionar modelo para {target_col}: no hubo folds validos.")

    pool_by_name = {name: pool for name, pool, _ in candidates}
    combined = pd.concat([results for _, _, results in candidates], ignore_index=True)
    combined = combined.sort_values(
        ["log_loss", "brier", "roc_auc", "average_precision"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    selected_name = combined.iloc[0]["feature_set"]
    return combined, pool_by_name[selected_name], selected_name


def selected_feature_names(model, feature_cols):
    names = np.array(feature_cols)
    variance_mask = model.named_steps["variance"].get_support()
    names = names[variance_mask]
    names = names[model.named_steps["corr"].keep_indices_]
    select = model.named_steps["select"]
    if getattr(select, "k", "all") != "all":
        names = names[select.get_support()]
    return names.tolist()


def split_train_test(usable, current_season_id, test_round_min):
    if current_season_id is None:
        current_season_id = usable.sort_values(["date", "time"])["season_id"].dropna().iloc[-1]
    current_mask = usable["season_id"].astype(str).eq(str(current_season_id))
    test = usable[current_mask & (usable["round_num"] >= test_round_min)].copy()
    train = usable[~(current_mask & (usable["round_num"] >= test_round_min))].copy()
    return train, test, current_season_id


def add_goal_market_targets(dataset):
    data = dataset.copy()
    total_goals = pd.to_numeric(data["home_goals"], errors="coerce") + pd.to_numeric(
        data["away_goals"],
        errors="coerce",
    )
    data["over_15"] = (total_goals >= 2).astype(float)
    data["over_25"] = (total_goals >= 3).astype(float)
    data["btts"] = (
        (pd.to_numeric(data["home_goals"], errors="coerce") > 0)
        & (pd.to_numeric(data["away_goals"], errors="coerce") > 0)
    ).astype(float)
    data.loc[total_goals.isna(), list(GOAL_MARKETS)] = np.nan
    return data


def add_goal_market_probabilities(models, dataset):
    predictions = dataset[
        ["date", "season_id", "season_name", "round_num", "local_team", "away_team"]
    ].copy()
    for target, bundle in models.items():
        probabilities = bundle["model"].predict_proba(dataset[bundle["features"]])[:, 1]
        predictions[f"p_{target}_raw"] = probabilities
        predictions[f"p_{target}"] = probabilities
        if target in dataset.columns:
            predictions[f"target_{target}"] = dataset[target]

    if {"p_over_15", "p_over_25"}.issubset(predictions.columns):
        predictions["monotonic_adjusted_over_15"] = predictions["p_over_15"] < predictions["p_over_25"]
        predictions["p_over_15"] = np.maximum(predictions["p_over_15"], predictions["p_over_25"])
    else:
        predictions["monotonic_adjusted_over_15"] = False

    for target in models:
        predictions[f"pred_{target}"] = (predictions[f"p_{target}"] >= 0.5).astype(int)
    return predictions


def train_goal_market_models(
    dataset,
    feature_cols,
    root,
    league,
    current_season_id=None,
    test_round_min=20,
):
    out_models = root / "models" / league
    out_reports = root / "reports" / league
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    data = add_goal_market_targets(dataset)
    data = data.dropna(subset=list(GOAL_MARKETS)).sort_values(["date", "time"]).copy()
    for target in GOAL_MARKETS:
        data[target] = data[target].astype(int)

    rolling_feature_cols = [col for col in feature_cols if "_rolling" in col]
    usable = data.dropna(subset=rolling_feature_cols, how="all").copy()
    if usable.empty:
        raise RuntimeError("No hay filas con historial suficiente para entrenar goles.")

    train, test, current_season_id = split_train_test(usable, current_season_id, test_round_min)
    if test.empty:
        split_at = max(1, int(len(usable) * 0.75))
        train = usable.iloc[:split_at].copy()
        test = usable.iloc[split_at:].copy()
        current_season_id = None

    final_models = {}
    selection_frames = []
    metric_rows = []
    selected_feature_rows = []

    for target, label in GOAL_MARKETS.items():
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        selection_results, selected_feature_pool, selected_feature_set = choose_binary_feature_set(
            train,
            feature_cols,
            target,
        )
        selection_results = selection_results[selection_results["feature_set"].eq(selected_feature_set)].copy()
        best = selection_results.iloc[0]
        k_value = "all" if str(best["k"]) == "all" else int(best["k"])
        class_weight = "balanced" if best["class_weight"] == "balanced" else None
        model = build_model(
            float(best["C"]),
            k_value,
            float(best["corr_threshold"]),
            class_weight,
        )
        model.fit(train[selected_feature_pool], train[target])
        selected_features = selected_feature_names(model, selected_feature_pool)
        final_models[target] = {
            "model": model,
            "features": selected_feature_pool,
            "selected_features": selected_features,
            "label": label,
            "selection": best.to_dict(),
        }
        selection_frames.append(selection_results)
        selected_feature_rows.extend(
            {"target": target, "market": label, "feature": feature}
            for feature in selected_features
        )

        for split_name, split_df in [("train", train), ("test", test)]:
            scores = evaluate_binary_model(model, split_df, selected_feature_pool, target)
            metric_rows.append(
                {
                    "league": league,
                    "target": target,
                    "market": label,
                    "split": split_name,
                    "rows": int(len(split_df)),
                    "positive_rate": float(split_df[target].mean()),
                    "current_season_id": None if current_season_id is None else int(current_season_id),
                    "test_round_min": int(test_round_min),
                    "feature_set": best["feature_set"],
                    "features_pool": int(len(selected_feature_pool)),
                    "features_selected": int(len(selected_features)),
                    "corr_threshold": float(best["corr_threshold"]),
                    "k": best["k"],
                    "C": float(best["C"]),
                    "class_weight": best["class_weight"],
                    **scores,
                }
            )

    if not final_models:
        raise RuntimeError(f"No se entreno ningun mercado de goles para {league}.")

    predictions = add_goal_market_probabilities(final_models, test)
    monotonic_violations = int(predictions["monotonic_adjusted_over_15"].sum())
    for row in metric_rows:
        row["monotonic_over15_adjustments_test"] = monotonic_violations

    pd.concat(selection_frames, ignore_index=True).to_csv(
        out_reports / f"{league}_goal_market_selection.csv",
        index=False,
    )
    pd.DataFrame(metric_rows).to_csv(
        out_reports / f"{league}_goal_market_metrics.csv",
        index=False,
    )
    pd.DataFrame(selected_feature_rows).to_csv(
        out_reports / f"{league}_goal_market_selected_features.csv",
        index=False,
    )
    predictions.to_csv(
        out_reports / f"{league}_goal_market_predictions.csv",
        index=False,
    )
    joblib.dump(
        {
            "models": final_models,
            "markets": GOAL_MARKETS,
            "all_features": feature_cols,
            "current_season_id": current_season_id,
            "test_round_min": test_round_min,
            "relationship": "over_15_probability_is_adjusted_to_be_at_least_over_25",
        },
        out_models / "goal_market_models.joblib",
    )
    return pd.DataFrame(metric_rows)


def train_model(
    dataset,
    feature_cols,
    root,
    league,
    current_season_id=None,
    test_round_min=20,
    selection_objective="calibrated_macro",
):
    out_features = root / "features" / league
    out_models = root / "models" / league
    out_reports = root / "reports" / league
    out_features.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    dataset = dataset.dropna(subset=["target"]).sort_values(["date", "time"]).copy()
    rolling_feature_cols = [col for col in feature_cols if "_rolling" in col]
    usable = dataset.dropna(subset=rolling_feature_cols, how="all").copy()
    if usable.empty:
        raise RuntimeError("No hay filas con historial suficiente para entrenar.")

    train, test, current_season_id = split_train_test(usable, current_season_id, test_round_min)
    if train["target"].nunique() < 3 or test.empty:
        split_at = max(1, int(len(usable) * 0.75))
        train = usable.iloc[:split_at].copy()
        test = usable.iloc[split_at:].copy()
        current_season_id = None

    selection_results, selected_feature_pool, selected_feature_set = choose_feature_set(
        train,
        feature_cols,
        selection_objective,
    )
    selection_results = selection_results[selection_results["feature_set"].eq(selected_feature_set)].copy()
    best = selection_results.iloc[0]
    k_value = "all" if str(best["k"]) == "all" else int(best["k"])
    class_weight = "balanced" if best["class_weight"] == "balanced" else None
    model = build_model(
        float(best["C"]),
        k_value,
        float(best["corr_threshold"]),
        class_weight,
    )
    model.fit(train[selected_feature_pool], train["target"])

    classes = model.named_steps["logreg"].classes_.tolist()
    train_scores = evaluate_model(model, train, selected_feature_pool)
    test_scores = evaluate_model(model, test, selected_feature_pool)
    selected_features = selected_feature_names(model, selected_feature_pool)
    metrics = {
        "league": league,
        "rows_total": int(len(usable)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "current_season_id": None if current_season_id is None else int(current_season_id),
        "test_round_min": int(test_round_min),
        "selection_objective": selection_objective,
        "features_raw": int(len(feature_cols)),
        "features_pool": int(len(selected_feature_pool)),
        "features_selected": int(len(selected_features)),
        "feature_set": best["feature_set"],
        "corr_threshold": float(best["corr_threshold"]),
        "k": best["k"],
        "C": float(best["C"]),
        "class_weight": best["class_weight"],
        "test_accuracy": test_scores["accuracy"],
        "test_balanced_accuracy": test_scores["balanced_accuracy"],
        "test_macro_f1": test_scores["macro_f1"],
        "test_weighted_f1": test_scores["weighted_f1"],
        "test_log_loss": test_scores["log_loss"],
        "test_brier_multiclass": test_scores["brier_multiclass"],
        "train_accuracy": train_scores["accuracy"],
        "train_balanced_accuracy": train_scores["balanced_accuracy"],
        "train_macro_f1": train_scores["macro_f1"],
        "train_weighted_f1": train_scores["weighted_f1"],
        "train_log_loss": train_scores["log_loss"],
        "train_brier_multiclass": train_scores["brier_multiclass"],
        "classes": classes,
    }

    preds = model.predict(test[selected_feature_pool])
    proba = model.predict_proba(test[selected_feature_pool])
    predictions = test[["date", "season_id", "season_name", "round_num", "local_team", "away_team", "target"]].copy()
    predictions["pred"] = preds
    for idx, class_name in enumerate(classes):
        predictions[f"prob_{class_name}"] = proba[:, idx]

    dataset.to_csv(out_features / f"{league}_match_dataset.csv", index=False)
    predictions.to_csv(out_reports / f"{league}_logreg_predictions.csv", index=False)
    selection_results.to_csv(out_reports / f"{league}_logreg_selection.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(
        out_reports / f"{league}_selected_features.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"split": "train", "rows": len(train), **train_scores},
            {"split": "test", "rows": len(test), **test_scores},
        ]
    ).to_csv(out_reports / f"{league}_logreg_train_test_metrics.csv", index=False)
    pd.Series(metrics).to_json(out_reports / f"{league}_logreg_metrics.json", indent=2)
    (out_reports / f"{league}_classification_report.txt").write_text(
        classification_report(test["target"], preds, labels=classes, zero_division=0),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": model,
            "features": selected_feature_pool,
            "all_features": feature_cols,
            "selected_features": selected_features,
            "metrics": metrics,
            "selection_results": selection_results,
        },
        out_models / "logreg_1x2.joblib",
    )
    return metrics


def run_scraper(args):
    command = [
        sys.executable,
        "src/01_scrape_sofascore.py",
        "--league",
        args.league,
        "--pipeline-root",
        str(PIPELINE_ROOT),
    ]
    if args.include_future:
        command.append("--include-future")
    if args.seasons_back:
        command.extend(["--seasons-back", str(args.seasons_back)])
    if args.odds:
        command.append("--odds")
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.no_headless:
        command.append("--no-headless")
    subprocess.run(command, cwd=BASE_DIR, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--include-future", action="store_true")
    parser.add_argument("--odds", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seasons-back", type=int, default=2)
    parser.add_argument("--current-season-id", type=int, default=None)
    parser.add_argument("--test-round-min", type=int, default=20)
    parser.add_argument(
        "--selection-objective",
        choices=SELECTION_OBJECTIVES,
        default="calibrated_macro",
    )
    parser.add_argument("--goal-markets-only", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    if not args.skip_scrape:
        run_scraper(args)

    raw = load_raw_matchlogs(PIPELINE_ROOT, args.league)
    h2h_features = load_h2h_features(PIPELINE_ROOT, args.league)
    dataset, features = build_match_dataset(raw, h2h_features)
    metrics = None
    if not args.goal_markets_only:
        metrics = train_model(
            dataset,
            features,
            PIPELINE_ROOT,
            args.league,
            current_season_id=args.current_season_id,
            test_round_min=args.test_round_min,
            selection_objective=args.selection_objective,
        )
    goal_metrics = train_goal_market_models(
        dataset,
        features,
        PIPELINE_ROOT,
        args.league,
        current_season_id=args.current_season_id,
        test_round_min=args.test_round_min,
    )
    if metrics is not None:
        print("\nSofaScore logistic regression baseline")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    print("\nSofaScore goal market models")
    print(goal_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
