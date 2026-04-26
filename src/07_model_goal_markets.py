import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "files", "03_features")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(FEATURES_DIR, "dataset_modelo_train_mw1_28.csv")
TEST_PATH = os.path.join(FEATURES_DIR, "dataset_modelo_test_mw29_33.csv")
PREDICT_PATH = os.path.join(FEATURES_DIR, "dataset_prediccion_mw34_plus.csv")

MODEL_SELECTION_FOLDS = [
    {"train_max_round": 16, "valid_min_round": 17, "valid_max_round": 20},
    {"train_max_round": 20, "valid_min_round": 21, "valid_max_round": 24},
    {"train_max_round": 24, "valid_min_round": 25, "valid_max_round": 28},
]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
MARKETS = {
    "over_15": "Over 1.5 goles",
    "over_25": "Over 2.5 goles",
    "btts": "Ambos anotan",
}


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, smoothing=0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing

    def fit(self, x_data, y_data):
        x_array = np.asarray(x_data, dtype=float)
        y_array = np.asarray(y_data, dtype=int)
        self.bin_edges_ = []
        self.woe_maps_ = []
        total_pos = max((y_array == 1).sum(), 1)
        total_neg = max((y_array == 0).sum(), 1)

        for col_idx in range(x_array.shape[1]):
            values = x_array[:, col_idx]
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.nanquantile(values, quantiles))
            if len(edges) <= 2:
                edges = np.array([np.nanmin(values), np.nanmax(values)])
            cuts = edges[1:-1]
            bins = np.digitize(values, cuts, right=False)
            mapping = {}
            for bin_id in np.unique(bins):
                mask = bins == bin_id
                pos = ((y_array == 1) & mask).sum()
                neg = ((y_array == 0) & mask).sum()
                pos_rate = (pos + self.smoothing) / (total_pos + self.smoothing * len(np.unique(bins)))
                neg_rate = (neg + self.smoothing) / (total_neg + self.smoothing * len(np.unique(bins)))
                mapping[int(bin_id)] = np.log(pos_rate / neg_rate)

            self.bin_edges_.append(cuts)
            self.woe_maps_.append(mapping)
        return self

    def transform(self, x_data):
        x_array = np.asarray(x_data, dtype=float)
        transformed = np.zeros_like(x_array, dtype=float)
        for col_idx in range(x_array.shape[1]):
            bins = np.digitize(x_array[:, col_idx], self.bin_edges_[col_idx], right=False)
            mapping = self.woe_maps_[col_idx]
            transformed[:, col_idx] = [mapping.get(int(bin_id), 0.0) for bin_id in bins]
        return transformed


def model_features(df):
    feature_cols = []
    contextual_metrics = (
        "gf",
        "ga",
        "sot",
        "sot_allowed",
        "winrate",
        "points_per_match",
    )
    for col in df.columns:
        is_global_team_feature = (
            col.startswith("_")
            and col.endswith(("_local", "_away"))
            and ("rolling" in col or "season_avg" in col)
        )
        is_contextual_home_feature = (
            col.startswith("home_")
            and col.endswith("_local")
            and any(metric in col for metric in contextual_metrics)
        )
        is_contextual_away_feature = (
            col.startswith("away_")
            and col.endswith("_away")
            and any(metric in col for metric in contextual_metrics)
        )
        is_table_difference = col.startswith("diff_table_")
        if (
            is_global_team_feature
            or is_contextual_home_feature
            or is_contextual_away_feature
            or is_table_difference
        ):
            feature_cols.append(col)
    return feature_cols


def add_market_targets(df):
    required = {"gf_local", "ga_local", "gf_away", "ga_away"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Faltan goles para crear targets {missing}. Ejecuta src/03_feature_engineering.py"
        )

    data = df.copy()
    goals_home = pd.to_numeric(data["gf_local"], errors="coerce")
    goals_away = pd.to_numeric(data["gf_away"], errors="coerce")
    total_goals = goals_home + goals_away
    data["over_15"] = (total_goals >= 2).astype(float)
    data["over_25"] = (total_goals >= 3).astype(float)
    data["btts"] = ((goals_home > 0) & (goals_away > 0)).astype(float)
    data.loc[total_goals.isna(), ["over_15", "over_25", "btts"]] = np.nan
    return data


def load_dataset(path, require_targets=True):
    df = pd.read_csv(path, parse_dates=["date"])
    df = add_market_targets(df)
    if require_targets:
        df = df.dropna(subset=list(MARKETS)).copy()
        for market in MARKETS:
            df[market] = df[market].astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def build_model(model_kind, c_value):
    if model_kind == "logreg_woe":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("woe", WOETransformer(n_bins=5, smoothing=0.5)),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=5000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def safe_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, probabilities)


def score_model(model, x_data, y_true):
    pred = model.predict(x_data)
    probabilities = model.predict_proba(x_data)[:, 1]
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": safe_auc(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "brier": brier_score_loss(y_true, probabilities),
    }


def model_selection(train, features, target):
    rows = []
    for model_kind in ["logreg"]:
        for c_value in C_GRID:
            fold_scores = []
            for fold in MODEL_SELECTION_FOLDS:
                train_core = train[train["round_num"] <= fold["train_max_round"]].copy()
                validation = train[
                    train["round_num"].between(
                        fold["valid_min_round"],
                        fold["valid_max_round"],
                    )
                ].copy()
                if train_core.empty or validation.empty:
                    continue
                if train_core[target].nunique() < 2 or validation[target].nunique() < 2:
                    continue

                model = build_model(model_kind, c_value)
                model.fit(train_core[features], train_core[target])
                fold_scores.append(score_model(model, validation[features], validation[target]))

            if not fold_scores:
                continue

            rows.append(
                {
                    "target": target,
                    "model_kind": model_kind,
                    "C": c_value,
                    "n_features": len(features),
                    "folds": len(fold_scores),
                    **{
                        metric: np.nanmean([score[metric] for score in fold_scores])
                        for metric in fold_scores[0]
                    },
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["log_loss", "brier", "roc_auc"],
        ascending=[True, True, False],
    )


def coefficient_importance(model, features):
    coefficients = np.abs(model.named_steps["model"].coef_[0])
    importance = pd.DataFrame({"feature": features, "coef_abs": coefficients})
    total = importance["coef_abs"].sum()
    importance["importance_pct"] = importance["coef_abs"] / total * 100 if total else 0
    return importance.sort_values("importance_pct", ascending=False)


def save_market_predictions(models, features, dataset, path):
    predictions = dataset[["date", "round_num", "local_team", "away_team"]].copy()
    for target, model in models.items():
        probabilities = model.predict_proba(dataset[features])[:, 1]
        predictions[f"p_{target}"] = probabilities
        predictions[f"pred_{target}"] = (probabilities >= 0.5).astype(int)
        if target in dataset.columns:
            predictions[f"target_{target}"] = dataset[target]
    predictions.to_csv(path, index=False)
    return predictions


def main():
    print("Entrenando modelos binarios: Over 1.5, Over 2.5 y Ambos Anotan\n")

    train = load_dataset(TRAIN_PATH)
    test = load_dataset(TEST_PATH)
    predict = load_dataset(PREDICT_PATH, require_targets=False)
    features = model_features(train)
    print(f"Features limpias usadas: {len(features)}")
    print(f"Train: {len(train)} filas | Test: {len(test)} filas\n")

    all_selection = []
    final_metrics = []
    final_models = {}
    importance_rows = []

    for target, label in MARKETS.items():
        print(f"=== {label} ({target}) ===")
        selection = model_selection(train, features, target)
        all_selection.append(selection)
        print(selection.head(6).to_string(index=False, float_format=lambda value: f"{value:.3f}"))

        best = selection.iloc[0]
        model = build_model(best["model_kind"], float(best["C"]))
        model.fit(train[features], train[target])
        final_models[target] = model

        for split_name, split_df in [("train_mw6_28", train), ("test_mw29_33", test)]:
            scores = score_model(model, split_df[features], split_df[target])
            final_metrics.append(
                {
                    "target": target,
                    "market": label,
                    "split": split_name,
                    "rows": len(split_df),
                    "positive_rate": split_df[target].mean(),
                    "model_kind": best["model_kind"],
                    "C": float(best["C"]),
                    **scores,
                }
            )

        importance = coefficient_importance(model, features).head(30)
        importance["target"] = target
        importance_rows.append(importance)

        print("Reporte test:")
        y_pred = model.predict(test[features])
        print(classification_report(test[target], y_pred, digits=3, zero_division=0))
        print()

    selection_path = os.path.join(REPORT_DIR, "seleccion_modelos_goles.csv")
    metrics_path = os.path.join(REPORT_DIR, "metricas_modelos_goles.csv")
    importance_path = os.path.join(REPORT_DIR, "importancia_modelos_goles.csv")
    test_pred_path = os.path.join(REPORT_DIR, "predicciones_goles_test_mw29_33.csv")
    future_pred_path = os.path.join(REPORT_DIR, "predicciones_goles_mw34_plus.csv")
    model_path = os.path.join(MODEL_DIR, "modelos_goles.joblib")

    pd.concat(all_selection, ignore_index=True).to_csv(selection_path, index=False)
    pd.DataFrame(final_metrics).to_csv(metrics_path, index=False)
    pd.concat(importance_rows, ignore_index=True).to_csv(importance_path, index=False)
    save_market_predictions(final_models, features, test, test_pred_path)
    save_market_predictions(final_models, features, predict, future_pred_path)
    joblib.dump(
        {
            "models": final_models,
            "features": features,
            "markets": MARKETS,
        },
        model_path,
    )

    print("Resumen final:")
    print(pd.DataFrame(final_metrics).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nModelos guardados en: {model_path}")
    print(f"Metricas guardadas en: {metrics_path}")
    print(f"Predicciones futuras guardadas en: {future_pred_path}")


if __name__ == "__main__":
    main()
