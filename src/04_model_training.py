import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "files", "03_features")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(FEATURES_DIR, "dataset_modelo_train_mw1_28.csv")
TEST_PATH = os.path.join(FEATURES_DIR, "dataset_modelo_test_mw29_33.csv")

MODEL_SELECTION_FOLDS = [
    {"train_max_round": 16, "valid_min_round": 17, "valid_max_round": 20},
    {"train_max_round": 20, "valid_min_round": 21, "valid_max_round": 24},
    {"train_max_round": 24, "valid_min_round": 25, "valid_max_round": 28},
]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
MODEL_CLASSES = [-1, 0, 1]
MACRO_F1_TOLERANCE = 0.01


def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df.dropna(subset=["Target"]).replace([np.inf, -np.inf], np.nan)
    df["Target"] = df["Target"].astype(int)
    return df


def build_logistic_pipeline(c_value):
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


def build_model(model_kind, c_value):
    base_model = build_logistic_pipeline(c_value)
    if model_kind == "logreg_calibrated":
        return CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    return base_model


def model_features(train):
    feature_cols = []
    for col in train.columns:
        is_global_team_feature = (
            col.startswith("_")
            and col.endswith(("_local", "_away"))
            and ("rolling" in col or "season_avg" in col)
        )
        is_table_difference = col.startswith("diff_table_")
        if is_global_team_feature or is_table_difference:
            feature_cols.append(col)
    return feature_cols


def aligned_probabilities(model, x_data):
    probabilities = model.predict_proba(x_data)
    aligned = np.zeros((len(x_data), len(MODEL_CLASSES)))
    class_to_index = {int(klass): idx for idx, klass in enumerate(model.classes_)}
    for out_idx, klass in enumerate(MODEL_CLASSES):
        if klass in class_to_index:
            aligned[:, out_idx] = probabilities[:, class_to_index[klass]]
    return aligned


def multiclass_brier(y_true, probabilities):
    y_array = y_true.astype(int).to_numpy()
    encoded = np.zeros_like(probabilities)
    class_to_index = {klass: idx for idx, klass in enumerate(MODEL_CLASSES)}
    for row_idx, klass in enumerate(y_array):
        encoded[row_idx, class_to_index[klass]] = 1
    return np.mean(np.sum((encoded - probabilities) ** 2, axis=1))


def score_model(model, x_data, y_true):
    pred = model.predict(x_data)
    probabilities = aligned_probabilities(model, x_data)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "log_loss": log_loss(y_true, probabilities, labels=MODEL_CLASSES),
        "brier_multiclass": multiclass_brier(y_true, probabilities),
    }


def model_selection(train):
    rows = []
    features = model_features(train)
    for model_kind in ["logreg", "logreg_calibrated"]:
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

                model = build_model(model_kind, c_value)
                model.fit(train_core[features], train_core["Target"])
                fold_scores.append(score_model(model, validation[features], validation["Target"]))

            if not fold_scores:
                continue

            rows.append(
                {
                    "feature_set": "all_generated_features",
                    "model_kind": model_kind,
                    "C": c_value,
                    "n_features": len(features),
                    "folds": len(fold_scores),
                    "accuracy": np.mean([score["accuracy"] for score in fold_scores]),
                    "balanced_accuracy": np.mean(
                        [score["balanced_accuracy"] for score in fold_scores]
                    ),
                    "macro_f1": np.mean([score["macro_f1"] for score in fold_scores]),
                    "weighted_f1": np.mean([score["weighted_f1"] for score in fold_scores]),
                    "log_loss": np.mean([score["log_loss"] for score in fold_scores]),
                    "brier_multiclass": np.mean(
                        [score["brier_multiclass"] for score in fold_scores]
                    ),
                    "macro_f1_std": np.std([score["macro_f1"] for score in fold_scores]),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "log_loss", "brier_multiclass", "n_features"],
        ascending=[False, True, True, True],
    )
    return results


def choose_best_config(selection_results):
    best_macro_f1 = selection_results["macro_f1"].max()
    contenders = selection_results[
        selection_results["macro_f1"] >= best_macro_f1 - MACRO_F1_TOLERANCE
    ].copy()
    return contenders.sort_values(
        ["log_loss", "brier_multiclass", "n_features", "macro_f1"],
        ascending=[True, True, True, False],
    ).iloc[0]


def fit_coefficient_model(features, c_value, train):
    model = build_logistic_pipeline(c_value)
    model.fit(train[features], train["Target"])
    return model


def coefficient_importance(model, features):
    coefficients = np.abs(model.named_steps["model"].coef_).sum(axis=0)
    importance = pd.DataFrame({"feature": features, "coef_abs_sum": coefficients})
    total = importance["coef_abs_sum"].sum()
    importance["importance_pct"] = importance["coef_abs_sum"] / total * 100 if total else 0
    return importance.sort_values("importance_pct", ascending=False)


def save_predictions(model, test, features):
    y_pred = model.predict(test[features])
    y_proba = aligned_probabilities(model, test[features])
    proba_cols = [f"P_{klass}" for klass in MODEL_CLASSES]

    predictions = test[["date", "round_num", "local_team", "away_team", "Target"]].copy()
    for index, col in enumerate(proba_cols):
        predictions[col] = y_proba[:, index]
    predictions["Prediccion_Final"] = y_pred
    return predictions


def main():
    print("Entrenando regresion logistica con features diferenciales y calibracion\n")

    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    print(f"Train MW 1-28: {len(train)} filas")
    print("Validacion interna walk-forward:")
    for fold in MODEL_SELECTION_FOLDS:
        print(
            f" - train <= MW {fold['train_max_round']}, "
            f"valid MW {fold['valid_min_round']}-{fold['valid_max_round']}"
        )
    print(f"Test final MW 29-33: {len(test)} filas\n")

    selection_results = model_selection(train)
    selection_path = os.path.join(REPORT_DIR, "seleccion_modelo_logreg.csv")
    selection_results.to_csv(selection_path, index=False)

    print("Top configuraciones en validacion interna:")
    print(selection_results.head(12).to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best = choose_best_config(selection_results)
    set_name = best["feature_set"]
    model_kind = best["model_kind"]
    c_value = float(best["C"])
    features = model_features(train)

    final_model = build_model(model_kind, c_value)
    final_model.fit(train[features], train["Target"])

    final_rows = []
    for split_name, split_df in [("train_mw1_28", train), ("test_mw29_33", test)]:
        scores = score_model(final_model, split_df[features], split_df["Target"])
        final_rows.append({"split": split_name, "rows": len(split_df), **scores})

    final_metrics = pd.DataFrame(final_rows)
    print("\nMetricas finales:")
    print(final_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    y_test = test["Target"]
    y_pred = final_model.predict(test[features])
    print("\nReporte test final:")
    print(classification_report(y_test, y_pred, labels=MODEL_CLASSES, digits=3, zero_division=0))
    print("Matriz de confusion test labels [-1, 0, 1]:")
    print(confusion_matrix(y_test, y_pred, labels=MODEL_CLASSES))

    model_path = os.path.join(MODEL_DIR, "modelo_partidos.joblib")
    metadata_path = os.path.join(MODEL_DIR, "modelo_partidos_metadata.joblib")
    final_metrics_path = os.path.join(REPORT_DIR, "metricas_modelos_partidos.csv")
    pred_path = os.path.join(REPORT_DIR, "predicciones_partidos_test_mw29_33.csv")
    importance_path = os.path.join(REPORT_DIR, "importancia_coeficientes_logreg.csv")

    coefficient_model = fit_coefficient_model(features, c_value, train)

    joblib.dump(final_model, model_path)
    joblib.dump(
        {
            "features": features,
            "selected_features": features,
            "best_model": model_kind,
            "feature_set": set_name,
            "C": c_value,
            "select_k": None,
            "classes": MODEL_CLASSES,
            "selection_metric": "macro_f1_desc_log_loss_asc",
            "probabilities": "model_calibrated" if model_kind == "logreg_calibrated" else "model_direct",
        },
        metadata_path,
    )
    final_metrics.to_csv(final_metrics_path, index=False)
    save_predictions(final_model, test, features).to_csv(pred_path, index=False)
    coefficient_importance(coefficient_model, features).to_csv(importance_path, index=False)

    print(
        f"\nConfiguracion final guardada: {set_name}, {model_kind}, "
        f"C={c_value:g}, features={len(features)}"
    )
    print(
        f"Criterio final: mejor log_loss entre modelos a <= "
        f"{MACRO_F1_TOLERANCE:.3f} del mejor macro F1 interno."
    )
    print(f"Modelo guardado en: {model_path}")
    print(f"Metadata guardada en: {metadata_path}")
    print(f"Seleccion guardada en: {selection_path}")
    print(f"Metricas finales guardadas en: {final_metrics_path}")
    print(f"Predicciones test guardadas en: {pred_path}")
    print(f"Importancia coeficientes guardada en: {importance_path}")
    print("\n04_model_training completado.")


if __name__ == "__main__":
    main()
