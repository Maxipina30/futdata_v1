import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ========================
# CONFIGURACION
# ========================
MATCHWEEK_OBJETIVO = int(os.getenv("FUTDATA_MATCHWEEK_OBJETIVO", "34"))
MODEL_CLASSES = [-1, 0, 1]

# ========================
# RUTAS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "files", "03_features")
MODEL_DIR = os.path.join(BASE_DIR, "files", "04_models")
REPORT_DIR = os.path.join(BASE_DIR, "files", "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PREDICT_DATASET = os.path.join(FEATURES_DIR, "dataset_prediccion_mw34_plus.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_partidos.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "modelo_partidos_metadata.joblib")


def outcome_text(row):
    pred = int(row["prediccion"])
    if pred == 1:
        return f"Gana {row['local_team']}"
    if pred == -1:
        return f"Gana {row['away_team']}"
    return "Empate"


def aligned_probabilities(model, x_data):
    probabilities = model.predict_proba(x_data)
    aligned = pd.DataFrame(0.0, index=x_data.index, columns=MODEL_CLASSES)
    for index, klass in enumerate(model.classes_):
        aligned[int(klass)] = probabilities[:, index]
    return aligned


def add_display_probabilities(predictions, probabilities):
    predictions["p_away_win"] = probabilities[-1].to_numpy()
    predictions["p_draw"] = probabilities[0].to_numpy()
    predictions["p_home_win"] = probabilities[1].to_numpy()
    return predictions


def main():
    print(f"Generando predicciones para Matchweek {MATCHWEEK_OBJETIVO}\n")

    if not os.path.exists(PREDICT_DATASET):
        raise FileNotFoundError(PREDICT_DATASET)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(METADATA_PATH)

    df = pd.read_csv(PREDICT_DATASET, parse_dates=["date"])
    df = df[df["round_num"] == MATCHWEEK_OBJETIVO].copy()
    if df.empty:
        print(f"No hay partidos para Matchweek {MATCHWEEK_OBJETIVO}.")
        return

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    features = metadata["features"]
    selected_features = metadata.get("selected_features", features)

    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise RuntimeError(f"Faltan features en dataset de prediccion: {missing}")

    probabilities = aligned_probabilities(model, df[features])
    predictions = df[["date", "round_num", "local_team", "away_team", "Target"]].copy()
    predictions["prediccion"] = probabilities.idxmax(axis=1).astype(int)
    predictions = add_display_probabilities(predictions, probabilities)
    predictions["resultado_esperado"] = predictions.apply(outcome_text, axis=1)
    predictions["confianza"] = predictions[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
    predictions = predictions.sort_values(["date", "local_team"]).reset_index(drop=True)

    out_path = os.path.join(REPORT_DIR, f"predicciones_matchweek{MATCHWEEK_OBJETIVO}.csv")
    predictions.to_csv(out_path, index=False)

    print(
        "Modelo usado: "
        f"{metadata.get('best_model')} | "
        f"{metadata.get('feature_set')} | "
        f"C={metadata.get('C')} | "
        f"features={len(selected_features)}"
    )
    print("Predicciones:")
    for _, row in predictions.iterrows():
        print(f"{row['local_team']} vs {row['away_team']}")
        print(
            f"  Local: {row['p_home_win']:.3f} | "
            f"Empate: {row['p_draw']:.3f} | "
            f"Visita: {row['p_away_win']:.3f}"
        )
        print(f"  Resultado esperado: {row['resultado_esperado']}\n")

    known = predictions[predictions["Target"].notna()].copy()
    if not known.empty:
        y_true = known["Target"].astype(int)
        y_pred = known["prediccion"].astype(int)
        print("Evaluacion parcial MW34 con partidos que ya tienen resultado:")
        print(f"Partidos evaluables: {len(known)}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
        print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
        print("Matriz de confusion labels [-1, 0, 1]:")
        print(confusion_matrix(y_true, y_pred, labels=MODEL_CLASSES))
        print(classification_report(y_true, y_pred, labels=MODEL_CLASSES, digits=3, zero_division=0))
    else:
        print("MW34 aun no tiene resultados reales en el dataset; solo se generaron predicciones.")

    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
