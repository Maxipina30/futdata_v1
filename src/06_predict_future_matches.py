import argparse
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = BASE_DIR / "files" / "03_features"
MODEL_DIR = BASE_DIR / "files" / "04_models"
REPORT_DIR = BASE_DIR / "files" / "05_reports"
MODEL_CLASSES = [-1, 0, 1]


def league_dir(base_dir, league):
    return base_dir / league


def aligned_probabilities(model, x_data):
    probabilities = model.predict_proba(x_data)
    aligned = pd.DataFrame(0.0, index=x_data.index, columns=MODEL_CLASSES)
    for index, klass in enumerate(model.classes_):
        aligned[int(klass)] = probabilities[:, index]
    return aligned


def outcome_text(row):
    pred = int(row["prediccion"])
    if pred == 1:
        return f"Gana {row['local_team']}"
    if pred == -1:
        return f"Gana {row['away_team']}"
    return "Empate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier")
    args = parser.parse_args()

    features_dir = league_dir(FEATURES_DIR, args.league)
    model_dir = league_dir(MODEL_DIR, args.league)
    report_dir = league_dir(REPORT_DIR, args.league)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = features_dir / "dataset_prediccion_mw34_plus.csv"
    model_path = model_dir / "modelo_partidos.joblib"
    metadata_path = model_dir / "modelo_partidos_metadata.joblib"

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    features = metadata["features"]

    probabilities = aligned_probabilities(model, df[features])
    predictions = df[["date", "round_num", "local_team", "away_team", "Target"]].copy()
    predictions["league"] = args.league
    predictions["prediccion"] = probabilities.idxmax(axis=1).astype(int)
    predictions["p_away_win"] = probabilities[-1].to_numpy()
    predictions["p_draw"] = probabilities[0].to_numpy()
    predictions["p_home_win"] = probabilities[1].to_numpy()
    predictions["resultado_esperado"] = predictions.apply(outcome_text, axis=1)
    predictions["confianza"] = predictions[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
    predictions = predictions.sort_values(["date", "round_num", "local_team"]).reset_index(drop=True)

    out_path = report_dir / "predicciones_futuras_1x2.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Guardado {out_path} ({len(predictions)} filas)")


if __name__ == "__main__":
    main()
