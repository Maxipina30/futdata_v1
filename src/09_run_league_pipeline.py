import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

LEAGUES = {
    "chile": "Liga de Primera",
    "la_liga": "La Liga",
    "serie_a": "Serie A",
}


def run_step(args):
    print(f"\n>>> {' '.join(args)}")
    subprocess.run(args, cwd=BASE_DIR, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, choices=sorted(LEAGUES))
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    run_step([python, "src/00_parse_team_matchlogs.py", "--league", args.league])
    run_step([python, "src/02_cleaning_transform.py", "--league", args.league])
    run_step([python, "src/03_feature_engineering.py", "--league", args.league])

    if not args.skip_model:
        feature_dir = BASE_DIR / "files" / "03_features" / args.league
        train_path = feature_dir / "dataset_modelo_train_mw1_28.csv"
        test_path = feature_dir / "dataset_modelo_test_mw29_33.csv"
        train_all = args.league == "chile"
        if not train_all and (not train_path.exists() or not test_path.exists()):
            print(
                "\nNo se entrena modelo todavia: faltan splits train/test. "
                "Con pocos equipos descargados el dataset de partidos queda incompleto."
            )
            return
        train_all_args = ["--train-all"] if train_all else []
        run_step([python, "src/04_model_training.py", "--league", args.league, *train_all_args])
        run_step([python, "src/07_model_goal_markets.py", "--league", args.league, *train_all_args])
        run_step([python, "src/10_predict_future_league.py", "--league", args.league])


if __name__ == "__main__":
    main()
