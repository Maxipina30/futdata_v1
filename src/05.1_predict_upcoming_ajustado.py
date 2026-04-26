"""Compatibility wrapper for the current Premier League predictor.

The old adjusted predictor used legacy Chile paths/model files. Keep this
entrypoint so old commands still run, but delegate to the maintained script.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("05_predict_upcoming.py"), run_name="__main__")
