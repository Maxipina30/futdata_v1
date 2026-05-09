from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = BASE_DIR / ".runtime" / "python312" / "python.exe"
LOG_DIR = BASE_DIR / "logs"
STDOUT_LOG = LOG_DIR / "streamlit_stdout.log"
STDERR_LOG = LOG_DIR / "streamlit_stderr.log"


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"

    with STDOUT_LOG.open("ab") as stdout, STDERR_LOG.open("ab") as stderr:
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
        )
        process = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "streamlit",
                "run",
                "apps/predictions/dashboard_streamlit.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ],
            cwd=BASE_DIR,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
