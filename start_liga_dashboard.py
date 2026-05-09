from __future__ import annotations

import os
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = BASE_DIR / ".runtime" / "python312" / "python.exe"
LOG_DIR = BASE_DIR / "logs"
STDOUT_LOG = LOG_DIR / "liga_dashboard_stdout.log"
STDERR_LOG = LOG_DIR / "liga_dashboard_stderr.log"


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"

    with STDOUT_LOG.open("ab") as stdout, STDERR_LOG.open("ab") as stderr:
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | 0x01000000
        )
        process = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "streamlit",
                "run",
                "analysis/liga_chilena_pm/dashboard.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8502",
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
