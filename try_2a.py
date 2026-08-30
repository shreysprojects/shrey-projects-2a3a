"""Interactive demo for Project 2A (Service Request Cleanup).

The Python replacement for the old TEST-2A.bat — cross-platform, no batch files.

Two ways to run:
  * Double-click this file in File Explorer, OR
  * From the project root:  python try_2a.py

Type or paste a messy service note, press Enter, and see the cleaned result.
Leave the line blank (or press Ctrl+C) to quit.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bootstrap_venv() -> None:
    """Double-clicking a .py usually launches the *system* Python, which doesn't
    have this project's dependencies (pydantic, python-dotenv). If the project's
    virtual-env exists and we're not already running under it, relaunch this same
    script with the venv's interpreter so the imports below work either way.
    """
    if os.environ.get("_TRY_BOOTSTRAPPED") == "1":
        return  # already relaunched once — don't loop
    venv_py = HERE / ".venv" / "Scripts" / "python.exe"   # Windows
    if not venv_py.exists():
        venv_py = HERE / ".venv" / "bin" / "python"       # macOS / Linux
    if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
        env = {**os.environ, "_TRY_BOOTSTRAPPED": "1"}
        raise SystemExit(
            subprocess.call(
                [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]],
                cwd=str(HERE), env=env,
            )
        )


_bootstrap_venv()
os.chdir(HERE)                     # so `import src...` resolves and .env is found
sys.path.insert(0, str(HERE))

from src.service_request_cleanup import main  # noqa: E402  (after venv bootstrap)

BANNER = (
    "\n"
    "===================================================================\n"
    " Project 2A - Service Request Cleanup\n"
    "===================================================================\n"
    " Type or paste a messy service note, then press Enter.\n"
    " (Leave it blank and press Enter to quit.)\n"
)


def run() -> None:
    print(BANNER)
    while True:
        try:
            note = input("Note: ").replace("﻿", "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not note:
            break
        print()
        main([note])  # reuse the exact same CLI the command line uses
        print("\n-------------------------------------------------------------------")
    print("Bye.")


if __name__ == "__main__":
    run()
