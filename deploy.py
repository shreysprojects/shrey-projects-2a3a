#!/usr/bin/env python3
"""One-command deploy for Projects 2A + 3A (built for a CPU-only Linux box).

Run this from the project root and it does the whole setup, in order:

    python3 deploy.py

  1. creates a local virtual environment (.venv) if one isn't there
  2. installs the core Python packages (requirements.txt)
  3. installs the 3A semantic-check packages (requirements-nli.txt)  [skip: --no-nli]
  4. writes a safe .env if one is missing  (local model, NO cloud API key)
  5. checks the local model server (Ollama) is up and has the model,
     and pulls the model if it's missing
  6. (with NLI) pre-downloads the semantic-check model so the first
     request is fast instead of slow
  7. starts the web UI at  http://<host>:<port>

Python-only on purpose — no .sh / .bat, same rule as the rest of the repo.
It also runs on Windows for local testing; the Linux box is the real target.

Common runs:
    python3 deploy.py                     # set up + start the UI on 127.0.0.1:8765
    python3 deploy.py --host 0.0.0.0      # let other machines on the network reach it
    python3 deploy.py --setup-only        # do everything EXCEPT start the server
    python3 deploy.py --no-nli            # lighter box: skip the ~2.5GB NLI stack
    python3 deploy.py --model granite3.3:2b   # pick a different local model
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DEFAULT_MODEL = "granite3.3:2b"   # small + fast on CPU; both 2A and 3A use it
DEFAULT_OLLAMA = "http://127.0.0.1:11434/v1"


# --------------------------------------------------------------------------- #
# Small helpers so the output reads like a checklist, not a wall of logs.
# --------------------------------------------------------------------------- #

def step(n: int, msg: str) -> None:
    print(f"\n\033[1m[{n}/7] {msg}\033[0m" if sys.stdout.isatty() else f"\n[{n}/7] {msg}")


def ok(msg: str) -> None:
    print(f"   ok  {msg}")


def warn(msg: str) -> None:
    print(f"   !!  {msg}")


def die(msg: str) -> None:
    print(f"\n   STOP: {msg}\n")
    sys.exit(1)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, showing what it is. Raises on failure unless check=False."""
    print(f"   $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw)


def venv_python() -> Path:
    """Path to the python inside .venv (differs on Windows vs Linux/Mac)."""
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def read_env(path: Path) -> dict[str, str]:
    """Tiny .env reader (we can't rely on python-dotenv yet — it's not installed)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    return values


# --------------------------------------------------------------------------- #
# The steps
# --------------------------------------------------------------------------- #

def ensure_venv() -> None:
    step(1, "Virtual environment")
    if venv_python().exists():
        ok(f".venv already exists ({venv_python()})")
        return
    run([sys.executable, "-m", "venv", str(VENV)], check=True)
    ok(f"created {VENV}")


def pip_install(args: list[str], upgrade: bool = True) -> None:
    flags = ["--upgrade"] if upgrade else []
    run([str(venv_python()), "-m", "pip", "install", *flags, *args], check=True)


def install_core() -> None:
    step(2, "Core packages (requirements.txt)")
    pip_install(["pip"])                       # modern pip first
    pip_install(["-r", str(ROOT / "requirements.txt")])
    ok("core dependencies installed")


def install_nli(enabled: bool) -> None:
    step(3, "Semantic-check packages (requirements-nli.txt)")
    if not enabled:
        warn("skipped (--no-nli). 3A will run with model + safety gate only.")
        warn("enable later:  .venv/bin/pip install -r requirements-nli.txt  then set NLI_CHECK=on")
        return
    if sys.platform.startswith("linux"):
        # On Linux the default PyPI torch wheel drags in the full CUDA toolkit —
        # several GB of GPU-only libraries a CPU box can never use. Install the
        # official CPU-only build first (~200MB); the requirements file then sees
        # torch as already satisfied and only adds transformers/sentencepiece.
        print("   note: installing the CPU-only PyTorch build (a CPU box can't use the CUDA one; saves ~5GB).")
        proc = run([str(venv_python()), "-m", "pip", "install", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cpu"])
        if proc.returncode != 0:
            warn("CPU-only PyTorch index unreachable — falling back to the default (larger) build.")
    print("   note: large download on first run. One time.")
    # No --upgrade here: it would let pip 'upgrade' the CPU-only torch back to
    # the CUDA build if PyPI carries a newer version number.
    pip_install(["-r", str(ROOT / "requirements-nli.txt")], upgrade=False)
    ok("semantic-check dependencies installed")


def ensure_env(model: str) -> None:
    step(4, "Configuration (.env)")
    env_path = ROOT / ".env"
    if env_path.exists():
        current = read_env(env_path)
        if current.get("ANTHROPIC_API_KEY") or current.get("OPENAI_API_KEY") or current.get("GOOGLE_API_KEY"):
            warn(".env already has a cloud API key set — leaving it untouched.")
            warn("On a SHARED box, remove any real keys you don't want other users to see.")
        else:
            ok(".env already present — leaving it as is.")
        return
    # No .env yet: write a safe local-only one. Deliberately NO cloud API key.
    env_path.write_text(
        "# Auto-written by deploy.py for a local CPU box. No cloud key on purpose.\n"
        "MODEL_PROVIDER=local\n"
        f"LOCAL_MODEL={model}\n"
        f"CLEANUP_MODEL={model}\n"
        f"TRANSLATION_MODEL={model}\n"
        "# LOCAL_BASE_URL=http://127.0.0.1:11434/v1   # Ollama default\n"
        "# NLI_CHECK=off   # uncomment to turn the 3A semantic check off\n",
        encoding="utf-8",
    )
    ok(f"wrote a safe .env (provider=local, model={model}, no API key)")


def ollama_root() -> str:
    """Root URL of the Ollama server, derived from LOCAL_BASE_URL (strip /v1)."""
    base = read_env(ROOT / ".env").get("LOCAL_BASE_URL") or DEFAULT_OLLAMA
    return base.rstrip("/").removesuffix("/v1").rstrip("/")


def ollama_models(root: str) -> list[str] | None:
    """List of model names the server has, or None if the server isn't reachable."""
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def ensure_model(model: str, skip_pull: bool) -> None:
    step(5, "Local model server (Ollama)")
    root = ollama_root()
    names = ollama_models(root)

    if names is None:
        warn(f"couldn't reach the model server at {root}.")
        if shutil.which("ollama") is None:
            warn("Ollama isn't installed. Install it (one time), then re-run this script:")
            warn("    curl -fsSL https://ollama.com/install.sh | sh")
        else:
            warn("Ollama is installed but not responding. Start it, then re-run:")
            warn("    ollama serve      (on most Linux installs it auto-starts as a service)")
        die("model server not available — set it up and run deploy.py again.")

    ok(f"server reachable at {root}")
    base = model.split(":")[0]
    if any(n == model or n.split(":")[0] == base for n in names):
        ok(f"model '{model}' is already available")
        return

    if skip_pull:
        warn(f"model '{model}' not found and --skip-pull was set — the tools will fail without it.")
        return
    if shutil.which("ollama") is None:
        die(f"model '{model}' is missing and the 'ollama' command isn't on PATH to pull it.")
    print(f"   model '{model}' not found — pulling it now (one time, a few hundred MB)...")
    if run(["ollama", "pull", model]).returncode != 0:
        die(f"failed to pull '{model}'. Check the name and your connection, then re-run.")
    ok(f"pulled '{model}'")


def warm_nli(enabled: bool) -> None:
    step(6, "Pre-download the semantic-check model")
    if not enabled:
        ok("skipped (NLI not installed)")
        return
    print("   downloading the mDeBERTa model (~1.1GB, one time) so first use is fast...")
    # Best-effort: if this fails (e.g. no network), the server still downloads it
    # lazily on first request. Never let it block the deploy.
    proc = run(
        [str(venv_python()), "-c", "from src import nli_check; print('ready' if nli_check.warm_up() else 'unavailable')"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if "ready" in (proc.stdout or ""):
        ok("semantic-check model downloaded and ready")
    else:
        warn("could not pre-download it now — the server will fetch it on first use instead.")
        if proc.stderr.strip():
            warn(proc.stderr.strip().splitlines()[-1])


def start_server(host: str, port: int, setup_only: bool) -> None:
    step(7, "Web UI")
    if setup_only:
        ok("--setup-only set: everything is installed; not starting the server.")
        print(f"\n   Start it yourself with:")
        print(f"     {venv_python()} app.py --host {host} --port {port}")
        return
    print(f"   starting the web UI — open it in a browser at:  http://{host}:{port}")
    print("   (press Ctrl+C to stop)\n")
    # Foreground: this call runs the server until you stop it.
    subprocess.run([str(venv_python()), "app.py", "--host", host, "--port", str(port)], cwd=str(ROOT))


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="One-command deploy for Projects 2A + 3A.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to serve on. Use 0.0.0.0 to let other machines reach it (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765, help="port for the web UI (default: 8765)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"local model name to use (default: {DEFAULT_MODEL})")
    ap.add_argument("--no-nli", action="store_true", help="skip the ~2.5GB semantic-check stack (lighter box)")
    ap.add_argument("--setup-only", action="store_true", help="install and configure everything but don't start the server")
    ap.add_argument("--skip-pull", action="store_true", help="don't pull the model even if it's missing")
    args = ap.parse_args()

    nli = not args.no_nli
    print("=" * 68)
    print(" Deploying Service-Desk AI Utilities (Projects 2A + 3A)")
    print(f"   model: {args.model}    semantic check (NLI): {'ON' if nli else 'off'}    serve: {args.host}:{args.port}")
    print("=" * 68)

    ensure_venv()
    install_core()
    install_nli(nli)
    ensure_env(args.model)
    ensure_model(args.model, args.skip_pull)
    warm_nli(nli)
    start_server(args.host, args.port, args.setup_only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
