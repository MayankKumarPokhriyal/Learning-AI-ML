#!/usr/bin/env python3
"""Check that your machine is ready for the course — run it right after installing.

    python tools/check_setup.py          # main .venv: packages, key imports, kernel, optional extras
    python tools/check_setup.py --quick  # skip the (slower) import tests

✅ ready · ⚠️ works, but read the note · ❌ must fix · ⏭️ optional and not set up (only some notebooks need it)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
EXTRA_ENVS = {  # env -> notebook that needs it
    "llamaindex": "08_Generative_AI_LLM/04_LlamaIndex",
    "airflow": "09_MLOps/04_Airflow",
    "monitoring": "09_MLOps/05_Model_Monitoring_and_Drift",
    "kfp": "09_MLOps/06_Kubeflow_Pipelines",
    "autogluon": "12_AutoML_Experimentation/03_AutoGluon",
}
# Imported in a fresh process each, so one broken native library can't hide the others.
SMOKE_IMPORTS = {
    "numpy": "import numpy",
    "pandas": "import pandas",
    "scikit-learn": "import sklearn",
    "XGBoost (needs OpenMP)": "import xgboost",
    "LightGBM (needs OpenMP)": "import lightgbm",
    "PyTorch": "import torch; torch.zeros(2) + 1",
    "FAISS": "import faiss",
    "Transformers": "import transformers",
    "spaCy + en_core_web_sm": "import spacy; spacy.load('en_core_web_sm')",
    "OpenCV": "import cv2",
    "Jupyter Notebook": "import notebook, nbclient, ipykernel",
    "Agent frameworks (OpenAI Agents SDK, PydanticAI, smolagents)": "import agents, pydantic_ai, smolagents",
    "Audio I/O (soundfile)": "import soundfile",
}
problems = 0


def say(mark, text, note=""):
    global problems
    problems += mark == "❌"
    print(f"  {mark} {text}" + (f"\n       {note}" if note else ""))


def env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if WINDOWS else "bin/python")


def pinned(req_file: Path) -> dict:
    pins = {}
    for line in req_file.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def compare_versions(python: Path, req_file: Path):
    """Return (missing, different) for the pins in req_file, checked inside `python`'s environment."""
    code = (
        "import json,sys\nfrom importlib import metadata\nout={}\n"
        "for d in json.loads(sys.argv[1]):\n"
        "    try: out[d]=metadata.version(d)\n"
        "    except metadata.PackageNotFoundError: out[d]=None\n"
        "print(json.dumps(out))"
    )
    pins = pinned(req_file)
    res = subprocess.run([str(python), "-c", code, json.dumps(list(pins))], capture_output=True, text=True, timeout=120)
    found = json.loads(res.stdout)
    missing = [d for d, v in found.items() if v is None]
    different = [f"{d} {found[d]} (course tested {pins[d]})" for d, v in found.items() if v and v != pins[d]]
    return missing, different


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the import tests")
    args = ap.parse_args()

    print("\n🐍 Python")
    v = sys.version_info
    say("✅" if (v.major, v.minor) == (3, 12) else "⚠️", f"Python {v.major}.{v.minor}.{v.micro}",
        "" if (v.major, v.minor) == (3, 12) else "The course is tested on Python 3.12 — create the venv with: uv venv --python 3.12")
    venv = REPO / ".venv"
    in_venv = Path(sys.prefix).resolve() == venv.resolve() if venv.exists() else False
    if not venv.exists():
        say("❌", "No .venv in the repository", "Create it: uv venv --python 3.12  then  uv pip install -r requirements.txt")
    elif not in_venv:
        say("⚠️", "This script is not running from the course .venv",
            "Activate it first (source .venv/bin/activate · Windows: .venv\\Scripts\\activate), then re-run.")
    else:
        say("✅", "Running inside the course .venv")

    print("\n📦 Main environment (requirements.txt)")
    missing, different = compare_versions(Path(sys.executable), REPO / "requirements.txt")
    total = len(pinned(REPO / "requirements.txt"))
    if missing:
        say("❌", f"{len(missing)} of {total} packages missing: {', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}",
            "Run: uv pip install -r requirements.txt")
    else:
        say("✅", f"All {total} packages installed")
    if different:
        say("⚠️", f"{len(different)} packages differ from the tested versions: {', '.join(different[:6])}{' …' if len(different) > 6 else ''}",
            "Usually fine. If a notebook misbehaves, re-run: uv pip install -r requirements.txt")

    if not args.quick:
        print("\n🔌 Import tests (each in its own process)")
        for label, stmt in SMOKE_IMPORTS.items():
            try:
                r = subprocess.run([sys.executable, "-c", stmt], capture_output=True, text=True, timeout=300)
                err = (r.stderr.strip().splitlines() or [""])[-1]
            except subprocess.TimeoutExpired:
                r, err = None, "timed out"
            if r is not None and r.returncode == 0:
                say("✅", label)
            else:
                hint = ""
                if "OpenMP" in label and sys.platform == "darwin":
                    hint = " — on macOS run: brew install libomp"
                if "spaCy" in label:
                    hint = " — run: uv pip install -r requirements.txt (it includes the en_core_web_sm model)"
                say("❌", label, f"{err[:200]}{hint}")

    print("\n📓 Jupyter kernel")
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        spec = KernelSpecManager().get_kernel_spec("python3")
        exe = spec.argv[0]
        uses_venv = exe in ("python", "python3", "{python}") or Path(exe).resolve() == Path(sys.executable).resolve()
        if uses_venv:
            say("✅", "The default 'python3' kernel uses this environment")
        else:
            say("⚠️", "Your default 'python3' kernel points at a different Python",
                "Notebooks would run outside the course .venv. Start Jupyter with tools/start_notebook.sh "
                "(macOS/Linux), or pick the .venv interpreter with Kernel → Change kernel.")
    except Exception as exc:  # jupyter_client missing or broken
        say("❌", "Could not inspect Jupyter kernels", f"{type(exc).__name__}: {exc}")

    print("\n🧩 Separate environments (only for the notebook listed)")
    for env, nb in EXTRA_ENVS.items():
        py = env_python(REPO / ".venvs" / env)
        if not py.exists():
            say("⏭️", f"{env:<10} not created — needed only for {nb}")
            continue
        miss, _ = compare_versions(py, REPO / f"requirements-{env}.txt")
        say("❌" if miss else "✅", f"{env:<10} .venvs/{env}", f"missing: {', '.join(miss)}" if miss else "")

    print("\n🛠️  Optional system tools")
    # Same lookup order as the PySpark notebook: JAVA_HOME, Homebrew's openjdk@17, then java on PATH
    java_home = os.environ.get("JAVA_HOME") or next(
        (str(p) for p in (Path("/opt/homebrew/opt/openjdk@17"), Path("/usr/local/opt/openjdk@17")) if p.exists()), None)
    java = str(Path(java_home) / "bin" / ("java.exe" if WINDOWS else "java")) if java_home else shutil.which("java")
    major = 0
    if java and Path(java).exists():
        banner = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
        m = re.search(r'version "(\d+)', banner)
        major = int(m.group(1)) if m else 0
    if major >= 17:
        say("✅", f"Java {major} (PySpark)")
    elif major:
        say("⚠️", f"Java {major} found — PySpark needs Java 17 or 21", "Install Java 17 and point JAVA_HOME at it.")
    else:
        say("⏭️", "Java 17 not found — needed only for 11_Data_Processing/02_PySpark (see README → system tools)")
    # In a child process: Playwright's driver can print asyncio noise when the interpreter exits
    probe = ("from pathlib import Path\nfrom playwright.sync_api import sync_playwright\n"
             "with sync_playwright() as pw:\n    print(Path(pw.chromium.executable_path).exists())")
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
        chromium_ok = r.returncode == 0 and r.stdout.strip().endswith("True")
        playwright_missing = "ModuleNotFoundError" in r.stderr
    except subprocess.TimeoutExpired:
        chromium_ok, playwright_missing = False, False
    if chromium_ok:
        say("✅", "Playwright Chromium installed")
    elif playwright_missing:
        say("⏭️", "Playwright not installed — needed only for 16_Agentic_AI/03 (run: uv pip install -r requirements.txt)")
    else:
        say("⏭️", "Playwright Chromium not installed — needed only for 16_Agentic_AI/03 (run: python -m playwright install chromium)")
    say("✅" if shutil.which("ffmpeg") else "⏭️", "FFmpeg found (audio decoding)" if shutil.which("ffmpeg") else
        "FFmpeg not found — some audio decoding in 17_Multimodal_and_Generative_Models/03 may need it (brew/apt install ffmpeg)")
    docker = shutil.which("docker")
    running = docker and subprocess.run([docker, "info"], capture_output=True).returncode == 0
    say("✅" if running else "⏭️", "Docker is running" if running else
        "Docker not running — needed only for 10_Model_Serving/02_Docker_and_CI_CD and capstone container builds")
    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
    try:
        req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'local')}"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            models = [m.get("id") for m in json.load(resp).get("data", [])]
        say("✅", f"LLM server at {base} — models: {', '.join(map(str, models)) or 'none listed'}")
    except Exception:
        say("⏭️", f"No LLM server at {base} — needed for module 08 and the LLM capstones (see README → Local LLM)")

    print()
    if problems:
        print(f"❌ {problems} problem(s) to fix before starting — see the notes above.")
        sys.exit(1)
    print("🎉 Ready. Start with: tools/start_notebook.sh  (or: jupyter notebook) and open 00_Foundations/01_Python_Basics.ipynb")


if __name__ == "__main__":
    main()
