#!/usr/bin/env bash
# Start the classic-style Jupyter Notebook for this course with the right Python kernels.
#
#   tools/start_notebook.sh            # opens the course folder in your browser
#   tools/start_notebook.sh --port 8899
#
# Why: every notebook asks for a kernel named "python3". If you already have a user-level
# "python3" kernel (e.g. pointing at a system Python), notebooks would run on the wrong
# interpreter. This script creates project-local kernels in .jupyter/ (git-ignored) and puts
# them first on JUPYTER_PATH — your global Jupyter setup is not modified.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KDIR="$REPO/.jupyter/kernels"
VENV_PY="$REPO/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "No .venv found. Create it first:  uv venv --python 3.12 && uv pip install -r requirements.txt" >&2
  exit 1
fi
if ! "$VENV_PY" -c "import notebook" 2>/dev/null; then
  echo "Jupyter Notebook is not installed in .venv. Run:  uv pip install -r requirements.txt  (or: uv pip install notebook)" >&2
  exit 1
fi

make_kernel() {  # name, display name, python executable
  mkdir -p "$KDIR/$1"
  cat > "$KDIR/$1/kernel.json" <<EOF
{"argv": ["$3", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "$2", "language": "python"}
EOF
}

# "python3" is the kernel the course notebooks request -> the course .venv
make_kernel python3 "Python 3 (AI course .venv)" "$VENV_PY"

# Notebooks whose libraries conflict with the main environment get their own kernels
for env in airflow kfp monitoring autogluon llamaindex; do
  py="$REPO/.venvs/$env/bin/python"
  if [ -x "$py" ]; then
    make_kernel "ai-course-$env" "AI course: $env env" "$py"
  fi
done

export JUPYTER_PATH="$REPO/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"

cat <<'MSG'
Kernels ready:
  • "Python 3 (AI course .venv)"  — picked automatically by almost every notebook
  • For these five, use Kernel → Change kernel… after opening:
      08_Generative_AI_LLM/04_LlamaIndex       → AI course: llamaindex env
      09_MLOps/04_Airflow                      → AI course: airflow env
      09_MLOps/05_Model_Monitoring_and_Drift   → AI course: monitoring env
      09_MLOps/06_Kubeflow_Pipelines           → AI course: kfp env
      12_AutoML_Experimentation/03_AutoGluon   → AI course: autogluon env
  • Module 08 and the LLM capstones also need the local LLM server:
      llama-server -m ~/.hermes/models/gpt-oss-20b-MXFP4.gguf --port 8080 --jinja
MSG

cd "$REPO/AI_Full_Stack_Engineer_Course"
exec "$REPO/.venv/bin/jupyter" notebook "$@"
