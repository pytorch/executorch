#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11.0}"
ENV_DIR="${ENV_DIR:-.venvs/repro_mypy_ci}"
TARGET_FILE="${1:-backends/arm/_passes/fold_scalar_mul_into_conv_pass.py}"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found" >&2
  exit 1
fi

eval "$(pyenv init -)"

if [ ! -x "$(pyenv root)/versions/${PYTHON_VERSION}/bin/python" ]; then
  pyenv install "${PYTHON_VERSION}"
fi

PYTHON_BIN="$(pyenv prefix "${PYTHON_VERSION}")/bin/python"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  mkdir -p "$(dirname "${ENV_DIR}")"
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

run_in_env() {
  PATH="$(cd "${ENV_DIR}/bin" && pwd):${PATH}" "${ENV_DIR}/bin/$1" "${@:2}"
}

run_in_env python -m pip install --upgrade pip
run_in_env python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
run_in_env python -m pip install lintrunner==0.12.7 lintrunner-adapters==0.14.0
run_in_env python -m pip install -r requirements-lintrunner.txt
run_in_env lintrunner init

echo "=== Versions ==="
run_in_env python - <<'PY'
import sys
import torch
import mypy.version

print("python", sys.version)
print("torch", torch.__version__)
print("mypy", mypy.version.__version__)
PY

echo "=== lintrunner --take MYPY ${TARGET_FILE} ==="
run_in_env lintrunner --take MYPY "${TARGET_FILE}"
