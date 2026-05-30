#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_VENV="${REPO_ROOT}/.venv-export-only-build"
TEST_VENV="${REPO_ROOT}/.venv-export-only-test"

rm -rf "${BUILD_VENV}" "${TEST_VENV}" "${REPO_ROOT}/dist" "${REPO_ROOT}/pip-out"

"${PYTHON_EXECUTABLE}" -m venv "${BUILD_VENV}"
source "${BUILD_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "cmake>=3.24,<4.0.0" \
  "numpy>=2.0.0" \
  packaging \
  pyyaml \
  setuptools \
  wheel \
  zstd \
  certifi \
  torch \
  torchvision \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

(
  cd "${REPO_ROOT}"
  EXECUTORCH_BUILD_EXPORT_ONLY=1 python setup.py bdist_wheel
)

WHEEL_FILE="$(find "${REPO_ROOT}/dist" -maxdepth 1 -name 'executorch-*.whl' | head -1)"
test -n "${WHEEL_FILE}"

python - "${WHEEL_FILE}" <<'PY'
import sys
import zipfile

wheel_file = sys.argv[1]
with zipfile.ZipFile(wheel_file) as wheel:
    names = wheel.namelist()

for forbidden in (
    "executorch/backends/",
    "executorch/examples/",
    "executorch/kernels/",
    "executorch/runtime/",
    "executorch/devtools/",
    "executorch/extension/pybindings/",
):
    matches = [name for name in names if name.startswith(forbidden)]
    if matches:
        raise AssertionError(f"{wheel_file} unexpectedly contains {matches[:5]}")

extensions = [
    name
    for name in names
    if name.endswith((".so", ".dylib", ".dll", ".pyd")) and "flatc" not in name
]
if extensions:
    raise AssertionError(f"{wheel_file} unexpectedly contains extensions: {extensions}")
PY

deactivate

"${PYTHON_EXECUTABLE}" -m venv "${TEST_VENV}"
source "${TEST_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "flatbuffers" \
  "numpy>=2.0.0" \
  "sympy" \
  "torch" \
  "torchvision" \
  "typing-extensions>=4.10.0" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
python -m pip install --no-deps "${WHEEL_FILE}"

python - <<'PY'
from pathlib import Path

import torch
from torch.export import export
from torchvision.models import mobilenet_v2

from executorch.exir import to_edge_transform_and_lower

model = mobilenet_v2(weights=None).eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

edge_program = to_edge_transform_and_lower(export(model, example_inputs))
executorch_program = edge_program.to_executorch()

output_path = Path("mv2_export_only.pte")
with output_path.open("wb") as output_file:
    executorch_program.write_to_file(output_file)

assert output_path.stat().st_size > 0
PY
