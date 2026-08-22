#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_VENV="${REPO_ROOT}/.venv-minimal-build"
TEST_VENV="${REPO_ROOT}/.venv-minimal-test"

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
  EXECUTORCH_BUILD_MINIMAL=1 python setup.py bdist_wheel
)

WHEEL_FILE="$(find "${REPO_ROOT}/dist" -maxdepth 1 -name 'executorch-*.whl' | head -1)"
test -n "${WHEEL_FILE}"

python - "${WHEEL_FILE}" <<'PY'
import re
import sys
import zipfile

wheel_file = sys.argv[1]
with zipfile.ZipFile(wheel_file) as wheel:
    names = wheel.namelist()
    metadata_name = next(
        (name for name in names if name.endswith(".dist-info/METADATA")), None
    )
    if metadata_name is None:
        raise AssertionError(f"{wheel_file} has no METADATA")
    metadata_text = wheel.read(metadata_name).decode("utf-8")

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


def _dist_name(requirement):
    name = re.split(r"[ ;\[<>=!~(]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


# Only the core (non-extra) Requires-Dist entries define what a plain
# "pip install" pulls; ignore the optional extras (cortex_m, vgf, ...).
declared = {
    _dist_name(line.split(":", 1)[1])
    for line in metadata_text.splitlines()
    if line.startswith("Requires-Dist:") and "extra==" not in line.replace(" ", "")
}
# The minimal wheel must declare EXACTLY this core set and nothing else -- the
# same names as `keep` in setup.py:_minimal_dependencies(). Exact match catches
# both a heavy full-wheel dep leaking in (coremltools, pandas, or a re-added
# mpmath/torch) and a required dep going missing.
expected = {
    "flatbuffers",
    "numpy",
    "packaging",
    "pyyaml",
    "ruamel-yaml",
    "sympy",
    "tabulate",
    "typing-extensions",
}
if declared != expected:
    raise AssertionError(
        f"{wheel_file} minimal core deps mismatch: "
        f"unexpected={sorted(declared - expected)} missing={sorted(expected - declared)}"
    )
PY

deactivate

"${PYTHON_EXECUTABLE}" -m venv "${TEST_VENV}"
source "${TEST_VENV}/bin/activate"
python -m pip install --upgrade pip
# torch and torchvision are needed to export a model but are intentionally not
# declared as wheel dependencies (consumers are expected to bring their own).
python -m pip install \
  "torch" \
  "torchvision" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
# Install the minimal wheel WITHOUT --no-deps so pip resolves its declared
# dependencies, confirming the slim set is correct and resolvable. (That no heavy
# deps sneak in is guaranteed by the METADATA exact-match check above, which
# covers the wheel's direct Requires-Dist.)
python -m pip install \
  "${WHEEL_FILE}" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

# flatc is the only compiled artifact in the minimal wheel and the reason it is
# platform specific. Confirm it ships, resolves through _get_flatc_path() (the
# executorch.data.bin lookup added for this build mode), and actually runs.
python - <<'PY'
import subprocess

from executorch.exir._serialize._flatbuffer import _get_flatc_path

flatc_path = _get_flatc_path()
print(f"flatc resolved to: {flatc_path}")
subprocess.run([flatc_path, "--version"], check=True)
PY

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

output_path = Path("mv2_minimal.pte")
with output_path.open("wb") as output_file:
    executorch_program.write_to_file(output_file)

assert output_path.stat().st_size > 0
PY
