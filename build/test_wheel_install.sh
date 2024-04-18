#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage:
#   ./build/test_wheel_install.sh <path-to-whl>|<path-to-zipped-whl>
#
# Installs the wheel in a fresh conda environment named ${CONDA_ENV} and tries
# creating and running an XNNPACK-delegated .pte file.
#
# Must be run from the root of an `executorch` repo so that it can use the
# XNNPACK example code to create a .pte file.

set -euxo pipefail

CONDA_ENV=executorch-tmp
PYTHON_VERSION=3.10.0

# Create a temp dir that we can create files under.
WORK_DIR="$(mktemp -d /tmp/test-wheel-install-XXXXXXXXXX)"
readonly WORK_DIR

# Creates or resets a conda environment named ${CONDA_ENV}
clean_conda() {
  eval "$(conda shell.bash hook)"
  conda activate base
  conda remove -y --name "${CONDA_ENV}" --all || echo "(ignoring error)"
  conda create -yn "${CONDA_ENV}" python="${PYTHON_VERSION}"
  conda activate "${CONDA_ENV}"
}

test_xnnpack_e2e() {
  # Create a .pte file that delegates to XNNPACK.
  local pte="mv2_xnnpack_fp32.pte"
  rm -f "${pte}"  # Make sure it's a fresh one.
  python3 -m examples.xnnpack.aot_compiler --model_name="mv2" --delegate
  test -f "${pte}"

  # Test python script
  local test_py="${WORK_DIR}/test_xnnpack_e2e.py"
  cat > "${test_py}" << HERE
from executorch.extension.pybindings import portable_lib
m = portable_lib._load_for_executorch("${pte}")

# Import torch after importing and using portable_lib to demonstrate that
# portable_lib works without importing this first.
import torch
t = torch.randn((1, 3, 224, 224))

output = m.forward([t])
assert len(output) == 1, f"Unexpected output length {len(output)}"
assert output[0].size() == torch.Size([1, 1000]), f"Unexpected output size {output[0].size()}"
print("PASS")
HERE
  (
    set +x
    echo "===== BEGIN ${test_py} ====="
    cat "${test_py}"
    echo "===== END ${test_py} ====="
  )

  python "${test_py}"
}

main() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: $(basename "$0") <path-to-whl>|<path-to-zipped-whl>" >&2
    exit 1
  fi
  local wheel="$1"

  if [[ "${wheel}" == *".zip" ]]; then
    local unzip_dir="${WORK_DIR}/unzip"
    unzip -d "${unzip_dir}" "${wheel}"
    wheel="$(ls "${unzip_dir}"/*.whl | head -1)"
  fi

  # Create a fresh conda environment.
  clean_conda

  # Install the minimal deps.
  pip install --extra-index-url "https://download.pytorch.org/whl/test/cpu" \
      torch=="2.3.0" \
      torchvision=="0.18.0"

  # Install the provided wheel.
  pip install "${wheel}"

  # Try creating and running a .pte file that delegates to XNNPACK.
  test_xnnpack_e2e

  # Only delete on success so we can debug failures.
  rm -rf "${WORK_DIR}"
}

main "$@"
