#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build ExecuTorch and its locally-built dependencies (torchao, tokenizers)
# as Python wheels into the output directory passed as $1.
#
# Designed to be invoked from a CI "build artifact" job. The produced wheels
# can then be uploaded with `upload-artifact:` and consumed by downstream
# jobs via `install_executorch.sh --prebuilt-wheel-dir <path>`.
#
# Caller-controlled environment variables that influence the build (must
# match downstream consumer expectations):
#   EXECUTORCH_BUILD_KERNELS_TORCHAO
#   TORCHAO_BUILD_EXPERIMENTAL_MPS
#   CMAKE_ARGS
#   MACOSX_DEPLOYMENT_TARGET  (e.g. 14.0 to make the wheel installable on
#                              older macOS runners in the same cluster)
#
# Usage:
#   build_macos_wheels.sh <output-dir>
#
# Output:
#   <output-dir>/torchao-*.whl
#   <output-dir>/pytorch_tokenizers-*.whl
#   <output-dir>/executorch-*.whl

set -euxo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-dir>" >&2
  exit 1
fi

OUTPUT_DIR="$1"
WHEEL_DIR="${OUTPUT_DIR}"
mkdir -p "${WHEEL_DIR}"

# cd to repo root regardless of invocation directory.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )"
cd "${REPO_ROOT}"

# Ensure all required submodules are populated before any build runs.
git submodule sync --recursive
git submodule update --init --recursive

# Install runtime/build dependencies.
# We need torch installed before invoking `pip wheel` on the local sources
# because their build hooks `import torch`. Single source of truth for the
# pinned torch version + requirements-dev.txt is install_requirements.py.
PYTHON="${PYTHON_EXECUTABLE:-python}"
"${PYTHON}" -c "from install_requirements import install_torch_and_dev_requirements; install_torch_and_dev_requirements(use_pytorch_nightly=False)"

# Build torchao wheel. install_requirements.py sets USE_CPP/CMAKE_POLICY_VERSION_MINIMUM
# based on EXECUTORCH_BUILD_KERNELS_TORCHAO; replicate that here so the produced
# wheel matches what install_executorch.sh would have built.
if [[ "${EXECUTORCH_BUILD_KERNELS_TORCHAO:-0}" == "1" ]]; then
  export USE_CPP=1
  export CMAKE_POLICY_VERSION_MINIMUM="3.5"
else
  export USE_CPP=0
fi

"${PYTHON}" -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${WHEEL_DIR}" \
  ./third-party/ao

# Install the just-built torchao so the executorch wheel build (which
# imports torchao at build time in some configurations) succeeds.
"${PYTHON}" -m pip install "${WHEEL_DIR}"/torchao-*.whl

# Build the tokenizers wheel (parity with install_requirements LOCAL_REQUIREMENTS).
"${PYTHON}" -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${WHEEL_DIR}" \
  ./extension/llm/tokenizers

"${PYTHON}" -m pip install "${WHEEL_DIR}"/pytorch_tokenizers-*.whl

# Finally, build the executorch wheel. CMAKE_ARGS / EXECUTORCH_BUILD_KERNELS_TORCHAO
# from the caller's environment are honored by the build backend.
"${PYTHON}" -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${WHEEL_DIR}" \
  .

echo "Built wheels:"
ls -lah "${WHEEL_DIR}"
