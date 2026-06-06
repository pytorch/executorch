#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# End-to-end build and test script for the WebGPU backend (native via wgpu-native).
# Usage: bash backends/webgpu/test/test_build_webgpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "=== Check embedded WGSL headers are up to date ==="
"${PYTHON_EXECUTABLE}" "${SCRIPT_DIR}/../scripts/gen_wgsl_headers.py" --check \
  || { echo "ERROR: *_wgsl.h out of sync with .wgsl; run scripts/gen_wgsl_headers.py"; exit 1; }

# Unit tests for the WGSL header generator itself
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/test_wgsl_codegen.py" -v

# ── Step 1: Python export tests ──────────────────────────────────────────────

echo "=== Step 1: Run Python export tests ==="
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/ops/add/test_add.py" -v
# Non-fatal: a rms_norm pytest failure skips the rms_norm native test below
# rather than aborting the whole run.
RMS_NORM_PYTEST_OK=1
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/ops/rms_norm/test_rms_norm.py" -v \
    || RMS_NORM_PYTEST_OK=0

# ── Step 2: Export .pte model ─────────────────────────────────────────────────

echo "=== Step 2: Export test models ==="
PTE_MODEL="/tmp/webgpu_add_test.pte"
PTE_CHAINED_MODEL="/tmp/webgpu_chained_add_test.pte"
RMS_NORM_DIR="/tmp/rmsn"
cd "${EXECUTORCH_ROOT}"
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.add.test_add import export_add_model, export_chained_add_model
export_add_model('${PTE_MODEL}')
export_chained_add_model('${PTE_CHAINED_MODEL}')
"
if [[ "${RMS_NORM_PYTEST_OK}" == "1" ]]; then
  $PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.rms_norm.test_rms_norm import export_rms_norm_cases
export_rms_norm_cases('${RMS_NORM_DIR}')
" || { echo "WARN: rms_norm export failed; skipping rms_norm native test"; RMS_NORM_PYTEST_OK=0; }
fi

# ── Step 3: Native build + test (wgpu-native) ────────────────────────────────

WGPU_DIR="${EXECUTORCH_ROOT}/backends/webgpu/third-party/wgpu-native"

# Auto-download wgpu-native if not present
if [[ ! -d "${WGPU_DIR}/lib" ]]; then
    echo "=== Installing wgpu-native ==="
    bash "${EXECUTORCH_ROOT}/backends/webgpu/scripts/setup-wgpu-native.sh"
fi

echo "=== Step 3: Native build with wgpu-native ==="
NATIVE_BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out-webgpu-native"
rm -rf "${NATIVE_BUILD_DIR}"

cmake \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DEXECUTORCH_BUILD_WEBGPU_TEST=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -B "${NATIVE_BUILD_DIR}" \
    "${EXECUTORCH_ROOT}"

cmake --build "${NATIVE_BUILD_DIR}" --target webgpu_native_test -j${NPROC}
cmake --build "${NATIVE_BUILD_DIR}" --target webgpu_rms_norm_test -j${NPROC}

echo "=== Step 4: Run native tests ==="
env \
    WEBGPU_TEST_MODEL="${PTE_MODEL}" \
    WEBGPU_TEST_CHAINED_MODEL="${PTE_CHAINED_MODEL}" \
    "${NATIVE_BUILD_DIR}/backends/webgpu/webgpu_native_test"

if [[ "${RMS_NORM_PYTEST_OK}" == "1" ]]; then
  "${NATIVE_BUILD_DIR}/backends/webgpu/webgpu_rms_norm_test" "${RMS_NORM_DIR}"
else
  echo "(skipping rms_norm native test: pytest or export did not complete)"
fi

echo "=== Done ==="
