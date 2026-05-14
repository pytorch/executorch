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

# ── Step 1: Python export tests ──────────────────────────────────────────────

echo "=== Step 1: Run Python export test ==="
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/ops/add/test_add.py" -v

# ── Step 2: Export .pte model ─────────────────────────────────────────────────

echo "=== Step 2: Export test model ==="
PTE_MODEL="/tmp/webgpu_add_test.pte"
cd "${EXECUTORCH_ROOT}"
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.add.test_add import export_add_model
export_add_model('${PTE_MODEL}')
"

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

echo "=== Step 4: Run native test ==="
WEBGPU_TEST_MODEL="${PTE_MODEL}" \
    "${NATIVE_BUILD_DIR}/backends/webgpu/webgpu_native_test"

echo "=== Done ==="
