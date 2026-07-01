#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# End-to-end build and test script for the WebGPU backend (native via Dawn).
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
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/ops/test_add.py" -v
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/ops/test_rms_norm.py" -v

# ── Step 2: Export .pte model ─────────────────────────────────────────────────

echo "=== Step 2: Export test models ==="
DISPATCH_ORDER_DIR="/tmp/dispatch_order"
PTE_UPDATE_CACHE_MODEL="/tmp/webgpu_update_cache_test.pte"
cd "${EXECUTORCH_ROOT}"
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_dispatch_order import export_dispatch_order_cases
export_dispatch_order_cases('${DISPATCH_ORDER_DIR}')
"

echo "=== Export update_cache model ==="
UPDATE_CACHE_OK=1
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_update_cache import export_update_cache_model
export_update_cache_model('${PTE_UPDATE_CACHE_MODEL}')
" || { echo "WARN: update_cache export failed; skipping update_cache native test"; UPDATE_CACHE_OK=0; }

echo "=== Export SDPA sweep models (sdpa_<name>.pte + .golden.bin to /tmp) ==="
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_sdpa import export_all_sdpa_models
export_all_sdpa_models('/tmp')
" || echo "WARN: sdpa export failed; the native test self-skips configs whose .pte is absent"

echo "=== Export SDPA replay sequences (sdpa_<seq>_step<t>_S<S>_pos<p>.* to /tmp) ==="
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_sdpa import export_replay_sequences
export_replay_sequences('/tmp')
" || echo "WARN: sdpa replay export failed; the native test self-skips absent sequences"

echo "=== Export SDPA dynamic-input_pos decode (sdpa_dyn_<name>.* to /tmp) ==="
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_sdpa import export_dynamic_decode
export_dynamic_decode('/tmp')
" || echo "WARN: sdpa dynamic export failed; the native test self-skips when absent"

echo "=== Export SDPA in-graph-cache decode (sdpa_incache_<name>.* to /tmp) ==="
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_sdpa import export_incache_decode
export_incache_decode('/tmp')
" || echo "WARN: sdpa in-graph-cache export failed; the native test self-skips when absent"

# ── Step 3: Native build + test (Dawn + SwiftShader) ─────────────────────────

# Vendor Dawn (Tint) + SwiftShader and export Dawn_DIR/VK_ICD_FILENAMES. Set
# DAWN_PREBUILT_DIR to an existing Dawn install to skip the download locally.
echo "=== Installing Dawn (Tint) + SwiftShader ==="
source "${EXECUTORCH_ROOT}/.ci/scripts/setup-webgpu-linux-deps.sh"

echo "=== Step 3: Native build with Dawn ==="
NATIVE_BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out-webgpu-native"
rm -rf "${NATIVE_BUILD_DIR}"

cmake \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DDawn_DIR="${Dawn_DIR}" \
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
cmake --build "${NATIVE_BUILD_DIR}" --target webgpu_dispatch_order_test -j${NPROC}
cmake --build "${NATIVE_BUILD_DIR}" --target webgpu_scratch_buffer_test -j${NPROC}

echo "=== Step 4: Run native tests ==="
UPDATE_CACHE_ENV_VAR=""
if [[ "${UPDATE_CACHE_OK}" == "1" && -f "${PTE_UPDATE_CACHE_MODEL}" ]]; then
  UPDATE_CACHE_ENV_VAR="WEBGPU_TEST_UPDATE_CACHE_MODEL=${PTE_UPDATE_CACHE_MODEL}"
else
  echo "(skipping update_cache native test: export did not complete)"
fi
env \
    ${UPDATE_CACHE_ENV_VAR} \
    WEBGPU_TEST_SDPA_DIR=/tmp/ \
    "${NATIVE_BUILD_DIR}/backends/webgpu/webgpu_native_test"

"${NATIVE_BUILD_DIR}/backends/webgpu/webgpu_dispatch_order_test" "${DISPATCH_ORDER_DIR}"
"${NATIVE_BUILD_DIR}/backends/webgpu/webgpu_scratch_buffer_test"

echo "=== Done ==="
