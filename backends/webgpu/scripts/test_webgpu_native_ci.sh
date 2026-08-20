#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build + run the WebGPU native test executables on Dawn (Tint) + SwiftShader.
# This is the substantive op-coverage gate: unlike the python operators suite
# (which only delegates add.Tensor to WebGPU, the rest CPU-fallback), these
# executables run quantized_linear / SDPA / update_cache / multi-dispatch
# ordering / scratch through the real WebGPU backend on Dawn. (Simple ops —
# add / rms_norm / the misc ops — run through the cases.py op-test framework.)
#
# Assumes the Dawn env is already sourced (Dawn_DIR + VK_ICD_FILENAMES +
# LD_LIBRARY_PATH) via .ci/scripts/setup-webgpu-linux-deps.sh. For local runs:
#   source .ci/scripts/setup-webgpu-linux-deps.sh
#   bash backends/webgpu/scripts/test_webgpu_native_ci.sh
#
# Builds and runs the fixed native target matrix defined by this tree. A missing
# target or fixture is a CI failure, not an optional skip.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out-webgpu-dawn-ci"

if [[ -z "${Dawn_DIR:-}" ]]; then
  echo "ERROR: Dawn_DIR not set. Source .ci/scripts/setup-webgpu-linux-deps.sh first." >&2
  exit 1
fi

cd "${EXECUTORCH_ROOT}"

# ── Exports for the model-driven executables ─────────────────────────────────
if ! "${PYTHON_EXECUTABLE}" -c "import executorch" 2>/dev/null; then
  echo "ERROR: executorch wheel unavailable; required fixture exports cannot run" >&2
  exit 1
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required WebGPU fixture missing: $1" >&2
    exit 1
  fi
}

run_with_required_device() {
  local output
  if ! output="$("$@" 2>&1)"; then
    printf '%s\n' "${output}"
    return 1
  fi
  printf '%s\n' "${output}"
  if ! grep -q '^WebGPU device acquired (native)$' <<<"${output}"; then
    echo "ERROR: WebGPU native test did not acquire a device" >&2
    return 1
  fi
}

DISPATCH_ORDER_DIR="/tmp/dispatch_order"
UPDATE_CACHE_DIR="/tmp/update_cache"
INDEX_DIR="/tmp/index"
DYNAMIC_SHAPE_DIR="/tmp/dynamic_shape"
ROPE_HF_DIR="/tmp/webgpu_rope_hf"
SYMINT_BLOB="/tmp/sdpa_dyn_small.pte"
OUTPUT_SUPPRESSION_DIR="/tmp/output_suppression"
EMBEDDING_MODEL="/tmp/webgpu_embedding_q4gsw.pte"
EMBEDDING_INDICES="/tmp/webgpu_embedding_q4gsw_indices.bin"
EMBEDDING_GOLDEN="/tmp/webgpu_embedding_q4gsw_golden.bin"
EMBEDDING_LLAMA1B_MODEL="/tmp/webgpu_embedding_q4gsw_llama1b.pte"
EMBEDDING_LLAMA1B_INDICES="/tmp/webgpu_embedding_q4gsw_llama1b_indices.bin"
EMBEDDING_LLAMA1B_GOLDEN="/tmp/webgpu_embedding_q4gsw_llama1b_golden.bin"
ROPE_MODEL="/tmp/webgpu_rope.pte"
ROPE_XQ_GOLDEN="/tmp/webgpu_rope_xq_golden.bin"
ROPE_XK_GOLDEN="/tmp/webgpu_rope_xk_golden.bin"
ROPE_DECODE_MODEL="/tmp/webgpu_rope_decode.pte"
ROPE_DECODE_XQ_GOLDEN="/tmp/webgpu_rope_decode_xq_golden.bin"
ROPE_DECODE_XK_GOLDEN="/tmp/webgpu_rope_decode_xk_golden.bin"
PREPACK_MODEL="/tmp/webgpu_prepack.pte"
PREPACK_GOLDEN="/tmp/webgpu_prepack_golden.bin"
PREPACK2_MODEL="/tmp/webgpu_prepack_two_const.pte"
PREPACK2_GOLDEN="/tmp/webgpu_prepack_two_const_golden.bin"
PREPACK_TIED_MODEL="/tmp/webgpu_prepack_tied_const.pte"
PREPACK_TIED_GOLDEN="/tmp/webgpu_prepack_tied_const_golden.bin"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_quantized_linear import export_all_quantized_linear_models, export_output_suppression_models
export_all_quantized_linear_models('/tmp')
export_output_suppression_models('${OUTPUT_SUPPRESSION_DIR}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_embedding_q4gsw import export_embedding_q4gsw_model
export_embedding_q4gsw_model('${EMBEDDING_MODEL}', '${EMBEDDING_GOLDEN}', '${EMBEDDING_INDICES}')
export_embedding_q4gsw_model('${EMBEDDING_LLAMA1B_MODEL}', '${EMBEDDING_LLAMA1B_GOLDEN}', '${EMBEDDING_LLAMA1B_INDICES}', 'llama1b')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_rope import export_rope_model
export_rope_model('${ROPE_MODEL}', '${ROPE_XQ_GOLDEN}', '${ROPE_XK_GOLDEN}')
export_rope_model('${ROPE_DECODE_MODEL}', '${ROPE_DECODE_XQ_GOLDEN}', '${ROPE_DECODE_XK_GOLDEN}', 'decode')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_rope_hf import export_rope_hf_dynamic
export_rope_hf_dynamic('${ROPE_HF_DIR}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_prepack import export_prepack_model, export_prepack_two_const_model, export_prepack_tied_const_model
export_prepack_model('${PREPACK_MODEL}', '${PREPACK_GOLDEN}')
export_prepack_two_const_model('${PREPACK2_MODEL}', '${PREPACK2_GOLDEN}')
export_prepack_tied_const_model('${PREPACK_TIED_MODEL}', '${PREPACK_TIED_GOLDEN}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_dispatch_order import export_dispatch_order_cases
export_dispatch_order_cases('${DISPATCH_ORDER_DIR}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_update_cache import (
    export_update_cache_cases,
    export_update_cache_replay,
    export_update_cache_negative,
)
export_update_cache_cases('${UPDATE_CACHE_DIR}')
export_update_cache_replay('${UPDATE_CACHE_DIR}')
export_update_cache_negative('${UPDATE_CACHE_DIR}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.index.test_index import export_all_index_models
export_all_index_models('${INDEX_DIR}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.dynamic_shape.test_dynamic_shape_export import export_dynamic_shape_cases
export_dynamic_shape_cases('${DYNAMIC_SHAPE_DIR}')
"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_sdpa import (
    export_all_sdpa_models,
    export_replay_sequences,
    export_dynamic_decode,
    export_incache_decode,
)
export_all_sdpa_models('/tmp')
export_replay_sequences('/tmp')
export_dynamic_decode('/tmp')
export_incache_decode('/tmp')
"

require_file "${ROPE_HF_DIR}/rope_hf_dynamic.pte"
require_file "${SYMINT_BLOB}"
require_file "${OUTPUT_SUPPRESSION_DIR}/input.bin"

# ── Configure (Dawn-only: no -DWEBGPU_IMPL; Dawn is the sole backend) ─────────
echo "=== Configure WebGPU native tests on Dawn ==="
rm -rf "${BUILD_DIR}"
cmake \
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DEXECUTORCH_BUILD_WEBGPU_TEST=ON \
    -DEXECUTORCH_BUILD_TESTS=ON \
    -DDawn_DIR="${Dawn_DIR}" \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -B "${BUILD_DIR}" \
    "${EXECUTORCH_ROOT}"

# ── Build + run every fixed native test target in this tree ──────────────────
REQUIRED_TARGETS=(webgpu_native_test webgpu_dispatch_order_test webgpu_scratch_buffer_test webgpu_update_cache_test webgpu_index_test webgpu_dynamic_shape_test webgpu_dispatch_2d_test webgpu_compute_dispatch_test webgpu_execution_options_test webgpu_output_suppression_test webgpu_op_test_util_test)
BIN_DIR="${BUILD_DIR}/backends/webgpu"

DEFINED_TARGETS="$(cmake --build "${BUILD_DIR}" --target help 2>/dev/null || true)"

for t in "${REQUIRED_TARGETS[@]}"; do
  if ! printf '%s\n' "${DEFINED_TARGETS}" | grep -qw "${t}"; then
    echo "ERROR: required CMake target is not defined: ${t}" >&2
    exit 1
  fi
  cmake --build "${BUILD_DIR}" --target "${t}" -j"${NPROC}"
  echo "built ${t}"
done

echo "=== Run native tests on Dawn + SwiftShader ==="
run_with_required_device env WEBGPU_TEST_SDPA_DIR=/tmp/ \
    WEBGPU_TEST_QUANTIZED_LINEAR_DIR=/tmp/ \
    WEBGPU_TEST_EMBEDDING_Q4GSW_MODEL="${EMBEDDING_MODEL}" \
    WEBGPU_TEST_EMBEDDING_Q4GSW_INDICES="${EMBEDDING_INDICES}" \
    WEBGPU_TEST_EMBEDDING_Q4GSW_GOLDEN="${EMBEDDING_GOLDEN}" \
    WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_MODEL="${EMBEDDING_LLAMA1B_MODEL}" \
    WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_INDICES="${EMBEDDING_LLAMA1B_INDICES}" \
    WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_GOLDEN="${EMBEDDING_LLAMA1B_GOLDEN}" \
    WEBGPU_TEST_ROPE_MODEL="${ROPE_MODEL}" \
    WEBGPU_TEST_ROPE_XQ_GOLDEN="${ROPE_XQ_GOLDEN}" \
    WEBGPU_TEST_ROPE_XK_GOLDEN="${ROPE_XK_GOLDEN}" \
    WEBGPU_TEST_ROPE_DECODE_MODEL="${ROPE_DECODE_MODEL}" \
    WEBGPU_TEST_ROPE_DECODE_XQ_GOLDEN="${ROPE_DECODE_XQ_GOLDEN}" \
    WEBGPU_TEST_ROPE_DECODE_XK_GOLDEN="${ROPE_DECODE_XK_GOLDEN}" \
    WEBGPU_TEST_ROPE_HF_DIR="${ROPE_HF_DIR}" \
    WEBGPU_TEST_SYMINT_BLOB="${SYMINT_BLOB}" \
    WEBGPU_TEST_PREPACK_MODEL="${PREPACK_MODEL}" \
    WEBGPU_TEST_PREPACK_GOLDEN="${PREPACK_GOLDEN}" \
    WEBGPU_TEST_PREPACK2_MODEL="${PREPACK2_MODEL}" \
    WEBGPU_TEST_PREPACK2_GOLDEN="${PREPACK2_GOLDEN}" \
    WEBGPU_TEST_PREPACK_TIED_MODEL="${PREPACK_TIED_MODEL}" \
    WEBGPU_TEST_PREPACK_TIED_GOLDEN="${PREPACK_TIED_GOLDEN}" \
    "${BIN_DIR}/webgpu_native_test"
"${BIN_DIR}/webgpu_update_cache_test" "${UPDATE_CACHE_DIR}"
"${BIN_DIR}/webgpu_dispatch_order_test" "${DISPATCH_ORDER_DIR}"
"${BIN_DIR}/webgpu_index_test" "${INDEX_DIR}"
"${BIN_DIR}/webgpu_dynamic_shape_test" "${DYNAMIC_SHAPE_DIR}"
"${BIN_DIR}/webgpu_scratch_buffer_test"
"${BIN_DIR}/webgpu_dispatch_2d_test"
"${BIN_DIR}/webgpu_compute_dispatch_test"
"${BIN_DIR}/webgpu_execution_options_test"
"${BIN_DIR}/webgpu_output_suppression_test" "${OUTPUT_SUPPRESSION_DIR}"
"${BIN_DIR}/webgpu_op_test_util_test"

echo "=== WebGPU native tests on Dawn: all run targets passed ==="

# ── Op-test codegen framework: generate manifest → build → run (Dawn+SwiftShader) ──
# Generate the op-test manifest, build the target from the existing test-enabled
# configuration, then run every op in cases.py against its torch golden.
OP_TEST_DIR="/tmp/webgpu_op_tests"
$PYTHON_EXECUTABLE -m executorch.backends.webgpu.test.op_tests.generate_op_tests \
  --output "${OP_TEST_DIR}"
cmake --build "${BUILD_DIR}" --target webgpu_op_test -j"${NPROC}"
"${BIN_DIR}/webgpu_op_test" --manifest "${OP_TEST_DIR}/manifest.json"
echo "=== WebGPU op-test framework on Dawn: passed ==="
