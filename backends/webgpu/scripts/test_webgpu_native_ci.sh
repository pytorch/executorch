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
# Builds whatever native test targets are present in the landed tree (NOT a fixed
# list): webgpu_native_test (base) + webgpu_dispatch_order_test,
# webgpu_scratch_buffer_test (D107576199) + webgpu_update_cache_test
# (D107547307). SDPA executables join once they land.

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

# ── Exports for the model-driven executables (best-effort) ───────────────────
# native_test (quantized_linear/SDPA/update_cache) + dispatch_order read .pte/
# golden inputs via env/dir and self-skip if absent; scratch is standalone.
# native_test itself is gated below on the executorch wheel being importable.
DISPATCH_ORDER_DIR="/tmp/dispatch_order"
DISPATCH_ORDER_OK=1
UPDATE_CACHE_DIR="/tmp/update_cache"
UPDATE_CACHE_OK=1
INDEX_DIR="/tmp/index"
INDEX_OK=1
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
from executorch.backends.webgpu.test.ops.test_quantized_linear import export_all_quantized_linear_models
export_all_quantized_linear_models('/tmp')
" || echo "WARN: q4gsw export failed; required configs will FAIL in webgpu_native_test"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_embedding_q4gsw import export_embedding_q4gsw_model
export_embedding_q4gsw_model('${EMBEDDING_MODEL}', '${EMBEDDING_GOLDEN}', '${EMBEDDING_INDICES}')
export_embedding_q4gsw_model('${EMBEDDING_LLAMA1B_MODEL}', '${EMBEDDING_LLAMA1B_GOLDEN}', '${EMBEDDING_LLAMA1B_INDICES}', 'llama1b')
" || echo "WARN: embedding_q4gsw export failed; embedding configs will FAIL in webgpu_native_test"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_rope import export_rope_model
export_rope_model('${ROPE_MODEL}', '${ROPE_XQ_GOLDEN}', '${ROPE_XK_GOLDEN}')
export_rope_model('${ROPE_DECODE_MODEL}', '${ROPE_DECODE_XQ_GOLDEN}', '${ROPE_DECODE_XK_GOLDEN}', 'decode')
" || echo "WARN: rope export failed; apply_rotary_emb configs will FAIL in webgpu_native_test"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_prepack import export_prepack_model, export_prepack_two_const_model, export_prepack_tied_const_model
export_prepack_model('${PREPACK_MODEL}', '${PREPACK_GOLDEN}')
export_prepack_two_const_model('${PREPACK2_MODEL}', '${PREPACK2_GOLDEN}')
export_prepack_tied_const_model('${PREPACK_TIED_MODEL}', '${PREPACK_TIED_GOLDEN}')
" || echo "WARN: prepack export failed; prepack configs will FAIL in webgpu_native_test"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_dispatch_order import export_dispatch_order_cases
export_dispatch_order_cases('${DISPATCH_ORDER_DIR}')
" || { echo "WARN: dispatch_order export failed; skipping dispatch_order native test"; DISPATCH_ORDER_OK=0; }

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.test_update_cache import (
    export_update_cache_cases,
    export_update_cache_replay,
    export_update_cache_negative,
)
export_update_cache_cases('${UPDATE_CACHE_DIR}')
export_update_cache_replay('${UPDATE_CACHE_DIR}')
export_update_cache_negative('${UPDATE_CACHE_DIR}')
" || { echo "WARN: update_cache export failed; skipping update_cache native test"; UPDATE_CACHE_OK=0; }

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.index.test_index import export_all_index_models
export_all_index_models('${INDEX_DIR}')
" || { echo "WARN: index export failed; skipping index native test"; INDEX_OK=0; }

# Non-fatal: a failed sdpa export makes the required 4k/8k configs hard-fail in
# webgpu_native_test below (precise per-config error), so don't exit/mask here.
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
" || echo "WARN: sdpa export failed; required 4k/8k configs will FAIL in webgpu_native_test"

# ── Configure (Dawn-only: no -DWEBGPU_IMPL; Dawn is the sole backend) ─────────
echo "=== Configure WebGPU native tests on Dawn ==="
rm -rf "${BUILD_DIR}"
cmake \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DEXECUTORCH_BUILD_WEBGPU_TEST=ON \
    -DDawn_DIR="${Dawn_DIR}" \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -B "${BUILD_DIR}" \
    "${EXECUTORCH_ROOT}"

# ── Build + run every native test target that exists in this tree ────────────
TARGETS=(webgpu_native_test webgpu_dispatch_order_test webgpu_scratch_buffer_test webgpu_update_cache_test webgpu_index_test)
BIN_DIR="${BUILD_DIR}/backends/webgpu"

# Which targets are defined depends on which diffs are landed (native_test +
# rms_norm here; dispatch_order + scratch from D107576199). Query the configured
# target list ONCE so a not-yet-landed target is skipped WITHOUT masking a real
# compile failure of a target that IS defined (CI uses the Make generator).
DEFINED_TARGETS="$(cmake --build "${BUILD_DIR}" --target help 2>/dev/null || true)"

# Fail loud if the probe found nothing (e.g. a non-Make generator or a cmake
# regression): otherwise every target would skip and the job would go green
# having tested nothing. webgpu_native_test is always defined at/after this diff.
if ! printf '%s\n' "${DEFINED_TARGETS}" | grep -qw webgpu_native_test; then
  echo "ERROR: cmake target probe returned no webgpu_native_test; aborting" >&2
  exit 1
fi

for t in "${TARGETS[@]}"; do
  if printf '%s\n' "${DEFINED_TARGETS}" | grep -qw "${t}"; then
    # Defined target: build with stderr visible; set -e fails the job on a real
    # build error (never silently skipped).
    cmake --build "${BUILD_DIR}" --target "${t}" -j"${NPROC}"
    echo "built ${t}"
  else
    echo "(target ${t} not defined in this tree — skipping)"
  fi
done

echo "=== Run native tests on Dawn + SwiftShader ==="
# webgpu_native_test hosts the quantized_linear / SDPA / update_cache / symint
# sweeps. Gate on the executorch wheel being importable (the proxy for "the
# exports above ran"): CI has the wheel so they ran; a bare local run without it
# skips here rather than hard-failing the required-config guards.
if [[ -x "${BIN_DIR}/webgpu_native_test" ]] &&
  "${PYTHON_EXECUTABLE}" -c "import executorch" 2>/dev/null; then
  env WEBGPU_TEST_SDPA_DIR=/tmp/ \
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
      WEBGPU_TEST_PREPACK_MODEL="${PREPACK_MODEL}" \
      WEBGPU_TEST_PREPACK_GOLDEN="${PREPACK_GOLDEN}" \
      WEBGPU_TEST_PREPACK2_MODEL="${PREPACK2_MODEL}" \
      WEBGPU_TEST_PREPACK2_GOLDEN="${PREPACK2_GOLDEN}" \
      WEBGPU_TEST_PREPACK_TIED_MODEL="${PREPACK_TIED_MODEL}" \
      WEBGPU_TEST_PREPACK_TIED_GOLDEN="${PREPACK_TIED_GOLDEN}" \
      "${BIN_DIR}/webgpu_native_test"
else
  echo "(skipping webgpu_native_test: executorch wheel absent — exports did not run)"
fi
if [[ "${UPDATE_CACHE_OK}" == "1" && -x "${BIN_DIR}/webgpu_update_cache_test" ]]; then
  "${BIN_DIR}/webgpu_update_cache_test" "${UPDATE_CACHE_DIR}"
fi
if [[ "${DISPATCH_ORDER_OK}" == "1" && -x "${BIN_DIR}/webgpu_dispatch_order_test" ]]; then
  "${BIN_DIR}/webgpu_dispatch_order_test" "${DISPATCH_ORDER_DIR}"
fi
if [[ "${INDEX_OK}" == "1" && -x "${BIN_DIR}/webgpu_index_test" ]]; then
  "${BIN_DIR}/webgpu_index_test" "${INDEX_DIR}"
fi
[[ -x "${BIN_DIR}/webgpu_scratch_buffer_test" ]] && "${BIN_DIR}/webgpu_scratch_buffer_test"

echo "=== WebGPU native tests on Dawn: all run targets passed ==="

# ── Op-test codegen framework: generate manifest → build → run (Dawn+SwiftShader) ──
# Reconfigure the SAME build dir adding GTest (EXECUTORCH_BUILD_TESTS=ON), then run
# every op in cases.py against its torch golden. Self-skips if the generator can't run.
OP_TEST_DIR="/tmp/webgpu_op_tests"
if $PYTHON_EXECUTABLE -m executorch.backends.webgpu.test.op_tests.generate_op_tests \
    --output "${OP_TEST_DIR}"; then
  echo "=== Reconfigure with GTest + build/run op-test framework ==="
  cmake -DEXECUTORCH_BUILD_TESTS=ON -B "${BUILD_DIR}" "${EXECUTORCH_ROOT}"
  OP_DEFINED="$(cmake --build "${BUILD_DIR}" --target help 2>/dev/null || true)"
  if printf '%s\n' "${OP_DEFINED}" | grep -qw webgpu_op_test_util_test; then
    cmake --build "${BUILD_DIR}" --target webgpu_op_test_util_test -j"${NPROC}"
    "${BIN_DIR}/webgpu_op_test_util_test"
  fi
  if printf '%s\n' "${OP_DEFINED}" | grep -qw webgpu_op_test; then
    cmake --build "${BUILD_DIR}" --target webgpu_op_test -j"${NPROC}"
    "${BIN_DIR}/webgpu_op_test" --manifest "${OP_TEST_DIR}/manifest.json"
  fi
  echo "=== WebGPU op-test framework on Dawn: passed ==="
else
  echo "WARN: op-test manifest generation failed (needs the executorch wheel); skipping"
fi
