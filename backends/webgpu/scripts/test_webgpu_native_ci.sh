#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build + run the WebGPU native test executables on Dawn (Tint) + SwiftShader.
# This is the substantive op-coverage gate: unlike the python operators suite
# (which only delegates add.Tensor to WebGPU, the rest CPU-fallback), these
# executables run rms_norm / multi-dispatch ordering / scratch through the real
# WebGPU backend on Dawn.
#
# Assumes the Dawn env is already sourced (Dawn_DIR + VK_ICD_FILENAMES +
# LD_LIBRARY_PATH) via .ci/scripts/setup-webgpu-linux-deps.sh. For local runs:
#   source .ci/scripts/setup-webgpu-linux-deps.sh
#   bash backends/webgpu/scripts/test_webgpu_native_ci.sh
#
# Builds whatever native test targets are present in the landed tree (NOT a fixed
# list). This stack lands: webgpu_native_test, webgpu_rms_norm_test (base) +
# webgpu_dispatch_order_test, webgpu_scratch_buffer_test (D107576199) +
# webgpu_update_cache_test (D107547307). SDPA executables join once they land.

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
# native_test + rms_norm + dispatch_order read .pte/golden inputs via env/dir and
# self-skip if absent; scratch is standalone (generates its own inputs).
PTE_MODEL="/tmp/webgpu_add_test.pte"
PTE_CHAINED_MODEL="/tmp/webgpu_chained_add_test.pte"
RMS_NORM_DIR="/tmp/rmsn"
RMS_NORM_OK=1
DISPATCH_ORDER_DIR="/tmp/dispatch_order"
DISPATCH_ORDER_OK=1
UPDATE_CACHE_DIR="/tmp/update_cache"
UPDATE_CACHE_OK=1
EMBEDDING_MODEL="/tmp/webgpu_embedding_q4gsw.pte"
EMBEDDING_INDICES="/tmp/webgpu_embedding_q4gsw_indices.bin"
EMBEDDING_GOLDEN="/tmp/webgpu_embedding_q4gsw_golden.bin"
ROPE_MODEL="/tmp/webgpu_rope.pte"
ROPE_XQ_GOLDEN="/tmp/webgpu_rope_xq_golden.bin"
ROPE_XK_GOLDEN="/tmp/webgpu_rope_xk_golden.bin"
PREPACK_MODEL="/tmp/webgpu_prepack.pte"
PREPACK_GOLDEN="/tmp/webgpu_prepack_golden.bin"
PREPACK2_MODEL="/tmp/webgpu_prepack_mul_add.pte"
PREPACK2_GOLDEN="/tmp/webgpu_prepack_mul_add_golden.bin"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.add.test_add import export_add_model, export_chained_add_model
export_add_model('${PTE_MODEL}')
export_chained_add_model('${PTE_CHAINED_MODEL}')
" || echo "WARN: add export failed; webgpu_native_test self-skips models whose .pte is absent"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.quantized_linear.test_quantized_linear import export_all_quantized_linear_models
export_all_quantized_linear_models('/tmp')
" || echo "WARN: q4gsw export failed; required configs will FAIL in webgpu_native_test"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.embedding_q4gsw.test_embedding_q4gsw import export_embedding_q4gsw_model
export_embedding_q4gsw_model('${EMBEDDING_MODEL}', '${EMBEDDING_GOLDEN}', '${EMBEDDING_INDICES}')
" || echo "WARN: embedding_q4gsw export failed; webgpu_native_test embedding case self-skips"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.rope.test_rope import export_rope_model
export_rope_model('${ROPE_MODEL}', '${ROPE_XQ_GOLDEN}', '${ROPE_XK_GOLDEN}')
" || echo "WARN: rope export failed; webgpu_native_test apply_rotary_emb case self-skips"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.prepack.test_prepack import export_prepack_model, export_prepack_mul_add_model
export_prepack_model('${PREPACK_MODEL}', '${PREPACK_GOLDEN}')
export_prepack_mul_add_model('${PREPACK2_MODEL}', '${PREPACK2_GOLDEN}')
" || echo "WARN: prepack export failed; webgpu_native_test prepack cases self-skip"

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.rms_norm.test_rms_norm import export_rms_norm_cases
export_rms_norm_cases('${RMS_NORM_DIR}')
" || { echo "WARN: rms_norm export failed; skipping rms_norm native test"; RMS_NORM_OK=0; }

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.dispatch_order.test_dispatch_order import export_dispatch_order_cases
export_dispatch_order_cases('${DISPATCH_ORDER_DIR}')
" || { echo "WARN: dispatch_order export failed; skipping dispatch_order native test"; DISPATCH_ORDER_OK=0; }

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.sdpa.test_update_cache import (
    export_update_cache_cases,
    export_update_cache_replay,
    export_update_cache_negative,
)
export_update_cache_cases('${UPDATE_CACHE_DIR}')
export_update_cache_replay('${UPDATE_CACHE_DIR}')
export_update_cache_negative('${UPDATE_CACHE_DIR}')
" || { echo "WARN: update_cache export failed; skipping update_cache native test"; UPDATE_CACHE_OK=0; }

# Non-fatal: a failed sdpa export makes the required 4k/8k configs hard-fail in
# webgpu_native_test below (precise per-config error), so don't exit/mask here.
$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.sdpa.test_sdpa import (
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
TARGETS=(webgpu_native_test webgpu_rms_norm_test webgpu_dispatch_order_test webgpu_scratch_buffer_test webgpu_update_cache_test)
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
# native_test is model-driven; only run it if the export produced its .pte
# (CI's setup-linux.sh provides the executorch wheel so exports succeed; a bare
# local run without the wheel self-skips here rather than hard-failing on load).
if [[ -x "${BIN_DIR}/webgpu_native_test" && -f "${PTE_MODEL}" ]]; then
  env WEBGPU_TEST_MODEL="${PTE_MODEL}" \
      WEBGPU_TEST_CHAINED_MODEL="${PTE_CHAINED_MODEL}" \
      WEBGPU_TEST_SDPA_DIR=/tmp/ \
      WEBGPU_TEST_QUANTIZED_LINEAR_DIR=/tmp/ \
      WEBGPU_TEST_EMBEDDING_Q4GSW_MODEL="${EMBEDDING_MODEL}" \
      WEBGPU_TEST_EMBEDDING_Q4GSW_INDICES="${EMBEDDING_INDICES}" \
      WEBGPU_TEST_EMBEDDING_Q4GSW_GOLDEN="${EMBEDDING_GOLDEN}" \
      WEBGPU_TEST_ROPE_MODEL="${ROPE_MODEL}" \
      WEBGPU_TEST_ROPE_XQ_GOLDEN="${ROPE_XQ_GOLDEN}" \
      WEBGPU_TEST_ROPE_XK_GOLDEN="${ROPE_XK_GOLDEN}" \
      WEBGPU_TEST_PREPACK_MODEL="${PREPACK_MODEL}" \
      WEBGPU_TEST_PREPACK_GOLDEN="${PREPACK_GOLDEN}" \
      WEBGPU_TEST_PREPACK2_MODEL="${PREPACK2_MODEL}" \
      WEBGPU_TEST_PREPACK2_GOLDEN="${PREPACK2_GOLDEN}" \
      "${BIN_DIR}/webgpu_native_test"
else
  echo "(skipping webgpu_native_test: no exported .pte — needs the executorch python wheel)"
fi
if [[ "${RMS_NORM_OK}" == "1" && -x "${BIN_DIR}/webgpu_rms_norm_test" ]]; then
  "${BIN_DIR}/webgpu_rms_norm_test" "${RMS_NORM_DIR}"
fi
if [[ "${UPDATE_CACHE_OK}" == "1" && -x "${BIN_DIR}/webgpu_update_cache_test" ]]; then
  "${BIN_DIR}/webgpu_update_cache_test" "${UPDATE_CACHE_DIR}"
fi
if [[ "${DISPATCH_ORDER_OK}" == "1" && -x "${BIN_DIR}/webgpu_dispatch_order_test" ]]; then
  "${BIN_DIR}/webgpu_dispatch_order_test" "${DISPATCH_ORDER_DIR}"
fi
[[ -x "${BIN_DIR}/webgpu_scratch_buffer_test" ]] && "${BIN_DIR}/webgpu_scratch_buffer_test"

echo "=== WebGPU native tests on Dawn: all run targets passed ==="
