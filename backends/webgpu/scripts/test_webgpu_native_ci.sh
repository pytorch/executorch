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

$PYTHON_EXECUTABLE -c "
from executorch.backends.webgpu.test.ops.add.test_add import export_add_model, export_chained_add_model
export_add_model('${PTE_MODEL}')
export_chained_add_model('${PTE_CHAINED_MODEL}')
" || echo "WARN: add export failed; webgpu_native_test self-skips models whose .pte is absent"

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
