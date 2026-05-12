#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Smoke tests for backends/native: incrementally build the runtime, export
# tiny models, and run them through the test binaries.
#
# Tests:
#   1. test_native_simple  — TinyAdd (a + b). No constants. Exercises
#                            HostExtern IO + a single CPU/Metal segment.
#   2. test_native_linear  — ConstantyModel with three frozen tensor
#                            constants. Exercises upload_constants
#                            end-to-end (each constant must be
#                            materialized on the consuming engine).
#   3. test_native_scalar  — ScalarAddModel: forward(x) = x + 1.0.
#                            Exercises the scalar-EValue path through
#                            aten::add.Scalar.
#
# This script does NOT delete anything. It only:
#   1. Incrementally builds cmake-out/{test_native_simple,test_native_linear}
#      (cmake --build). Configure + build are no-ops if nothing changed.
#   2. Runs the python exporters to write /tmp/native_simple.pte and
#      /tmp/native_linear.{pte,ref}.
#   3. Runs each test binary against its model.
#
# Usage (from any directory):
#   third-party/executorch/backends/native/build_and_run_tests.sh
#   third-party/.../build_and_run_tests.sh --compute-unit=cpu
#   third-party/.../build_and_run_tests.sh --compute-unit=metal
#   third-party/.../build_and_run_tests.sh --compute-unit="cpu|metal"
#
# --compute-unit (default: auto) is forwarded to NativeBackend's
# load-time `compute_unit` option via NATIVE_COMPUTE_UNIT env var on
# every test run. "auto" means "all available providers".
#
# Or override the model paths:
#   NATIVE_SIMPLE_PTE_PATH=/tmp/foo.pte \
#   NATIVE_LINEAR_PTE_PATH=/tmp/bar.pte \
#   NATIVE_LINEAR_REF_PATH=/tmp/bar.ref \
#     third-party/.../build_and_run_tests.sh

set -euo pipefail

# --- Arg parsing -------------------------------------------------------
COMPUTE_UNIT="auto"
for arg in "$@"; do
    case "${arg}" in
        --compute-unit=*)
            COMPUTE_UNIT="${arg#--compute-unit=}"
            ;;
        --help|-h)
            sed -n '8,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown arg: ${arg}" >&2
            echo "Try --help" >&2
            exit 2
            ;;
    esac
done
export NATIVE_COMPUTE_UNIT="${COMPUTE_UNIT}"

# Resolve script + repo locations.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXPORT_SIMPLE_PY="${SCRIPT_DIR}/test/export_simple_model.py"
EXPORT_LINEAR_PY="${SCRIPT_DIR}/test/export_linear_model.py"
EXPORT_SCALAR_PY="${SCRIPT_DIR}/test/export_scalar_model.py"
EXPORT_CHAIN_PY="${SCRIPT_DIR}/test/export_chain_model.py"
EXPORT_TRI_PY="${SCRIPT_DIR}/test/export_tri_model.py"
EXPORT_STATEFUL_PY="${SCRIPT_DIR}/test/export_stateful_model.py"
EXPORT_INPLACE_PY="${SCRIPT_DIR}/test/export_inplace_model.py"
EXPORT_DYN_PY="${SCRIPT_DIR}/test/export_dyn_model.py"
EXPORT_COND_PY="${SCRIPT_DIR}/test/export_cond_model.py"
EXPORT_COND_INNER_PY="${SCRIPT_DIR}/test/export_cond_inner.py"

# Repo root: third-party/executorch/backends/native -> ../../../..
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/../../../.." &> /dev/null && pwd )"
# Use a dedicated build dir to avoid conflicting with any pre-existing
# cmake-out that may have been generated against a different source root
# (e.g. configured directly from third-party/executorch/ rather than the
# wrapper at the repo root).
CMAKE_OUT="${REPO_ROOT}/cmake-out-native-test"
TEST_BIN_SIMPLE="${CMAKE_OUT}/test_native_simple"
TEST_BIN_LINEAR="${CMAKE_OUT}/test_native_linear"
TEST_BIN_SCALAR="${CMAKE_OUT}/test_native_scalar"
TEST_BIN_CHAIN="${CMAKE_OUT}/test_native_chain"
TEST_BIN_TRI="${CMAKE_OUT}/test_native_tri_provider"
TEST_BIN_STATEFUL="${CMAKE_OUT}/test_native_stateful"
TEST_BIN_INPLACE="${CMAKE_OUT}/test_native_inplace"
TEST_BIN_DYN="${CMAKE_OUT}/test_native_dyn_shapes"
TEST_BIN_COND="${CMAKE_OUT}/test_native_cond"
TEST_BIN_COND_INNER="${CMAKE_OUT}/test_native_cond_inner"

CHAIN_PTE="${NATIVE_CHAIN_PTE_PATH:-/tmp/native_chain.pte}"
CHAIN_REF="${NATIVE_CHAIN_REF_PATH:-/tmp/native_chain.ref}"
TRI_PTE="${NATIVE_TRI_PTE_PATH:-/tmp/native_tri.pte}"
TRI_REF="${NATIVE_TRI_REF_PATH:-/tmp/native_tri.ref}"
STATEFUL_PTE="${NATIVE_STATEFUL_PTE_PATH:-/tmp/native_stateful.pte}"
INPLACE_PTE="${NATIVE_INPLACE_PTE_PATH:-/tmp/native_inplace.pte}"
DYN_PTE="${NATIVE_DYN_PTE_PATH:-/tmp/native_dyn.pte}"
COND_PTE="${NATIVE_COND_PTE_PATH:-/tmp/native_cond.pte}"
COND_INNER_FBB="${NATIVE_COND_INNER_PATH:-/tmp/native_cond_inner.fbb}"

SIMPLE_PTE="${NATIVE_SIMPLE_PTE_PATH:-/tmp/native_simple.pte}"
LINEAR_PTE="${NATIVE_LINEAR_PTE_PATH:-/tmp/native_linear.pte}"
LINEAR_REF="${NATIVE_LINEAR_REF_PATH:-/tmp/native_linear.ref}"
SCALAR_PTE="${NATIVE_SCALAR_PTE_PATH:-/tmp/native_scalar.pte}"
SCALAR_REF="${NATIVE_SCALAR_REF_PATH:-/tmp/native_scalar.ref}"

echo "=== build_and_run_tests.sh ==="
echo "  Repo root:       ${REPO_ROOT}"
echo "  Build dir:       ${CMAKE_OUT}"
echo "  Compute unit:    ${COMPUTE_UNIT}  (forwarded as NATIVE_COMPUTE_UNIT)"
echo "  Simple binary:   ${TEST_BIN_SIMPLE}"
echo "  Linear binary:   ${TEST_BIN_LINEAR}"
echo "  Scalar binary:   ${TEST_BIN_SCALAR}"
echo "  Chain binary:    ${TEST_BIN_CHAIN}"
echo "  Simple PTE:      ${SIMPLE_PTE}"
echo "  Linear PTE:      ${LINEAR_PTE}"
echo "  Linear ref:      ${LINEAR_REF}"
echo "  Scalar PTE:      ${SCALAR_PTE}"
echo "  Scalar ref:      ${SCALAR_REF}"
echo "  Chain PTE:       ${CHAIN_PTE}"
echo "  Chain ref:       ${CHAIN_REF}"
echo

# 1. Configure (idempotent — only updates files inside cmake-out, never
# touches source). Then incrementally build. Re-running configure is
# necessary when CMakeLists.txt has changed (e.g., new targets added)
# since the cmake-out was last generated.
if [[ ! -d "${CMAKE_OUT}" ]]; then
    echo "  cmake-out missing — running fresh configure..."
fi
echo "=== Step 1a/21: cmake configure (regenerates build system if stale) ==="
CMAKE_PREFIX_PATH="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
cmake \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DET_MIN_LOG_LEVEL=Info \
    -S "${REPO_ROOT}" \
    -B "${CMAKE_OUT}"

echo
echo "=== Step 1b/21: incremental build of test binaries ==="
cmake --build "${CMAKE_OUT}" -j 16 --config Release \
    --target test_native_simple test_native_linear test_native_scalar test_native_chain test_native_tri_provider test_native_stateful test_native_inplace test_native_dyn_shapes test_native_cond test_native_cond_inner

for bin in "${TEST_BIN_SIMPLE}" "${TEST_BIN_LINEAR}" "${TEST_BIN_SCALAR}" "${TEST_BIN_CHAIN}" "${TEST_BIN_TRI}" "${TEST_BIN_STATEFUL}" "${TEST_BIN_INPLACE}" "${TEST_BIN_DYN}" "${TEST_BIN_COND}" "${TEST_BIN_COND_INNER}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "ERROR: ${bin} not found after build." >&2
        exit 1
    fi
done

# 2. Export the simple model.
echo
echo "=== Step 2/21: exporting TinyAdd to ${SIMPLE_PTE} ==="
NATIVE_SIMPLE_PTE_PATH="${SIMPLE_PTE}" python "${EXPORT_SIMPLE_PY}"
if [[ ! -f "${SIMPLE_PTE}" ]]; then
    echo "ERROR: simple export did not produce ${SIMPLE_PTE}" >&2
    exit 1
fi

# 3. Run the simple test.
echo
echo "=== Step 3/21: running test_native_simple ==="
ET_TESTING_MODEL_PATH="${SIMPLE_PTE}" "${TEST_BIN_SIMPLE}"

# 4. Export the linear MLP (writes both .pte and reference output).
echo
echo "=== Step 4/21: exporting ConstantyModel to ${LINEAR_PTE} (+ ref ${LINEAR_REF}) ==="
NATIVE_LINEAR_PTE_PATH="${LINEAR_PTE}" \
NATIVE_LINEAR_REF_PATH="${LINEAR_REF}" \
    python "${EXPORT_LINEAR_PY}"
if [[ ! -f "${LINEAR_PTE}" ]]; then
    echo "ERROR: linear export did not produce ${LINEAR_PTE}" >&2
    exit 1
fi
if [[ ! -f "${LINEAR_REF}" ]]; then
    echo "ERROR: linear export did not produce reference ${LINEAR_REF}" >&2
    exit 1
fi

# 5. Run the linear test.
echo
echo "=== Step 5/21: running test_native_linear (exercises upload_constants) ==="
ET_TESTING_MODEL_PATH="${LINEAR_PTE}" \
NATIVE_LINEAR_REF_PATH="${LINEAR_REF}" \
    "${TEST_BIN_LINEAR}"

# 6. Export the scalar model.
echo
echo "=== Step 6/21: exporting ScalarAddModel to ${SCALAR_PTE} (+ ref ${SCALAR_REF}) ==="
NATIVE_SCALAR_PTE_PATH="${SCALAR_PTE}" \
NATIVE_SCALAR_REF_PATH="${SCALAR_REF}" \
    python "${EXPORT_SCALAR_PY}"
if [[ ! -f "${SCALAR_PTE}" ]]; then
    echo "ERROR: scalar export did not produce ${SCALAR_PTE}" >&2
    exit 1
fi
if [[ ! -f "${SCALAR_REF}" ]]; then
    echo "ERROR: scalar export did not produce reference ${SCALAR_REF}" >&2
    exit 1
fi

# 7. Run the scalar test.
echo
echo "=== Step 7/21: running test_native_scalar (exercises aten::add.Scalar) ==="
ET_TESTING_MODEL_PATH="${SCALAR_PTE}" \
NATIVE_SCALAR_REF_PATH="${SCALAR_REF}" \
    "${TEST_BIN_SCALAR}"

echo
echo "=== Step 8/21: exporting ChainModel to ${CHAIN_PTE} (+ ref ${CHAIN_REF}) ==="
NATIVE_CHAIN_PTE_PATH="${CHAIN_PTE}" \
NATIVE_CHAIN_REF_PATH="${CHAIN_REF}" \
    python "${EXPORT_CHAIN_PY}"
if [[ ! -f "${CHAIN_PTE}" ]]; then
    echo "ERROR: chain export did not produce ${CHAIN_PTE}" >&2
    exit 1
fi
if [[ ! -f "${CHAIN_REF}" ]]; then
    echo "ERROR: chain export did not produce reference ${CHAIN_REF}" >&2
    exit 1
fi

echo
echo "=== Step 9/21: running test_native_chain (chain of unary ops; in-place + multi-op segment) ==="
ET_TESTING_MODEL_PATH="${CHAIN_PTE}" \
NATIVE_CHAIN_REF_PATH="${CHAIN_REF}" \
    "${TEST_BIN_CHAIN}"

echo
echo "=== Step 10/21: exporting TriProviderModel to ${TRI_PTE} (+ ref ${TRI_REF}) ==="
NATIVE_TRI_PTE_PATH="${TRI_PTE}" \
NATIVE_TRI_REF_PATH="${TRI_REF}" \
    python "${EXPORT_TRI_PY}"
if [[ ! -f "${TRI_PTE}" ]]; then
    echo "ERROR: tri export did not produce ${TRI_PTE}" >&2
    exit 1
fi
if [[ ! -f "${TRI_REF}" ]]; then
    echo "ERROR: tri export did not produce reference ${TRI_REF}" >&2
    exit 1
fi

echo
echo "=== Step 11/21: running test_native_tri_provider (fake_accel + metal + cpu in one plan) ==="
ET_TESTING_MODEL_PATH="${TRI_PTE}" \
NATIVE_TRI_REF_PATH="${TRI_REF}" \
    "${TEST_BIN_TRI}"


echo
echo "=== Step 12/21: exporting StatefulAdd to ${STATEFUL_PTE} ==="
NATIVE_STATEFUL_PTE_PATH="${STATEFUL_PTE}" python "${EXPORT_STATEFUL_PY}"
if [[ ! -f "${STATEFUL_PTE}" ]]; then echo "ERROR: stateful export failed" >&2; exit 1; fi

echo
echo "=== Step 13/21: running test_native_stateful (mutable buffer across calls) ==="
ET_TESTING_MODEL_PATH="${STATEFUL_PTE}" "${TEST_BIN_STATEFUL}"

echo
echo "=== Step 14/21: exporting InplaceModel to ${INPLACE_PTE} ==="
NATIVE_INPLACE_PTE_PATH="${INPLACE_PTE}" python "${EXPORT_INPLACE_PY}"
if [[ ! -f "${INPLACE_PTE}" ]]; then echo "ERROR: inplace export failed" >&2; exit 1; fi

echo
echo "=== Step 15/21: running test_native_inplace (explicit add_/mul_/relu_) ==="
ET_TESTING_MODEL_PATH="${INPLACE_PTE}" "${TEST_BIN_INPLACE}"

echo
echo "=== Step 16/21: exporting DynModel to ${DYN_PTE} ==="
NATIVE_DYN_PTE_PATH="${DYN_PTE}" python "${EXPORT_DYN_PY}"
if [[ ! -f "${DYN_PTE}" ]]; then echo "ERROR: dyn export failed" >&2; exit 1; fi

echo
echo "=== Step 17/21: running test_native_dyn_shapes (batch={1,3,5,8}) ==="
ET_TESTING_MODEL_PATH="${DYN_PTE}" "${TEST_BIN_DYN}"


echo
echo "=== Step 18/21: exporting CondModel to ${COND_PTE} ==="
NATIVE_COND_PTE_PATH="${COND_PTE}" python "${EXPORT_COND_PY}"
if [[ ! -f "${COND_PTE}" ]]; then echo "ERROR: cond export failed" >&2; exit 1; fi

echo
echo "=== Step 19/21: running test_native_cond (torch.cond, both branches) ==="
ET_TESTING_MODEL_PATH="${COND_PTE}" "${TEST_BIN_COND}"


echo
echo "=== Step 20/21: exporting raw cond inner program to ${COND_INNER_FBB} ==="
NATIVE_COND_INNER_PATH="${COND_INNER_FBB}" python "${EXPORT_COND_INNER_PY}"
if [[ ! -f "${COND_INNER_FBB}" ]]; then echo "ERROR: cond inner export failed" >&2; exit 1; fi

echo
echo "=== Step 21/21: running test_native_cond_inner (bypass: cond into NativeBackend init) ==="
NATIVE_COND_INNER_PATH="${COND_INNER_FBB}" "${TEST_BIN_COND_INNER}"

echo
echo "=== build_and_run_tests.sh: ALL PASS ==="
