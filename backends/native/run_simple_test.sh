#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Smoke test for backends/native: incrementally build the runtime, export
# a tiny model, and run it through cmake-out/test_model.
#
# This script does NOT delete anything. It only:
#   1. Incrementally builds cmake-out/test_model (cmake --build).
#      Configure + build are no-ops if nothing changed.
#   2. Runs the python exporter to write /tmp/native_simple.pte.
#   3. Runs cmake-out/test_model with that .pte path.
#
# Usage (from any directory):
#   third-party/executorch/backends/native/run_simple_test.sh
#
# Or override the model path:
#   NATIVE_SIMPLE_PTE_PATH=/tmp/foo.pte third-party/.../run_simple_test.sh

set -euo pipefail

# Resolve script + repo locations.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXPORT_PY="${SCRIPT_DIR}/test/export_simple_model.py"

# Repo root: third-party/executorch/backends/native -> ../../../..
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/../../../.." &> /dev/null && pwd )"
# Use a dedicated build dir to avoid conflicting with any pre-existing
# cmake-out that may have been generated against a different source root
# (e.g. configured directly from third-party/executorch/ rather than the
# wrapper at the repo root).
CMAKE_OUT="${REPO_ROOT}/cmake-out-native-test"
TEST_BIN="${CMAKE_OUT}/test_native_simple"

PTE_PATH="${NATIVE_SIMPLE_PTE_PATH:-/tmp/native_simple.pte}"

echo "=== run_simple_test.sh ==="
echo "  Repo root:    ${REPO_ROOT}"
echo "  Build dir:    ${CMAKE_OUT}"
echo "  Test binary:  ${TEST_BIN}"
echo "  PTE path:     ${PTE_PATH}"
echo

# 1. Configure (idempotent — only updates files inside cmake-out, never
# touches source). Then incrementally build. Re-running configure is
# necessary when CMakeLists.txt has changed (e.g., new targets added)
# since the cmake-out was last generated.
if [[ ! -d "${CMAKE_OUT}" ]]; then
    echo "  cmake-out missing — running fresh configure..."
fi
echo "=== Step 1a/3: cmake configure (regenerates build system if stale) ==="
CMAKE_PREFIX_PATH="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
cmake \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_OUT}" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -S "${REPO_ROOT}" \
    -B "${CMAKE_OUT}"

echo
echo "=== Step 1b/3: incremental build of test_native_simple ==="
cmake --build "${CMAKE_OUT}" -j 16 --config Release --target test_native_simple

if [[ ! -x "${TEST_BIN}" ]]; then
    echo "ERROR: ${TEST_BIN} not found after build." >&2
    exit 1
fi

# 2. Export the model. Sets NATIVE_SIMPLE_PTE_PATH so the python script
# knows where to write.
echo
echo "=== Step 2/3: exporting tiny model to ${PTE_PATH} ==="
NATIVE_SIMPLE_PTE_PATH="${PTE_PATH}" python "${EXPORT_PY}"

if [[ ! -f "${PTE_PATH}" ]]; then
    echo "ERROR: export did not produce ${PTE_PATH}" >&2
    exit 1
fi

# 3. Run the test_native_simple binary against the exported .pte.
echo
echo "=== Step 3/3: running test_native_simple on ${PTE_PATH} ==="
ET_TESTING_MODEL_PATH="${PTE_PATH}" "${TEST_BIN}"

echo
echo "=== run_simple_test.sh: PASS ==="
