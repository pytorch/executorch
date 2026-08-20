#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the Cadence Xtensa op-level gtest tests for the configured backend and
# run them on the Instruction Set Simulator (xt-run).
#
# Requires the Xtensa toolchain env to already be set (run
# .ci/scripts/setup-xtensa-tools.sh <backend> first): XTENSA_TOOLCHAIN,
# TOOLCHAIN_VER, XTENSA_CORE, CADENCE_OPT_FLAG, and xt-clang/xt-run on PATH.
#
# Unlike build-cadence-xtensa.sh (the runner, built -fno-exceptions -fno-rtti),
# the gtest tests need exceptions + RTTI, so those flags are NOT set here.

set -euo pipefail

: "${XTENSA_TOOLCHAIN:?run setup-xtensa-tools.sh first}"
: "${TOOLCHAIN_VER:?run setup-xtensa-tools.sh first}"
: "${XTENSA_CORE:?run setup-xtensa-tools.sh first}"
: "${CADENCE_OPT_FLAG:?run setup-xtensa-tools.sh first}"

# Map the optimized-kernel flag to the backend dir + gtest target name.
case "${CADENCE_OPT_FLAG}" in
  EXECUTORCH_NNLIB_OPT) TARGET_DIR=hifi ;;
  EXECUTORCH_VISION_OPT) TARGET_DIR=vision ;;
  EXECUTORCH_FUSION_G3_OPT) TARGET_DIR=fusion_g3 ;;
  *)
    echo "ERROR: unknown CADENCE_OPT_FLAG='${CADENCE_OPT_FLAG}'" >&2
    exit 1
    ;;
esac
TEST_TARGET="cadence_${TARGET_DIR}_op_tests"
TEST_ELF="cmake-out/backends/cadence/${TARGET_DIR}/operators/tests/${TEST_TARGET}"

NPROC=$(nproc)
echo "=== building ${TEST_TARGET} for ${XTENSA_CORE} (${CADENCE_OPT_FLAG}) ==="
xt-clang --version | head -1

rm -rf cmake-out
cmake \
  -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_CADENCE=ON \
  "-D${CADENCE_OPT_FLAG}=ON" \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_CADENCE_OP_TESTS=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DEXECUTORCH_USE_DL=OFF \
  -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
  -DEXECUTORCH_BUILD_DEVTOOLS=OFF \
  -DHAVE_FNMATCH_H=OFF \
  -DFLATCC_ALLOW_WERROR=OFF \
  -DPYTHON_EXECUTABLE="$(which python3)" \
  -Bcmake-out .

cmake --build cmake-out --target "${TEST_TARGET}" -j"${NPROC}"

if [[ ! -f "${TEST_ELF}" ]]; then
  echo "ERROR: ${TEST_ELF} was not produced" >&2
  exit 1
fi

echo "=== running ${TEST_TARGET} on xt-run ==="
LOG=$(mktemp)
# --exit_with_target_code propagates gtest_main's exit code, so a failing test
# fails this step; also assert on the gtest summary lines as a backstop.
xt-run --turbo --exit_with_target_code "${TEST_ELF}" 2>&1 | tee "${LOG}"
if grep -q "\[  FAILED  \]" "${LOG}"; then
  echo "ERROR: gtest reported failures for ${TEST_TARGET}" >&2
  exit 1
fi
if ! grep -q "\[  PASSED  \]" "${LOG}"; then
  echo "ERROR: ${TEST_TARGET} did not report a gtest PASSED summary" >&2
  exit 1
fi
echo "Cadence ${TARGET_DIR} op tests passed on ${XTENSA_CORE}."
