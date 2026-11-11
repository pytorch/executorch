#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the Cadence generic (reference) op-level gtest tests for the host and
# run them, then build and run cadence_op_registry_test, which checks that every
# op in backends/cadence/aot/functions.yaml is registered by cadence_ops_lib.
#
# Unlike backends/cadence/build_cadence_runner.sh (built -fno-exceptions
# -fno-rtti), the gtest tests need exceptions + RTTI, so those flags are NOT set
# here.

set -euo pipefail

BUILD_TARGETS=(cadence_generic_op_tests cadence_op_registry_test)
TEST_BINS=(
  cmake-out/backends/cadence/generic/operators/tests/cadence_generic_op_tests
  cmake-out/backends/cadence/cadence_op_registry_test
)

rm -rf cmake-out
cmake \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_CADENCE=ON \
  -DEXECUTORCH_BUILD_CADENCE_OP_TESTS=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DPYTHON_EXECUTABLE="$(which python3)" \
  -Bcmake-out .

cmake --build cmake-out --target "${BUILD_TARGETS[@]}" -j"$(nproc)"

for TEST_BIN in "${TEST_BINS[@]}"; do
  echo "=== running $(basename "${TEST_BIN}") ==="
  "${TEST_BIN}"
done
