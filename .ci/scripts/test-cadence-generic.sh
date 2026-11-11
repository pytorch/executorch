#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the Cadence generic (reference) op-level gtest tests for the host and
# run them.
#
# Unlike backends/cadence/build_cadence_runner.sh (built -fno-exceptions
# -fno-rtti), the gtest tests need exceptions + RTTI, so those flags are NOT set
# here.

set -euo pipefail

TEST_TARGET=cadence_generic_op_tests
TEST_BIN="cmake-out/backends/cadence/generic/operators/tests/${TEST_TARGET}"

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

cmake --build cmake-out --target "${TEST_TARGET}" -j"$(nproc)"

"${TEST_BIN}"
