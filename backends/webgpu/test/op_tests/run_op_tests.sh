#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generate WebGPU op-test artifacts, build the generic driver, and run it.
# Op-count-independent: the same 3 steps regardless of how many cases cases.py declares.
set -eux

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)" # executorch repo root
OUT="${1:-/tmp/webgpu_op_tests}"
BUILD="${BUILD_DIR:-$ROOT/cmake-out-webgpu-optest}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
NPROC="$(nproc)"

# 1. Generate .pte + serialized inputs + torch goldens + manifest.json.
"$PYTHON_EXECUTABLE" -m executorch.backends.webgpu.test.op_tests.generate_op_tests \
  --output "$OUT"

# 2. Configure with the WebGPU test build AND EXECUTORCH_BUILD_TESTS=ON (the latter
#    pulls in third-party/googletest so GTest::gtest is defined).
cmake \
  -DEXECUTORCH_BUILD_WEBGPU=ON \
  -DEXECUTORCH_BUILD_WEBGPU_TEST=ON \
  -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -B "$BUILD" \
  "$ROOT"

# 3. Build + run: the device-free util test, then the manifest-driven driver.
cmake --build "$BUILD" --target webgpu_op_test_util_test webgpu_op_test -j"$NPROC"
"$BUILD/backends/webgpu/webgpu_op_test_util_test"
"$BUILD/backends/webgpu/webgpu_op_test" --manifest "$OUT/manifest.json"
