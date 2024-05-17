#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

build_executorch() {
  cmake . -DCMAKE_INSTALL_PREFIX=cmake-out -DEXECUTORCH_BUILD_GTESTS=ON -Bcmake-out
  cmake --build cmake-out -j9 --target install
}

build_and_run_test() {
  local test_dir=$1
  cmake "${test_dir}" -Bcmake-out/"${test_dir}" -DCMAKE_INSTALL_PREFIX=cmake-out
  cmake --build cmake-out/"${test_dir}" -j9
  for t in $(cmake-out/"${test_dir}"/*test); do ./"$t"; done
}

build_executorch
build_and_run_test extension/data_loader/test/
build_and_run_test runtime/core/portable_type/test/
build_and_run_test runtime/core/test/
build_and_run_test runtime/core/exec_aten/util/test/
build_and_run_test runtime/core/exec_aten/testing_util/test/
