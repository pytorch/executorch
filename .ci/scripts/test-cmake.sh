#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

build_and_test_executorch() {
  CMAKE_OUTPUT_DIR=cmake-out
  # Build executorch runtime using cmake
  rm -rf "${CMAKE_OUTPUT_DIR}" && mkdir "${CMAKE_OUTPUT_DIR}"

  pushd "${CMAKE_OUTPUT_DIR}"
  cmake -DBUCK2=buck2 ..
  popd

  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build "${CMAKE_OUTPUT_DIR}" -j "${CMAKE_JOBS}"

  which python
  # Export a test model
  python -m examples.export.export_example --model_name="linear"

  # Run test model
  ./"${CMAKE_OUTPUT_DIR}"/executor_runner --model_path ./linear.pte
}

install_executorch
build_and_test_executorch
