#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds cadence_runner and prints its path.

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly SCRIPT_DIR

readonly EXECUTORCH_ROOT="${SCRIPT_DIR}/../.."

# Allow overriding the number of build jobs. Default to 9.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-9}"

main() {
  cd "${EXECUTORCH_ROOT}"

  rm -rf cmake-out
  cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -Bcmake-out .
  cmake --build cmake-out --target install --config Release -j16

  local example_dir=backends/cadence
  local build_dir="cmake-out/${example_dir}"
  local cmake_prefix_path="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
  rm -rf ${build_dir}
  cmake -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_CADENCE_CPU_RUNNER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -B"${build_dir}" \
    "${example_dir}"
  cmake --build "${build_dir}" --config Release -j16

  local runner="${PWD}/${build_dir}/cadence_runner"
  if [[ ! -f "${runner}" ]]; then
    echo "ERROR: Failed to build ${build_dir}/cadence_runner" >&2
    exit 1
  else
    echo "Built ${build_dir}/cadence_runner"
  fi
}

main "$@"
