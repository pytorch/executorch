#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds example_runner and prints its path.

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly SCRIPT_DIR

readonly EXECUTORCH_ROOT="${SCRIPT_DIR}/../.."

# Allow overriding the number of build jobs. Default to 9.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-9}"

BUILD_COREML=OFF
BUILD_XNNPACK=OFF

usage() {
  echo "Builds example runner."
  echo "Options:"
  echo "  --coreml             Include this flag to enable Core ML backend when building the Developer Tools."
  echo "  --xnnpack            Include this flag to enable XNNPACK backend when building the Developer Tools."
  exit 0
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --coreml) BUILD_COREML=ON ;;
      --xnnpack) BUILD_XNNPACK=ON ;;
      *)
  esac
done

main() {
  cd "${EXECUTORCH_ROOT}"

  ./install_executorch.sh --clean

  if [[ "${BUILD_COREML}" == "ON" ]] || [[ "${BUILD_XNNPACK}" == "ON" ]]; then
    cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
        -DEXECUTORCH_BUILD_COREML=$BUILD_COREML \
        -DEXECUTORCH_BUILD_XNNPACK=$BUILD_XNNPACK \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_EXAMPLES=OFF \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -Bcmake-out .
  else
   cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Debug \
      -DEXECUTORCH_BUILD_DEVTOOLS=ON \
      -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
      -Bcmake-out .
  fi

  cmake --build cmake-out --target install --config Release

  local example_dir=examples/devtools
  local build_dir="cmake-out/${example_dir}"

  # Check for both lib and lib64 directories
  local executorch_dir="${PWD}/cmake-out/lib/cmake/ExecuTorch"
  if [[ ! -d "${executorch_dir}" ]]; then
    executorch_dir="${PWD}/cmake-out/lib64/cmake/ExecuTorch"
  fi

  local cmake_prefix_path="${executorch_dir};${PWD}/cmake-out/third-party/gflags"

  rm -rf ${build_dir}
  cmake -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_COREML=$BUILD_COREML \
      -DEXECUTORCH_BUILD_XNNPACK=$BUILD_XNNPACK \
      -B"${build_dir}" \
      "${example_dir}"
  cmake --build "${build_dir}" --config Release

  local runner="${PWD}/${build_dir}/example_runner"
  if [[ ! -f "${runner}" ]]; then
    echo "ERROR: Failed to build ${build_dir}/example_runner" >&2
    exit 1
  else
    echo "Built ${build_dir}/example_runner"
  fi
}

main "$@"
