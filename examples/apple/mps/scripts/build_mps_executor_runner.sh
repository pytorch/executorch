#!/usr/bin/env bash
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


set -eux

MODE="Release"
OUTPUT="cmake-out"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Build frameworks for Apple platforms."
  echo "SOURCE_ROOT_DIR defaults to the current directory if not provided."
  echo
  echo "Options:"
  echo "  --output=DIR         Output directory. Default: 'cmake-out'"
  echo "  --Debug              Use Debug build mode. Default: 'Release'"
  echo "Example:"
  echo "  $0 --output=cmake-out --Debug"
  exit 0
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --output=*) OUTPUT="${arg#*=}" ;;
      --Debug) MODE="Debug" ;;
      *)
      if [[ -z "$SOURCE_ROOT_DIR" ]]; then
          SOURCE_ROOT_DIR="$arg"
      else
          echo "Invalid argument: $arg"
          exit 1
      fi
      ;;
  esac
done

rm -rf "$OUTPUT"

cmake -DCMAKE_INSTALL_PREFIX=${OUTPUT} \
      -DCMAKE_BUILD_TYPE=${MODE} \
      -DEXECUTORCH_ENABLE_LOGGING=ON \
      -B ${OUTPUT} \
      --preset macos

cmake --build ${OUTPUT} \
      -j $(sysctl -n hw.ncpu) \
      --config ${MODE} \
      --target install

cmake -DCMAKE_PREFIX_PATH="${OUTPUT}/lib/cmake/ExecuTorch;${OUTPUT}/third-party/gflags" \
      -DCMAKE_BUILD_TYPE="$MODE" \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="12.0" \
      -B "${OUTPUT}/examples/apple/mps" \
      -S examples/apple/mps

cmake --build "${OUTPUT}/examples/apple/mps" \
      -j $(sysctl -n hw.ncpu) \
      --config ${MODE} \
      --target mps_executor_runner

echo "Build succeeded!"
