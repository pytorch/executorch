#!/usr/bin/env bash
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.

set -e

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

cmake -DBUCK2="$BUCK" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE="$MODE" \
          -DEXECUTORCH_BUILD_DEVTOOLS=ON \
          -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
          -DEXECUTORCH_BUILD_MPS=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
cmake --build cmake-out -j9 --target install --config "$MODE"
CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
# build mps_executor_runner
rm -rf cmake-out/examples/apple/mps
cmake \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE="$MODE" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -Bcmake-out/examples/apple/mps \
    examples/apple/mps

cmake --build cmake-out/examples/apple/mps -j9 --config "$MODE"

echo "Build succeeded!"

./cmake-out/examples/apple/mps/mps_executor_runner --model_path mps_logical_not.pte --bundled_program
