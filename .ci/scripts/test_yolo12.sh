#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -model)
      MODEL_NAME="$2" # stories110M
      shift 2
      ;;
    -mode)
      MODE="$2" # portable or xnnpack+custom or xnnpack+custom+qe
      shift 2
      ;;
    -pt2e_quantize)
      PT2E_QUANTIZE="$2"
      shift 2
      ;;
    -upload)
      UPLOAD_DIR="$2"
      shift 2
      ;;
    -video_path)
      VIDEO_PATH="$2" # portable or xnnpack+custom or xnnpack+custom+qe
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Default mode to xnnpack+custom if not set
MODE=${MODE:-"openvino"}

# Default UPLOAD_DIR to empty string if not set
UPLOAD_DIR="${UPLOAD_DIR:-}"

# Default PT2E_QUANTIZE to empty string if not set
PT2E_QUANTIZE="${PT2E_QUANTIZE:-}"

# Default CMake Build Type to release mode
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

if [[ $# -lt 5 ]]; then # Assuming 4 mandatory args
    echo "Expecting atleast 5 positional arguments"
    echo "Usage: [...]"
fi
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi


if [[ -z "${MODE:-}" ]]; then
  echo "Missing mode, choose openvino or xnnpack, exiting..."
  exit 1
fi

TARGET_LIBS=""

if [[ "${MODE}" =~ .*openvino.* ]]; then
  OPENVINO=ON
  TARGET_LIBS="$TARGET_LIBS openvino_backend "
else
  OPENVINO=OFF
fi

if [[ "${MODE}" =~ .*xnnpack.* ]]; then
  XNNPACK=ON
  TARGET_LIBS="$TARGET_LIBS xnnpack_backend "
else
  XNNPACK=OFF
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    mkdir cmake-out

    retry cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DEXECUTORCH_BUILD_OPENVINO="$OPENVINO" \
        -DEXECUTORCH_BUILD_XNNPACK="$XNNPACK" \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_PYBIND=ON \
        -Bcmake-out
    cmake --build cmake-out -j9 --target install --config "$CMAKE_BUILD_TYPE"
}

cmake_build_demo() {
    echo "Building yolo12 runner"
    dir="examples/models/yolo12"
    retry cmake \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DUSE_OPENVINO_BACKEND="$OPENVINO" \
        -DUSE_OPENVINO_BACKEND="$XNNPACK" \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j9 --config "$CMAKE_BUILD_TYPE"

}

cleanup_files() {
 true
}

prepare_artifacts_upload() {
  if [ -n "${UPLOAD_DIR}" ]; then
    true
  fi
}


# Export model.
EXPORTED_MODEL_NAME="${MODEL_NAME}_fp32_${MODE}.pte"
echo "Exporting ${EXPORTED_MODEL_NAME}"
EXPORT_ARGS="--model_name=${MODEL_NAME} --backend=${MODE}"

# Add dynamically linked library location
cmake_install_executorch_libraries

$PYTHON_EXECUTABLE -m examples.models.yolo12.export_and_validate ${EXPORT_ARGS}


RUNTIME_ARGS="--model_path=${EXPORTED_MODEL_NAME} --input_path=${INPUT_VIDEO}"
# Check build tool.
cmake_build_demo
# Run llama runner
NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/yolo12/Yolo12Detection ${RUNTIME_ARGS} > result.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result.txt)

prepare_artifacts_upload
cleanup_files
