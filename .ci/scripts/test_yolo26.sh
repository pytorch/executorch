#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

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

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

if [[ -z "${MODE:-}" ]]; then
  echo "Missing mode, choose openvino or xnnpack, exiting..."
  exit 1
fi

if [[ -z "${VIDEO_PATH:-}" ]]; then
  echo "Missing video path, exiting..."
  exit 1
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

TARGET_LIBS=""

if [[ "${MODE}" =~ .*openvino.* ]]; then
  OPENVINO=ON
  TARGET_LIBS="$TARGET_LIBS openvino_backend "

  # Install specific OpenVINO runtime from pip.
  $PYTHON_EXECUTABLE -m pip install --pre openvino==2026.1.0.dev20260131 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
  $PYTHON_EXECUTABLE -m pip install -r backends/openvino/requirements.txt

  # Set OPENVINO_LIB_PATH so the C++ demo runner can also find libopenvino_c.so.
  OPENVINO_LIB_PATH=$($PYTHON_EXECUTABLE -c "import openvino, os, glob; print(sorted(glob.glob(os.path.join(os.path.dirname(openvino.__file__), 'libs', 'libopenvino_c.so*')))[-1])")
  export OPENVINO_LIB_PATH
else
  OPENVINO=OFF
fi

if [[ "${MODE}" =~ .*xnnpack.* ]]; then
  XNNPACK=ON
  TARGET_LIBS="$TARGET_LIBS xnnpack_backend "
else
  XNNPACK=OFF
fi

which "${PYTHON_EXECUTABLE}"

TORCH_URL=https://download.pytorch.org/whl/cpu

DIR="examples/models/yolo26"
$PYTHON_EXECUTABLE -m pip install --upgrade-strategy only-if-needed --extra-index-url "$TORCH_URL" -r ${DIR}/requirements.txt

cmake_install_executorch_libraries() {
    rm -rf cmake-out
    build_dir=cmake-out
    mkdir $build_dir


    retry cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
          -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
          -DEXECUTORCH_BUILD_OPENVINO="$OPENVINO" \
          -DEXECUTORCH_BUILD_XNNPACK="$XNNPACK" \
          -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
          -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
          -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
          -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
          -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
          -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
          -B"${build_dir}"

    # Build the project
    cmake --build ${build_dir} --target install --config ${CMAKE_BUILD_TYPE} -j$(nproc)

    export CMAKE_ARGS="
                       -DEXECUTORCH_BUILD_OPENVINO="$OPENVINO" \
                       -DEXECUTORCH_BUILD_XNNPACK="$XNNPACK" \
                       -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
                       -DEXECUTORCH_ENABLE_LOGGING=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
                       -DEXECUTORCH_BUILD_PYBIND=ON"

    echo $TARGET_LIBS
    export CMAKE_BUILD_ARGS="--target $TARGET_LIBS"
    $PYTHON_EXECUTABLE -m pip install . --no-build-isolation
}

cmake_build_demo() {
    echo "Building yolo26 runner"
    retry cmake \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DUSE_OPENVINO_BACKEND="$OPENVINO" \
        -DUSE_XNNPACK_BACKEND="$XNNPACK" \
        -Bcmake-out/${DIR} \
        ${DIR}
    cmake --build cmake-out/${DIR} -j9 --config "$CMAKE_BUILD_TYPE"

}

cleanup_files() {
 rm $EXPORTED_MODEL_NAME
}

prepare_artifacts_upload() {
  if [ -n "${UPLOAD_DIR}" ]; then
    echo "Preparing for uploading generated artifacs"
    zip -j model.zip "${EXPORTED_MODEL_NAME}"
    mkdir -p "${UPLOAD_DIR}"
    mv model.zip "${UPLOAD_DIR}"
    mv result.txt "${UPLOAD_DIR}"

  fi
}


# Export model.
EXPORT_ARGS="--model_name=${MODEL_NAME} --backend=${MODE}"
if [[ -n "${PT2E_QUANTIZE}" ]]; then
  EXPORTED_MODEL_NAME="${MODEL_NAME}_int8_${MODE}.pte"
  EXPORT_ARGS="${EXPORT_ARGS} --quantize --video_path=${VIDEO_PATH}"
else
  EXPORTED_MODEL_NAME="${MODEL_NAME}_fp32_${MODE}.pte"
fi
echo "Exporting ${EXPORTED_MODEL_NAME}"

# Add dynamically linked library location
cmake_install_executorch_libraries

$PYTHON_EXECUTABLE -m examples.models.yolo26.export_and_validate ${EXPORT_ARGS}


RUNTIME_ARGS="--model_path=${EXPORTED_MODEL_NAME} --input_path=${VIDEO_PATH}"
# Check build tool.
cmake_build_demo
# Run yolo26 runner
NOW=$(date +"%H:%M:%S")
echo "Starting to run yolo26 runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/yolo26/Yolo26DetectionDemo ${RUNTIME_ARGS} > result.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result.txt)

prepare_artifacts_upload
cleanup_files
