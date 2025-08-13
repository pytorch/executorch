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

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

TARGET_LIBS=""

if [[ "${MODE}" =~ .*openvino.* ]]; then
  OPENVINO=ON
  TARGET_LIBS="$TARGET_LIBS openvino_backend "

  git clone https://github.com/openvinotoolkit/openvino.git
  cd openvino && git b16b776ac119dafda51f69a80f1e6b7376d02c3b
  git submodule update --init --recursive
  sudo ./install_build_dependencies.sh
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
  make -j$(nproc)

  cd ..
  cmake --install build --prefix dist

  source dist/setupvars.sh
  cd ../backends/openvino
  pip install -r requirements.txt
  cd ../../
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


DIR="examples/models/yolo12"
$PYTHON_EXECUTABLE -m pip install -r ${DIR}/requirements.txt

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
                       -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
                       -DEXECUTORCH_ENABLE_LOGGING=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
                       -DEXECUTORCH_BUILD_PYBIND=ON"

    echo $TARGET_LIBS
    export CMAKE_BUILD_ARGS="--target $TARGET_LIBS"
    pip install . --no-build-isolation
}

cmake_build_demo() {
    echo "Building yolo12 runner"
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
EXPORTED_MODEL_NAME="${MODEL_NAME}_fp32_${MODE}.pte"
echo "Exporting ${EXPORTED_MODEL_NAME}"
EXPORT_ARGS="--model_name=${MODEL_NAME} --backend=${MODE}"

# Add dynamically linked library location
cmake_install_executorch_libraries

$PYTHON_EXECUTABLE -m examples.models.yolo12.export_and_validate ${EXPORT_ARGS}


RUNTIME_ARGS="--model_path=${EXPORTED_MODEL_NAME} --input_path=${VIDEO_PATH}"
# Check build tool.
cmake_build_demo
# Run yolo12 runner
NOW=$(date +"%H:%M:%S")
echo "Starting to run yolo12 runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/yolo12/Yolo12DetectionDemo ${RUNTIME_ARGS} > result.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result.txt)

prepare_artifacts_upload
cleanup_files
