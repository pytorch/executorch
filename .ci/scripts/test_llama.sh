#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME=$1 # stories110M
BUILD_TOOL=$2 # buck2 or cmake
DTYPE=$3 # fp16 or fp32
MODE=${4:-"xnnpack+custom"} # portable or xnnpack+custom or xnnpack+custom+qe
UPLOAD_DIR=${5:-}
if [[ $# -lt 4 ]]; then # Assuming 4 mandatory args
    echo "Expecting atleast 4 positional arguments"
    echo "Usage: [...]"
fi
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit
fi

if [[ -z "${DTYPE:-}" ]]; then
  echo "Missing dtype, choose fp16 or fp32, exiting..."
  exit 1
fi

if [[ -z "${MODE:-}" ]]; then
  echo "Missing mode, choose portable or xnnpack, exiting..."
  exit 1
fi

if [[ "${MODE}" =~ .*xnnpack.* ]]; then
  XNNPACK=ON
else
  XNNPACK=OFF
fi

if [[ "${MODE}" =~ .*custom.* ]]; then
  CUSTOM=ON
else
  CUSTOM=OFF
fi

if [[ "${MODE}" =~ .*qe.* ]]; then
  QE=ON
else
  QE=OFF
fi

if [[ "${MODE}" =~ .*mps.* ]]; then
  MPS=ON
else
  MPS=OFF
fi

echo "MPS option ${MPS}"

if [[ "${MODE}" =~ .*coreml.* ]]; then
  COREML=ON
else
  COREML=OFF
fi

echo "COREML option ${COREML}"

if [[ "${MODE}" =~ .*qnn.* ]]; then
  QNN=ON
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  export QNN_SDK_ROOT=/tmp/qnn/2.25.0.240728
  export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang"
  export PYTHONPATH=".."
  cp schema/program.fbs exir/_serialize/program.fbs
  cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
  cp -f build-x86/backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so backends/qualcomm/python
  cp -f build-x86/backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so backends/qualcomm/python

else
  QNN=OFF
  QNN_SDK_ROOT=""
fi

echo "QNN option ${QNN}"
echo "QNN_SDK_ROOT: ${QNN_SDK_ROOT}"

if [[ -z "${BUCK:-}" ]]; then
  BUCK=buck2
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    retry cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Debug \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM="$CUSTOM" \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK="$XNNPACK" \
        -DEXECUTORCH_BUILD_MPS="$MPS" \
        -DEXECUTORCH_BUILD_COREML="$COREML" \
        -DEXECUTORCH_BUILD_QNN="$QNN" \
        -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out .
    cmake --build cmake-out -j9 --target install --config Debug
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    dir="examples/models/llama2"
    retry cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Debug \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM="$CUSTOM" \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK="$XNNPACK" \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j9 --config Debug

}

cleanup_files() {
  echo "Deleting downloaded and generated files"
  rm "${CHECKPOINT_FILE_NAME}"
  rm tokenizer.model
  rm tokenizer.bin
  rm "${EXPORTED_MODEL_NAME}"
  rm result.txt
  rm params.json
}

prepare_artifacts_upload() {
  if [ -n "$UPLOAD_DIR" ]; then
    echo "Preparing for uploading generated artifacs"
    zip -j model.zip "${EXPORTED_MODEL_NAME}" tokenizer.bin
    mkdir -p "${UPLOAD_DIR}"
    mv model.zip "${UPLOAD_DIR}"
  fi
}

# Download and create artifacts.
PARAMS="params.json"
CHECKPOINT_FILE_NAME=""
touch "${PARAMS}"
if [[ "${MODEL_NAME}" == "stories110M" ]]; then
  CHECKPOINT_FILE_NAME="stories110M.pt"
  download_stories_model_artifacts
else
  echo "Unsupported model name ${MODEL_NAME}"
  exit 1
fi

# Check dtype.
EXPORTED_MODEL_NAME="tinyllama_${MODE}_${DTYPE}"
if [[ "${DTYPE}" == "fp16" ]]; then
  EXPORTED_MODEL_NAME="${EXPORTED_MODEL_NAME}_h"
elif [[ "${DTYPE}" == "fp32" ]]; then
  :
else
  echo "Unsupported dtype ${DTYPE}"
  exit 1
fi

# Export model.
EXPORTED_MODEL_NAME="${EXPORTED_MODEL_NAME}.pte"
echo "Exporting ${EXPORTED_MODEL_NAME}"
EXPORT_ARGS="-c ${CHECKPOINT_FILE_NAME} -p ${PARAMS} -d ${DTYPE} -n ${EXPORTED_MODEL_NAME} -kv"
if [[ "${XNNPACK}" == "ON" ]]; then
  EXPORT_ARGS="${EXPORT_ARGS} -X -qmode 8da4w -G 128"
fi
if [[ "${CUSTOM}" == "ON" ]]; then
  EXPORT_ARGS="${EXPORT_ARGS} --use_sdpa_with_kv_cache"
fi
if [[ "${QE}" == "ON" ]]; then
  EXPORT_ARGS="${EXPORT_ARGS} --embedding-quantize 8,1024"
fi
if [[ "${MPS}" == "ON" ]]; then
  EXPORT_ARGS="${EXPORT_ARGS} -kv -v --mps --disable_dynamic_shape"
fi
if [[ "${COREML}" == "ON" ]]; then
  EXPORT_ARGS="${EXPORT_ARGS} -kv -v --coreml --disable_dynamic_shape"
fi
if [[ "${QNN}" == "ON" ]]; then
  EXPORT_ARGS="${EXPORT_ARGS} -kv -v --qnn --disable_dynamic_shape"
fi
# Add dynamically linked library location
$PYTHON_EXECUTABLE -m examples.models.llama2.export_llama ${EXPORT_ARGS}

# Create tokenizer.bin.
echo "Creating tokenizer.bin"
$PYTHON_EXECUTABLE -m extension.llm.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin


RUNTIME_ARGS="--model_path=${EXPORTED_MODEL_NAME} --tokenizer_path=tokenizer.bin --prompt=Once --temperature=0 --seq_len=10"
# Check build tool.
echo "Running ${EXPORTED_MODEL_NAME} in portable mode"
if [[ "${BUILD_TOOL}" == "buck2" ]]; then
  # Run model.
  # shellcheck source=/dev/null
  $BUCK run examples/models/llama2:main -- ${RUNTIME_ARGS} > result.txt
elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
  cmake_install_executorch_libraries
  cmake_build_llama_runner
  # Run llama runner
  NOW=$(date +"%H:%M:%S")
  echo "Starting to run llama runner at ${NOW}"
  # shellcheck source=/dev/null
  cmake-out/examples/models/llama2/llama_main ${RUNTIME_ARGS} > result.txt
  NOW=$(date +"%H:%M:%S")
  echo "Finished at ${NOW}"
else
  echo "Invalid build tool ${BUILD_TOOL}. Only buck2 is supported atm"
  exit 1
fi
RESULT=$(cat result.txt)
# Check results.
EXPECTED_PREFIX="Once upon a time,"
# Expected result - may take too long to generate:
# "Once upon a time, there was a little girl named Lily. She loved to play outside" ...
if [[ "${RESULT}" == "${EXPECTED_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Success"

  prepare_artifacts_upload
  cleanup_files
else
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Failure; results not the same"

  cleanup_files
  exit 1
fi
