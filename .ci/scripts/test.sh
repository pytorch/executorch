#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME=$1
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

BUILD_TOOL=$2
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
fi

QUANTIZATION=$3
if [[ -z "${QUANTIZATION:-}" ]]; then
  QUANTIZATION=false
fi

XNNPACK_DELEGATION=$4
if [[ -z "${XNNPACK_DELEGATION:-}" ]]; then
  XNNPACK_DELEGATION=false
fi

which "${PYTHON_EXECUTABLE}"
# Just set this variable here, it's cheap even if we use buck2
CMAKE_OUTPUT_DIR=cmake-out

build_cmake_executor_runner() {
  echo "Building executor_runner"
  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && mkdir ${CMAKE_OUTPUT_DIR} \
    && cd ${CMAKE_OUTPUT_DIR} \
    && retry cmake -DBUCK2=buck2 \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build ${CMAKE_OUTPUT_DIR} -j4
}

test_model() {
  "${PYTHON_EXECUTABLE}" -m examples.export.export_example --model_name="${MODEL_NAME}"

  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/executor_runner:executor_runner -- --model_path "./${MODEL_NAME}.pte"
  elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
    if [[ ! -f ${CMAKE_OUTPUT_DIR}/executor_runner ]]; then
      build_cmake_executor_runner
    fi
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./${MODEL_NAME}.pte"
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

build_cmake_xnn_executor_runner() {
  echo "Building xnn_executor_runner"
  SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && mkdir ${CMAKE_OUTPUT_DIR} \
    && cd ${CMAKE_OUTPUT_DIR} \
    && retry cmake -DBUCK2=buck2 \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DREGISTER_QUANTIZED_OPS=ON \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build ${CMAKE_OUTPUT_DIR} -j4
}

test_model_with_xnnpack() {
  WITH_QUANTIZATION=$1
  if [[ ${WITH_QUANTIZATION} == true ]]; then
    SUFFIX="q8"
    "${PYTHON_EXECUTABLE}" -m examples.backend.xnnpack_examples --model_name="${MODEL_NAME}" --delegate --quantize
  else
    SUFFIX="fp32"
    "${PYTHON_EXECUTABLE}" -m examples.backend.xnnpack_examples --model_name="${MODEL_NAME}" --delegate
  fi

  OUTPUT_MODEL_PATH="${MODEL_NAME}_xnnpack_${SUFFIX}.pte"
  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/backend:xnn_executor_runner -- --model_path "${OUTPUT_MODEL_PATH}"
  elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
    if [[ ! -f ${CMAKE_OUTPUT_DIR}/backends/xnnpack/xnn_executor_runner ]]; then
      build_cmake_xnn_executor_runner
    fi
    ./${CMAKE_OUTPUT_DIR}/backends/xnnpack/xnn_executor_runner --model_path "${OUTPUT_MODEL_PATH}"
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

echo "Testing ${MODEL_NAME} (fp32, quantized, xnnpack) with ${BUILD_TOOL}..."
# Test the select model without XNNPACK or quantization
test_model

# Test quantization
if [[ "${QUANTIZATION}" == true ]]; then
  bash examples/quantization/test_quantize.sh "${BUILD_TOOL}" "${MODEL_NAME}"
else
  echo "The model ${MODEL_NAME} doesn't support quantization yet"
fi

# Test XNNPACK without quantization
if [[ "${XNNPACK_DELEGATION}" == true ]]; then
  test_model_with_xnnpack false
else
  echo "The model ${MODEL_NAME} doesn't support XNNPACK yet"
fi

# Test XNNPACK with quantization
if [[ "${XNNPACK_DELEGATION}" == true ]] && [[ "${QUANTIZATION}" == true ]]; then
  test_model_with_xnnpack true
fi
