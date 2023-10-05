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

XNNPACK_QUANTIZATION=$3
if [[ -z "${XNNPACK_QUANTIZATION:-}" ]]; then
  XNNPACK_QUANTIZATION=false
fi

XNNPACK_DELEGATION=$4
if [[ -z "${XNNPACK_DELEGATION:-}" ]]; then
  XNNPACK_DELEGATION=false
fi

DEMO_BACKEND_DELEGATION=$5
if [[ -z "${DEMO_BACKEND_DELEGATION:-}" ]]; then
  DEMO_BACKEND_DELEGATION=false
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
  if [[ "${MODEL_NAME}" == "llama2" ]]; then
    cd examples/third-party/llama
    pip install -e .
    cd ../../..
  fi

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
  WITH_DELEGATION=$2

  # Quantization-only
  if [[ ${WITH_QUANTIZATION} == true ]] && [[ ${WITH_DELEGATION} == false ]]; then
    bash examples/quantization/test_quantize.sh "${BUILD_TOOL}" "${MODEL_NAME}"
    exit 0
  fi

  # Delegation
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

test_demo_backend_delegation() {
  echo "Testing demo backend delegation on AddMul"
  "${PYTHON_EXECUTABLE}" -m examples.export.export_and_delegate  --option "composite"
  "${PYTHON_EXECUTABLE}" -m examples.export.export_and_delegate  --option "partition"
  "${PYTHON_EXECUTABLE}" -m examples.export.export_and_delegate  --option "whole"

  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/executor_runner:executor_runner -- --model_path "./composite_model.pte"
    buck2 run //examples/executor_runner:executor_runner -- --model_path "./partition_lowered_model.pte"
    buck2 run //examples/executor_runner:executor_runner -- --model_path "./whole.pte"
  elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
    if [[ ! -f ${CMAKE_OUTPUT_DIR}/executor_runner ]]; then
      build_cmake_executor_runner
    fi
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./composite_model.pte"
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./partition_lowered_model.pte"
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./whole.pte"
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

if [[ "${XNNPACK_DELEGATION}" == false ]] && [[ "${XNNPACK_QUANTIZATION}" == false ]]; then
  echo "Testing ${MODEL_NAME} with portable kernels..."
  test_model
else
  echo "Testing ${MODEL_NAME} with XNNPACK quantization=${XNNPACK_QUANTIZATION} delegation=${XNNPACK_DELEGATION}..."
  test_model_with_xnnpack "${XNNPACK_QUANTIZATION}" "${XNNPACK_DELEGATION}"
fi

# Test demo backend delegation
if [[ "${DEMO_BACKEND_DELEGATION}" == true ]]; then
  test_demo_backend_delegation
fi
