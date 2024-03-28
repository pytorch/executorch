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

BACKEND=$3
if [[ -z "${BACKEND:-}" ]]; then
  echo "Missing backend (require portable or xnnpack), exiting..."
  exit 1
fi

which "${PYTHON_EXECUTABLE}"
# Just set this variable here, it's cheap even if we use buck2
CMAKE_OUTPUT_DIR=cmake-out

build_cmake_executor_runner() {
  echo "Building executor_runner"
  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && mkdir ${CMAKE_OUTPUT_DIR} \
    && cd ${CMAKE_OUTPUT_DIR} \
    && retry cmake -DBUCK2=buck2 -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build ${CMAKE_OUTPUT_DIR} -j4
}

run_portable_executor_runner() {
  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/portable/executor_runner:executor_runner -- --model_path "./${MODEL_NAME}.pte"
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

test_model() {
  if [[ "${MODEL_NAME}" == "llama2" ]]; then
    # Install requirements for export_llama
    bash examples/models/llama2/install_requirements.sh
    # Test export_llama script: python3 -m examples.models.llama2.export_llama
    "${PYTHON_EXECUTABLE}" -m examples.models.llama2.export_llama -c examples/models/llama2/params/demo_rand_params.pth -p examples/models/llama2/params/demo_config.json
    run_portable_executor_runner
    rm "./${MODEL_NAME}.pte"
  fi
  if [[ "${MODEL_NAME}" == "llava_encoder" ]]; then
    # Install requirements for llava
    bash examples/models/llava_encoder/install_requirements.sh
  fi
  # python3 -m examples.portable.scripts.export --model_name="llama2" should works too
  "${PYTHON_EXECUTABLE}" -m examples.portable.scripts.export --model_name="${MODEL_NAME}"
  run_portable_executor_runner
}

build_cmake_xnn_executor_runner() {
  echo "Building xnn_executor_runner"
  SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && mkdir ${CMAKE_OUTPUT_DIR} \
    && cd ${CMAKE_OUTPUT_DIR} \
    && retry cmake -DBUCK2=buck2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build ${CMAKE_OUTPUT_DIR} -j4
}

test_model_with_xnnpack() {
  WITH_QUANTIZATION=$1
  WITH_DELEGATION=$2

  # Quantization-only
  if [[ ${WITH_QUANTIZATION} == true ]] && [[ ${WITH_DELEGATION} == false ]]; then
    bash examples/xnnpack/quantization/test_quantize.sh "${BUILD_TOOL}" "${MODEL_NAME}"
    return 0
  fi

  # Delegation
  if [[ ${WITH_QUANTIZATION} == true ]]; then
    SUFFIX="q8"
    "${PYTHON_EXECUTABLE}" -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate --quantize
  else
    SUFFIX="fp32"
    "${PYTHON_EXECUTABLE}" -m examples.xnnpack.aot_compiler --model_name="${MODEL_NAME}" --delegate
  fi

  OUTPUT_MODEL_PATH="${MODEL_NAME}_xnnpack_${SUFFIX}.pte"

  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/xnnpack:xnn_executor_runner -- --model_path "${OUTPUT_MODEL_PATH}"
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

if [[ "${BACKEND}" == "portable" ]]; then
  echo "Testing ${MODEL_NAME} with portable kernels..."
  test_model
else
  set +e
  if [[ "${BACKEND}" == *"quantization"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK quantization only..."
    test_model_with_xnnpack true false || Q_ERROR="error"
    echo "::endgroup::"
  fi
  if [[ "${BACKEND}" == *"delegation"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK delegation only..."
    test_model_with_xnnpack false true || D_ERROR="error"
    echo "::endgroup::"
  fi
  if [[ "${BACKEND}" == *"quantization"* ]] && [[ "${BACKEND}" == *"delegation"* ]]; then
    echo "::group::Testing ${MODEL_NAME} with XNNPACK quantization and delegation..."
    test_model_with_xnnpack true true || Q_D_ERROR="error"
    echo "::endgroup::"
  fi
  set -e
  if [[ -n "${Q_ERROR:-}" ]] || [[ -n "${D_ERROR:-}" ]] || [[ -n "${Q_D_ERROR:-}" ]]; then
    echo "Portable q8 ${Q_ERROR:-ok}," "Delegation fp32 ${D_ERROR:-ok}," "Delegation q8 ${Q_D_ERROR:-ok}"
    exit 1
  fi
fi
