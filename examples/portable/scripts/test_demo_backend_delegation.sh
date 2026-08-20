#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

which "${PYTHON_EXECUTABLE}"

BUILD_TOOL=$1
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
fi

CMAKE_OUTPUT_DIR=cmake-out

build_cmake_executor_runner() {
  echo "Building executor_runner"
  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && mkdir ${CMAKE_OUTPUT_DIR} \
    && cd ${CMAKE_OUTPUT_DIR} \
    && retry cmake -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build ${CMAKE_OUTPUT_DIR} -j4
}

test_demo_backend_delegation() {
  echo "Testing demo backend delegation on AddMul"
  "${PYTHON_EXECUTABLE}" -m examples.portable.scripts.export_and_delegate  --option "composite"
  "${PYTHON_EXECUTABLE}" -m examples.portable.scripts.export_and_delegate  --option "partition"
  "${PYTHON_EXECUTABLE}" -m examples.portable.scripts.export_and_delegate  --option "whole"

  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/portable/executor_runner:executor_runner -- --model_path "./composite_model.pte"
    buck2 run //examples/portable/executor_runner:executor_runner -- --model_path "./partition_lowered_model.pte"
    buck2 run //examples/portable/executor_runner:executor_runner -- --model_path "./whole.pte"
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

test_demo_backend_delegation
