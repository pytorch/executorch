#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of building devtools/example_runner and use it to run
# an actual model.

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly SCRIPT_DIR

readonly EXECUTORCH_ROOT="${SCRIPT_DIR}/../.."

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

test_cmake_devtools_example_runner() {
  cd "${EXECUTORCH_ROOT}"

  echo "Building example_runner using build_example_runner.sh"
  "${SCRIPT_DIR}/build_example_runner.sh"

  local example_dir=examples/devtools
  local build_dir=cmake-out/${example_dir}

  echo "Exporting MobilenetV2"
  ${PYTHON_EXECUTABLE} -m examples.devtools.scripts.export_bundled_program --model_name="mv2"

  echo 'Running example_runner'
  ${build_dir}/example_runner --bundled_program_path="./mv2_bundled.bpte"
}

test_cmake_devtools_example_runner
