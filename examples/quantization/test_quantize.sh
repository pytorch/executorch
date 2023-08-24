#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end quantization flow.

set -eu

get_shared_lib_ext() {
  UNAME=$(uname)
  if [[ $UNAME == "Darwin" ]];
  then
    EXT=".dylib"
  elif [[ $UNAME == "Linux" ]];
  then
    EXT=".so"
  else
    echo "Unsupported platform $UNAME"
    exit 1
  fi
  echo $EXT
}

test_buck2_quantization() {
  echo "Building quantized ops shared library"
  # TODO: DEBUG flaky MacOS issue
  SO_LIB=$(buck2 build //kernels/quantized:aot_lib -v 3 --show-output | grep "buck-out" | cut -d" " -f2)

  echo "Run example.py"
  ${PYTHON_EXECUTABLE} -m "examples.quantization.example" --so_library="$SO_LIB" --model_name="$1"

  echo 'Running executor_runner'
  buck2 run //examples/executor_runner:executor_runner \
    --config=executorch.register_quantized_ops=1 -- --model_path="./$1.pte"
  # should give correct result

  echo "Removing $1.pte"
  rm "./$1.pte"
}

test_cmake_quantization() {
  echo "Building quantized ops shared library"
  SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

  (rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake -DBUCK2=buck2 \
      -DREGISTER_QUANTIZED_OPS=ON \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build cmake-out -j4

  EXT=$(get_shared_lib_ext)
  SO_LIB="cmake-out/kernels/quantized/libquantized_ops_aot_lib$EXT"

  echo "Run example.py, shared library $SO_LIB"
  ${PYTHON_EXECUTABLE} -m "examples.quantization.example" --so_library="$SO_LIB" --model_name="$1"

  echo 'Running executor_runner'
  cmake-out/executor_runner --model_path="./$1.pte"
  # should give correct result

  echo "Removing $1.pte"
  rm "./$1.pte"
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi
if [[ "$1" == "cmake" ]];
then
  test_cmake_quantization "$2"
elif [[ "$1" == "buck2" ]];
then
  test_buck2_quantization "$2"
else
  test_cmake_quantization "$2"
  test_buck2_quantization "$2"
fi
