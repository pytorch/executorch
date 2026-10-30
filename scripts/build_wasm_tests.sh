#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

for arg in "$@"; do
  if [ "$arg" == "--enable-etdump" ]; then
    ETDUMP_OPTS="-DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
                 -DEXECUTORCH_BUILD_DEVTOOLS=ON \
                 -DFLATCC_ALLOW_WERROR=OFF"
                 # FlatCC generates warnings depending on the compiler version.
                 # This may be removed once the warnings are fixed.
  elif [ "$arg" == "--webgpu" ]; then
    WEBGPU_OPTS="-DEXECUTORCH_BUILD_WEBGPU=ON \
                 -DEXECUTORCH_BUILD_WASM_WEBGPU=ON"
  fi
done

CMAKE_OUT=cmake-out-wasm
if [ -n "${WEBGPU_OPTS:-}" ]; then
  # Separate build tree so the WebGPU configuration never overwrites the
  # CPU-only one.
  CMAKE_OUT=cmake-out-wasm-webgpu
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../"
if [ -n "${WEBGPU_OPTS:-}" ]; then
  python3 -m unittest discover \
      -s extension/wasm/test -p test_webgpu_configuration.py
fi
emcmake cmake . -DEXECUTORCH_BUILD_WASM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_SELECT_OPS_LIST="aten::mm.out,aten::add.out" \
    -DEXECUTORCH_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ${ETDUMP_OPTS} ${WEBGPU_OPTS} -B"${CMAKE_OUT}"

if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

cmake --build ${CMAKE_OUT} --target executorch_wasm_tests -j ${CMAKE_JOBS}
