# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

# Host build of the ESP-NN delegate, espnn_host_runner and
# espnn_host_interpreter, for backends/espressif/test/test_host_runtime.py and
# test_host_interpreter.py (see "On the host" in backends/espressif/README.md).
# -DESPNN_HOST_SANITIZE=ON builds everything with AddressSanitizer and
# UndefinedBehaviorSanitizer.

cmake_minimum_required(VERSION 3.24)
project(espnn_host_runner CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE
)

option(ESPNN_HOST_SANITIZE "Build with ASan and UBSan" OFF)
if(ESPNN_HOST_SANITIZE)
  add_compile_options(
    -fsanitize=address,undefined -fno-sanitize-recover=all
    -fno-omit-frame-pointer
  )
  add_link_options(-fsanitize=address,undefined)
endif()

set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set(EXECUTORCH_BUILD_PYBIND OFF)
set(EXECUTORCH_BUILD_DEVTOOLS OFF)
set(EXECUTORCH_BUILD_XNNPACK OFF)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED OFF)
set(EXECUTORCH_BUILD_PTHREADPOOL OFF)
set(EXECUTORCH_BUILD_CPUINFO OFF)
set(EXECUTORCH_BUILD_TESTS OFF)
add_subdirectory(${EXECUTORCH_ROOT} executorch EXCLUDE_FROM_ALL)
add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/espressif/espnn_runtime espnn_runtime
)
# ESP-IDF's default warnings, as errors.
target_compile_options(
  espnn_backend PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
                        -Werror
)
if(ESPNN_HOST_SANITIZE)
  # ESP-NN's C kernels shift negative values left, as TFLite's reference kernels
  # do; GCC and Clang define the result.
  target_compile_options(esp_nn_host PRIVATE -fno-sanitize=shift-base)
endif()

add_executable(espnn_host_runner espnn_host_runner.cpp)
target_include_directories(espnn_host_runner PRIVATE "${EXECUTORCH_ROOT}/..")
target_link_libraries(
  espnn_host_runner PRIVATE executorch_core extension_data_loader espnn_backend
)

add_executable(espnn_host_interpreter espnn_host_interpreter.cpp)
target_link_libraries(espnn_host_interpreter PRIVATE espnn_backend)
