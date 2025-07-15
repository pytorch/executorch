# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2025 Arm Limited and/or its affiliates.

# This source code is licensed under the BSD-style license found in the LICENSE
# file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
project(executorch_pico C CXX ASM)
pico_sdk_init()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Note: Replace simple_model_pte.c with your model's pte files
add_executable(executorch_pico main.cpp simple_model_pte.c)
pico_enable_stdio_usb(executorch_pico 1)
pico_enable_stdio_uart(executorch_pico 0)

# Set correct flags for Pico (Cortex-M0+)
target_compile_options(
  executorch_pico PRIVATE -mcpu=cortex-m0plus -mfloat-abi=soft -mthumb
)

target_include_directories(
  executorch_pico
  PRIVATE ${EXECUTORCH_ROOT} ${EXECUTORCH_ROOT}/executorch/third-party/
          ${EXECUTORCH_ROOT}/executorch/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# Add these optimization flags to your CMakeLists.txt
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Os -ffunction-sections -fdata-sections"
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
# Also add these size optimization definitions
add_compile_definitions(
  EXECUTORCH_ENABLE_LOGGING=OFF EXECUTORCH_PAL_DEFAULT=minimal
)

set(BAREMETAL_BUILD_DIR ${EXECUTORCH_ROOT}/executorch/cmake-out/)
# Link Executorch and Pico libraries
target_link_libraries(
  executorch_pico
  PRIVATE ${BAREMETAL_BUILD_DIR}/lib/libexecutorch.a
          ${BAREMETAL_BUILD_DIR}/lib/libexecutorch_core.a
          -Wl,--whole-archive
          ${BAREMETAL_BUILD_DIR}/lib/libportable_ops_lib.a
          -Wl,--no-whole-archive
          ${BAREMETAL_BUILD_DIR}/lib/libportable_kernels.a
          pico_stdlib
          pico_stdio_usb
)

pico_add_extra_outputs(executorch_pico)
