# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the LICENSE
# file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.13)

# Include FetchContent for Pico SDK
include(FetchContent)

FetchContent_Declare(
  pico_sdk
  GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk.git
  GIT_TAG 2.0.0
)

FetchContent_MakeAvailable(pico_sdk)
set(PICO_SDK_PATH ${pico_sdk_SOURCE_DIR})
include(${PICO_SDK_PATH}/pico_sdk_init.cmake)

project(executorch_pico C CXX ASM)
pico_sdk_init()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure input model file
set(INPUT_MODEL
    "default_model.pte"
    CACHE STRING "Input PTE model file to embed"
)

# Convert relative paths to absolute
if(NOT IS_ABSOLUTE "${INPUT_MODEL}")
  set(INPUT_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_MODEL}")
endif()

message(STATUS "Using model file: ${INPUT_MODEL}")

# Validate model file exists
if(NOT EXISTS "${INPUT_MODEL}")
  message(FATAL_ERROR "Model file '${INPUT_MODEL}' does not exist.")
endif()

# Create model_pte.c if it doesn't exist
set(MODEL_PTE_C "${CMAKE_CURRENT_SOURCE_DIR}/model_pte.c")
if(NOT EXISTS "${MODEL_PTE_C}")
  file(
    WRITE "${MODEL_PTE_C}"
    "#include \"model_pte.h\"\n"
    "\n"
    "const uint8_t model_pte[] __attribute__((aligned(8))) = {\n"
    "    // Model data will be injected here\n"
    "};\n"
    "\n"
    "const unsigned int model_pte_len = sizeof(model_pte);\n"
  )
  message(STATUS "Created initial model_pte.c")
endif()

# Use a stamp file instead of declaring model_pte.c as OUTPUT
set(MODEL_STAMP "${CMAKE_CURRENT_BINARY_DIR}/model_pte.stamp")

add_custom_command(
  OUTPUT ${MODEL_STAMP}
  COMMAND python3 ${EXECUTORCH_ROOT}/executorch/examples/rpi/pte_to_array.py
          --model ${INPUT_MODEL} --file ${MODEL_PTE_C}
  COMMAND ${CMAKE_COMMAND} -E touch ${MODEL_STAMP}
  DEPENDS ${INPUT_MODEL}
          ${EXECUTORCH_ROOT}/executorch/examples/rpi/pte_to_array.py
  COMMENT "Injecting PTE data from '${INPUT_MODEL}' into model_pte.c"
  VERBATIM
)

# Target to ensure model injection happens
add_custom_target(inject_model_data ALL DEPENDS ${MODEL_STAMP})

# Check if source files exist
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
  message(FATAL_ERROR "main.cpp not found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Create the executable
add_executable(
  executorch_pico main.cpp ${MODEL_PTE_C} # Use the full path to be explicit
)

# Ensure model data is injected before building
add_dependencies(executorch_pico inject_model_data)

# Set correct target properties for Pico2/RP2350
set_target_properties(
  executorch_pico PROPERTIES SUFFIX ".elf" OUTPUT_NAME "executorch_pico"
)

# Configure Pico-specific settings
pico_enable_stdio_usb(executorch_pico 1)
pico_enable_stdio_uart(executorch_pico 0)

# Set correct flags for Pico2 (RP2350)
target_compile_options(
  executorch_pico PRIVATE -mcpu=cortex-m33 -mfloat-abi=soft -mthumb
)

target_include_directories(
  executorch_pico
  PRIVATE ${EXECUTORCH_ROOT} ${EXECUTORCH_ROOT}/executorch/third-party
          ${EXECUTORCH_ROOT}/executorch/runtime/core/portable_type/c10
)

target_compile_definitions(
  executorch_pico
  PRIVATE C10_USING_CUSTOM_GENERATED_MACROS EXECUTORCH_ENABLE_LOGGING=OFF
          EXECUTORCH_PAL_DEFAULT=minimal
)

# Optimization flags
target_compile_options(
  executorch_pico PRIVATE -Os -ffunction-sections -fdata-sections
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

set(BAREMETAL_BUILD_DIR ${EXECUTORCH_ROOT}/executorch/cmake-out/)

# Link ExecuTorch and Pico libraries
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

# Only add extra outputs if the target builds successfully
if(TARGET executorch_pico)
  pico_add_extra_outputs(executorch_pico)
endif()
