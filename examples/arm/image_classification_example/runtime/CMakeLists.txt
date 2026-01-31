# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)

project(image_classification_minimal_application)

# Example ExecuTorch demo for bare metal Cortex-M based systems
set(ET_DIR_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
    CACHE PATH "Path to ExecuTorch dir"
)

set(ET_BUILD_DIR_PATH
    "${ET_DIR_PATH}/cmake-out-arm"
    CACHE PATH "Path to ExecuTorch build/install dir"
)
set(ET_INCLUDE_PATH
    "${ET_DIR_PATH}/.."
    CACHE PATH "Path to ExecuTorch headers"
)

set(ET_PTE_FILE_PATH
    ""
    CACHE PATH "Path to ExecuTorch model pte"
)

set(IMAGE_PATH
    ""
    CACHE
      PATH
      "Path to an RGB image to use for the application(e.g. a jpg image of a cat or a dog)"
)

set(ETHOS_SDK_PATH
    "${ET_DIR_PATH}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)

set(PYTHON_EXECUTABLE
    "python"
    CACHE PATH "Define to override python executable used"
)

if(NOT EXISTS "${IMAGE_PATH}")
  message(
    FATAL_ERROR
      "Image not provided. Please provide path to an image for the image classification application and retry."
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(
    FATAL_ERROR
      "pte file not provided. Please provide pte file for the application and retry."
  )
endif()

# Get Ethos-U drivers
include(${ET_DIR_PATH}/backends/arm/scripts/corstone_utils.cmake)
fetch_ethos_u_content(${ETHOS_SDK_PATH} ${ET_DIR_PATH})

if(NOT EXISTS "${ETHOS_SDK_PATH}")
  message(
    FATAL_ERROR
      "The ${ETHOS_SDK_PATH} directory does not exist. Please run examples/arm/setup.sh script and retry."
  )
endif()

find_package(
  executorch REQUIRED HINTS "${ET_BUILD_DIR_PATH}/lib/cmake/ExecuTorch"
)

# The core_platform project defines the corstone-320 target and contains the
# start-up code for the Cortex-M
add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target)

add_executable(img_class_example main.cpp)

target_sources(
  img_class_example
  PRIVATE main.cpp
          ${CMAKE_SOURCE_DIR}/../../executor_runner/arm_memory_allocator.cpp
)

target_link_libraries(
  img_class_example
  PUBLIC executorch
         ethosu_target_init
         extension_runner_util
         quantized_ops_lib
         portable_kernels
         cortex_m_kernels
         cortex_m_ops_lib
)

# We need to include whole archive for the EthosUBackend
target_link_libraries(
  img_class_example PUBLIC "-Wl,--whole-archive" executorch_delegate_ethos_u
                           "-Wl,--no-whole-archive"
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LINK_FILE_EXT ld)
  set(LINK_FILE_OPTION "-T")
  set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
endif()

set(LINK_FILE_OUT_BASE "platform_linker_script")
set(LINK_FILE_IN "${CMAKE_SOURCE_DIR}/../../executor_runner/Corstone-320.ld")
set(LINK_FILE_OUT
    ${CMAKE_CURRENT_BINARY_DIR}/${LINK_FILE_OUT_BASE}.${LINK_FILE_EXT}
)
# The ETHOSU_ARENA symbol is defined in the Corstone-320 linker script and it
# controls the placement of the intermediate tensors in the application memory
# map. Here, because in the compile spec in the AoT flow, we generate pte for
# Shared_Sram, in the application, we set ETHOSU_ARENA to 0 so that the
# intermediate tensors are placed in the SRAM. If you generate a pte for a
# different memory mode, you need to change the placement in the linker script.
# Read
# https://docs.pytorch.org/executorch/stable/backends-arm-ethos-u.html#ethos-u-memory-modes
# for more information.
set(ETHOSU_ARENA "0")
# Generate linker script - we have a few if/else statements in
# Corstone-320.ld/Corstone-300.ld that are compiled into a final linker script.
execute_process(
  COMMAND ${CMAKE_C_COMPILER} ${COMPILER_PREPROCESSOR_OPTIONS} -DETHOSU_ARENA=0
          -o ${LINK_FILE_OUT} ${LINK_FILE_IN}
)
target_link_options(img_class_example PRIVATE "-T" "${LINK_FILE_OUT}")

# Run the pte_to_header.py script to convert the .pte file to an array in a .h
# file
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/../../executor_runner/pte_to_header.py --pte
    ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
)

# Convert an RGB image to an array in a .h file
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/rgb_to_array.py --image
          ${IMAGE_PATH} --output ${CMAKE_CURRENT_BINARY_DIR}/image.h
)

target_include_directories(
  img_class_example
  PRIVATE ${ET_INCLUDE_PATH} ${ET_DIR_PATH}/runtime/core/portable_type/c10
          ${CMAKE_SOURCE_DIR}/../../executor_runner ${CMAKE_CURRENT_BINARY_DIR}
)
