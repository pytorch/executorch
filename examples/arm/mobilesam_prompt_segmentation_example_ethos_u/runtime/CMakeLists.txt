# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)

project(mobilesam_prompt_segmentation_ethos_u_application)

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
set(MODEL_METADATA_PATH
    ""
    CACHE PATH "Path to MobileSAM exporter metadata"
)
set(IMAGE_PATH
    ""
    CACHE PATH "Path to an RGB image to use for the application"
)
set(MASK_THRESHOLD
    "0.0"
    CACHE STRING "MobileSAM mask logit threshold"
)
set(SYSTEM_CONFIG
    "Ethos_U85_SYS_DRAM_Mid"
    CACHE STRING "Vela system configuration"
)
set(MEMORY_MODE
    "Dedicated_Sram_384KB"
    CACHE STRING "Vela memory mode"
)
set(ETHOS_SDK_PATH
    "${ET_DIR_PATH}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)
set(PYTHON_EXECUTABLE
    "python"
    CACHE PATH "Define to override python executable used"
)
option(ET_SEGMENTATION_DUMP_MASK
       "Dump the predicted segmentation mask as run-length encoded log chunks"
       OFF
)
option(ET_SEGMENTATION_SEMIHOSTING_OUTPUT
       "Emit validation logs through Arm semihosting for FVP smoke runs" ON
)

if(NOT EXISTS "${IMAGE_PATH}")
  message(
    FATAL_ERROR
      "Image not provided. Please provide -DIMAGE_PATH=<image> and retry."
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(
    FATAL_ERROR
      "PTE file not provided. Please provide -DET_PTE_FILE_PATH=<model.pte> and retry."
  )
endif()
if(NOT MODEL_METADATA_PATH)
  get_filename_component(model_directory "${ET_PTE_FILE_PATH}" DIRECTORY)
  get_filename_component(model_name "${ET_PTE_FILE_PATH}" NAME_WE)
  set(MODEL_METADATA_PATH "${model_directory}/${model_name}.json")
endif()
if(NOT EXISTS "${MODEL_METADATA_PATH}")
  message(
    FATAL_ERROR
      "Model metadata not found. Provide -DMODEL_METADATA_PATH=<model.json>."
  )
endif()
if(NOT SYSTEM_CONFIG MATCHES "Ethos_U85")
  message(FATAL_ERROR "This example currently supports Corstone-320/Ethos-U85.")
endif()

include(${ET_DIR_PATH}/backends/arm/scripts/corstone_utils.cmake)
fetch_ethos_u_content(${ETHOS_SDK_PATH} ${ET_DIR_PATH})

if(NOT EXISTS "${ETHOS_SDK_PATH}")
  message(
    FATAL_ERROR
      "The ${ETHOS_SDK_PATH} directory does not exist. Please run examples/arm/setup.sh and retry."
  )
endif()

find_package(
  executorch REQUIRED HINTS "${ET_BUILD_DIR_PATH}/lib/cmake/ExecuTorch"
)

add_corstone_subdirectory(${SYSTEM_CONFIG} ${ETHOS_SDK_PATH})
configure_timing_adapters(${SYSTEM_CONFIG} ${MEMORY_MODE})

add_executable(mobilesam_prompt_segmentation_example main.cpp)
target_sources(
  mobilesam_prompt_segmentation_example
  PRIVATE main.cpp
          ${CMAKE_SOURCE_DIR}/../../executor_runner/arm_memory_allocator.cpp
)
target_link_libraries(
  mobilesam_prompt_segmentation_example
  PUBLIC executorch
         ethosu_target_init
         extension_runner_util
         quantized_ops_lib
         portable_kernels
         cortex_m_kernels
         cortex_m_ops_lib
)

include(${ET_DIR_PATH}/tools/cmake/Utils.cmake)
executorch_target_link_options_shared_lib(executorch_delegate_ethos_u)
target_link_libraries(
  mobilesam_prompt_segmentation_example PUBLIC executorch_delegate_ethos_u
)

if(MEMORY_MODE MATCHES "^Dedicated_Sram($|_)")
  set(ETHOSU_ARENA "1")
  if(NOT DEFINED ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set(ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x1000000)
  endif()
  if(NOT DEFINED ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set_ethosu_dedicated_sram_fast_scratch_size(
      ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE "${MEMORY_MODE}"
    )
  endif()
else()
  set(ETHOSU_ARENA "0")
  if(NOT DEFINED ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set(ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x400000)
  endif()
endif()
if(NOT DEFINED ET_SEGMENTATION_METHOD_ALLOCATOR_POOL_SIZE)
  set(ET_SEGMENTATION_METHOD_ALLOCATOR_POOL_SIZE 0x4000000)
endif()

target_compile_definitions(
  mobilesam_prompt_segmentation_example
  PRIVATE
    ET_SEGMENTATION_METHOD_ALLOCATOR_POOL_SIZE=${ET_SEGMENTATION_METHOD_ALLOCATOR_POOL_SIZE}
    ET_SEGMENTATION_MASK_THRESHOLD=${MASK_THRESHOLD}
    ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
)
if(DEFINED ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    mobilesam_prompt_segmentation_example
    PRIVATE
      ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
  )
endif()
if(ET_SEGMENTATION_DUMP_MASK)
  target_compile_definitions(
    mobilesam_prompt_segmentation_example PRIVATE ET_SEGMENTATION_DUMP_MASK
  )
endif()
if(ET_SEGMENTATION_SEMIHOSTING_OUTPUT)
  target_compile_definitions(
    mobilesam_prompt_segmentation_example
    PRIVATE ET_SEGMENTATION_SEMIHOSTING_OUTPUT
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LINK_FILE_EXT ld)
  set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
endif()

set(LINK_FILE_OUT_BASE "platform_linker_script")
set(LINK_FILE_IN "${CMAKE_SOURCE_DIR}/../../executor_runner/Corstone-320.ld")
set(LINK_FILE_OUT
    ${CMAKE_CURRENT_BINARY_DIR}/${LINK_FILE_OUT_BASE}.${LINK_FILE_EXT}
)
execute_process(
  COMMAND ${CMAKE_C_COMPILER} ${COMPILER_PREPROCESSOR_OPTIONS} -DETHOSU_MODEL=1
          -DETHOSU_ARENA=${ETHOSU_ARENA} -o ${LINK_FILE_OUT} ${LINK_FILE_IN}
)
target_link_options(
  mobilesam_prompt_segmentation_example PRIVATE "-T" "${LINK_FILE_OUT}"
)

set(MODEL_PTE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/model_pte.h")
add_custom_command(
  OUTPUT "${MODEL_PTE_HEADER}"
  COMMAND
    ${PYTHON_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/../../executor_runner/pte_to_header.py --pte
    ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ET_PTE_FILE_PATH}
          ${CMAKE_SOURCE_DIR}/../../executor_runner/pte_to_header.py
  VERBATIM
)
set(IMAGE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/image.h")
add_custom_command(
  OUTPUT "${IMAGE_HEADER}"
  COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/image_to_array.py --image
    ${IMAGE_PATH} --metadata ${MODEL_METADATA_PATH} --output ${IMAGE_HEADER}
  DEPENDS ${IMAGE_PATH} ${MODEL_METADATA_PATH}
          ${CMAKE_SOURCE_DIR}/image_to_array.py
  VERBATIM
)
target_sources(
  mobilesam_prompt_segmentation_example PRIVATE ${MODEL_PTE_HEADER}
                                                ${IMAGE_HEADER}
)

target_include_directories(
  mobilesam_prompt_segmentation_example
  PRIVATE ${ET_INCLUDE_PATH} ${ET_DIR_PATH}/runtime/core/portable_type/c10
          ${CMAKE_SOURCE_DIR}/../../executor_runner ${CMAKE_CURRENT_BINARY_DIR}
)
