# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)

project(silero_vad_ethos_u_application)

set(ET_DIR_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
    CACHE PATH "Path to ExecuTorch dir"
)
include(${ET_DIR_PATH}/tools/cmake/Utils.cmake)
executorch_get_build_include_dir("${ET_DIR_PATH}" EXECUTORCH_SOURCE_INCLUDE_DIR)

set(ET_BUILD_DIR_PATH
    "${ET_DIR_PATH}/cmake-out-arm"
    CACHE PATH "Path to ExecuTorch build/install dir"
)
set(ET_INCLUDE_PATH
    "${EXECUTORCH_SOURCE_INCLUDE_DIR}"
    CACHE PATH "Path to ExecuTorch headers"
)

set(ET_PTE_FILE_PATH
    ""
    CACHE PATH "Path to ExecuTorch model pte"
)

set(AUDIO_PATH
    ""
    CACHE PATH "Path to a 16 kHz mono WAV file for the application"
)

set(VAD_THRESHOLD
    "0.5"
    CACHE STRING "Speech probability threshold"
)

set(MAX_AUDIO_SAMPLES
    "40000"
    CACHE STRING "Maximum number of WAV samples to embed"
)

option(ENABLE_SEMIHOSTING_OUTPUT
       "Write vad_probs.bin through semihosting for host-side validation" OFF
)

set(ETHOS_SDK_PATH
    "${ET_DIR_PATH}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)

set(PYTHON_EXECUTABLE
    "python"
    CACHE PATH "Define to override python executable used"
)

if(NOT EXISTS "${AUDIO_PATH}")
  message(
    FATAL_ERROR
      "Audio not provided. Please provide path to a 16 kHz mono WAV file and retry."
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(
    FATAL_ERROR
      "pte file not provided. Please provide pte file for the application and retry."
  )
endif()

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

set(CORE_SOFTWARE_RTOS
    "None"
    CACHE STRING "Select RTOS to include. (None, All, ThreadX, FreeRTOS)"
)
add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target)

set(EXECUTORCH_ROOT ${ET_DIR_PATH})
include(${ET_DIR_PATH}/tools/cmake/Utils.cmake)
include(${ET_DIR_PATH}/tools/cmake/Codegen.cmake)

gen_selected_ops(
  LIB_NAME
  "silero_vad_portable_ops_lib"
  OPS_SCHEMA_YAML
  ""
  ROOT_OPS
  ""
  INCLUDE_ALL_OPS
  ""
  OPS_FROM_MODEL
  "${ET_PTE_FILE_PATH}"
  DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
)

generate_bindings_for_kernels(
  LIB_NAME "silero_vad_portable_ops_lib" FUNCTIONS_YAML
  ${ET_DIR_PATH}/kernels/portable/functions.yaml DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
)

gen_operators_lib(
  LIB_NAME
  "silero_vad_portable_ops_lib"
  KERNEL_LIBS
  portable_kernels
  DEPS
  executorch
  DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
)

add_executable(silero_vad_ethos_u main.cpp)

target_sources(
  silero_vad_ethos_u
  PRIVATE main.cpp ${ET_DIR_PATH}/examples/arm/common/arm_memory_allocator.cpp
)

target_link_libraries(
  silero_vad_ethos_u
  PUBLIC executorch ethosu_target_init extension_runner_util quantized_ops_lib
         silero_vad_portable_ops_lib portable_kernels
)

executorch_target_link_options_shared_lib(executorch_delegate_ethos_u)

target_link_libraries(silero_vad_ethos_u PUBLIC executorch_delegate_ethos_u)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LINK_FILE_EXT ld)
  set(LINK_FILE_OPTION "-T")
  set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
endif()

set(LINK_FILE_OUT_BASE "platform_linker_script")
set(LINK_FILE_IN
    "${ET_DIR_PATH}/backends/arm/cmake/linker_scripts/Corstone-320.ld"
)
set(LINK_FILE_OUT
    ${CMAKE_CURRENT_BINARY_DIR}/${LINK_FILE_OUT_BASE}.${LINK_FILE_EXT}
)
execute_process(
  COMMAND ${CMAKE_C_COMPILER} ${COMPILER_PREPROCESSOR_OPTIONS} -DETHOSU_ARENA=0
          -o ${LINK_FILE_OUT} ${LINK_FILE_IN} COMMAND_ERROR_IS_FATAL ANY
)
target_link_options(silero_vad_ethos_u PRIVATE "-T" "${LINK_FILE_OUT}")

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${ET_DIR_PATH}/examples/arm/common/pte_to_header.py
    --pte ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/wav_to_array.py --audio
    ${AUDIO_PATH} --output ${CMAKE_CURRENT_BINARY_DIR}/audio.h --max-samples
    ${MAX_AUDIO_SAMPLES} COMMAND_ERROR_IS_FATAL ANY
)

target_compile_definitions(
  silero_vad_ethos_u PRIVATE VAD_THRESHOLD=${VAD_THRESHOLD}
)
if(ENABLE_SEMIHOSTING_OUTPUT)
  target_compile_definitions(
    silero_vad_ethos_u PRIVATE ENABLE_SEMIHOSTING_OUTPUT
  )
endif()

target_include_directories(
  silero_vad_ethos_u
  PRIVATE ${ET_INCLUDE_PATH} ${ET_DIR_PATH}/runtime/core/portable_type/c10
          ${ET_DIR_PATH}/examples/arm/common ${CMAKE_CURRENT_BINARY_DIR}
)
