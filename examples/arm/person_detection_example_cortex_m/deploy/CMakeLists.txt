# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(person_detection LANGUAGES C CXX)

option(ET_ENABLE_PROFILING
       "Enable ETDump event tracing and transfer profiles to the host" OFF
)
set(EXECUTORCH_BUILD_ARM_ETDUMP
    ${ET_ENABLE_PROFILING}
    CACHE BOOL "Build Arm ETDump support" FORCE
)

get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE
)
include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmRunnerUtils.cmake)

set(CMAKE_SKIP_INSTALL_RULES
    ON
    CACHE BOOL "" FORCE
)
set(ET_PTE_FILE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../artifacts/person_detection.pte"
    CACHE FILEPATH "Person-detection ExecuTorch model"
)
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(FATAL_ERROR "ET_PTE_FILE_PATH does not exist: ${ET_PTE_FILE_PATH}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_qparams_header "${CMAKE_CURRENT_BINARY_DIR}/generated/model_qparams.h")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/generate_qparams_header.py" --pte
    "${ET_PTE_FILE_PATH}" --output "${_qparams_header}"
  RESULT_VARIABLE _qparams_result
  ERROR_VARIABLE _qparams_error
)
if(NOT _qparams_result EQUAL 0)
  message(
    FATAL_ERROR "Could not read model quantization metadata:\n${_qparams_error}"
  )
endif()
set(EXECUTORCH_SELECT_OPS_LIST
    ""
    CACHE STRING "Explicit operator selection"
)

function(_person_detection_track_model_for_selected_ops prefix)
  foreach(kind portable quantized cortex_m)
    add_custom_command(
      OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/${prefix}_${kind}_ops/selected_operators.yaml"
      APPEND
      DEPENDS "${ET_PTE_FILE_PATH}"
    )
  endforeach()
endfunction()

set(ETHOS_SDK_PATH
    "${EXECUTORCH_ROOT}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to the Ethos-U bare-metal SDK"
)
set(SYSTEM_CONFIG
    "Ethos_U55_High_End_Embedded"
    CACHE STRING "Ethos-U system configuration"
)
set(MEMORY_MODE
    "Shared_Sram"
    CACHE STRING "Ethos-U memory mode"
)
set(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE
    0x400000
    CACHE STRING "Persistent method allocator size"
)
set(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE
    0
    CACHE STRING "Temporary allocator size"
)
if(NOT CMSIS_NN_LOCAL_PATH AND EXISTS
                               "${ETHOS_SDK_PATH}/core_software/cmsis-nn"
)
  set(CMSIS_NN_LOCAL_PATH
      "${ETHOS_SDK_PATH}/core_software/cmsis-nn"
      CACHE PATH "Path to existing local CMSIS-NN installation" FORCE
  )
endif()

arm_runner_add_standalone_executorch()
arm_runner_configure_ethos_u_platform(
  SDK_PATH "${ETHOS_SDK_PATH}" SYSTEM_CONFIG "${SYSTEM_CONFIG}" MEMORY_MODE
  "${MEMORY_MODE}"
)
if(TARGET ethosu_core_driver)
  install(TARGETS ethosu_core_driver EXPORT ExecuTorchTargets)
endif()
arm_runner_create_default_selected_ops_libs(
  PREFIX person_detection OP_LIST "${EXECUTORCH_SELECT_OPS_LIST}"
  OPS_FROM_MODEL "${ET_PTE_FILE_PATH}"
)
_person_detection_track_model_for_selected_ops(person_detection)

# The FVP runner preloads the selected model at this application-owned address.
file(SIZE "${ET_PTE_FILE_PATH}" _model_pte_size)
add_executable(
  person_detection
  main.cpp platform/fvp.cpp platform/fvp_lcd.cpp platform/vsi_camera.cpp
  ${EXECUTORCH_ROOT}/examples/arm/common/arm_memory_allocator.cpp
)
target_include_directories(
  person_detection
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}"
          "${CMAKE_CURRENT_BINARY_DIR}/generated"
          "${EXECUTORCH_ROOT}"
          "${EXECUTORCH_ROOT}/runtime/core/portable_type/c10"
          "${EXECUTORCH_ROOT}/examples/arm/common"
          "${CMAKE_CURRENT_BINARY_DIR}"
)
target_compile_definitions(
  person_detection
  PRIVATE
    C10_USING_CUSTOM_GENERATED_MACROS
    PERSON_DETECTION_MODEL_ADDR=0x72000000
    PERSON_DETECTION_MODEL_SIZE=${_model_pte_size}
    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
    ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE}
)
arm_runner_configure_linker_script(
  TARGET person_detection SYSTEM_CONFIG "${SYSTEM_CONFIG}"
)
arm_runner_link_minimal_specs(person_detection)
arm_runner_link_registration_libraries(
  TARGET
  person_detection
  BASE_LIBS
  extension_runner_util
  ethosu_target_init
  REGISTRATION_LIBS
  executorch
  executorch_delegate_ethos_u
  person_detection_portable_ops
  person_detection_quantized_ops
  person_detection_cortex_m_ops
  NORMAL_LIBS
  cortex_m_kernels
  quantized_kernels
  portable_kernels
  kernels_util_all_deps
)
target_link_options(person_detection PRIVATE LINKER:-Map=person_detection.map)
target_link_libraries(person_detection PRIVATE m)

if(ET_ENABLE_PROFILING)
  target_compile_definitions(person_detection PRIVATE ET_EVENT_TRACER_ENABLED)
  target_link_libraries(person_detection PRIVATE etdump flatccrt)
endif()
