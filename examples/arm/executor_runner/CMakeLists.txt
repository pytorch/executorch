# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(arm_executor_runner_redirect LANGUAGES C CXX)
  message(
    FATAL_ERROR
      "Configure standalone arm_executor_runner builds from ${CMAKE_CURRENT_LIST_DIR}/standalone instead of ${CMAKE_CURRENT_LIST_DIR}."
  )
endif()

get_filename_component(
  _default_executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)

if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT
      "${_default_executorch_root}"
      CACHE PATH "Path to an ExecuTorch checkout"
  )
endif()

set(ET_DIR_PATH
    "${EXECUTORCH_ROOT}"
    CACHE PATH "Kept for backward compatibility; synonym for EXECUTORCH_ROOT"
)
if(NOT EXISTS "${EXECUTORCH_ROOT}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project."
  )
endif()

if(NOT COMMAND executorch_target_link_options_shared_lib)
  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
endif()
executorch_get_build_include_dir(
  "${EXECUTORCH_ROOT}" EXECUTORCH_SOURCE_INCLUDE_DIR
)
if(NOT DEFINED ET_INCLUDE_PATH)
  set(ET_INCLUDE_PATH
      "${EXECUTORCH_SOURCE_INCLUDE_DIR}"
      CACHE PATH "Include root for ExecuTorch source headers"
  )
endif()

set(ET_NUM_INFERENCES
    "1"
    CACHE STRING "Number of inferences to run"
)

option(ET_LOG_DUMP_INPUT "Dump input in log" OFF)
option(ET_LOG_DUMP_OUTPUT "Dump output in log" ON)
option(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING
       "Report Ethos-U PMU statistics per delegate" OFF
)
set(ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES
    "16"
    CACHE STRING "Maximum delegates tracked by per-delegate profiling"
)

option(ET_BUNDLE_IO "Set to compile in BundleIO support" OFF)
set(BUNDLED_PROGRAM_LIBRARY_DIR
    ""
    CACHE
      PATH
      "Optional directory that contains a prebuilt libbundled_program.a when ET_BUNDLE_IO is enabled without building devtools."
)
set(ET_ATOL
    "0.01"
    CACHE STRING "Set atol to use for BundleIO testing (Requires ET_BUNDLE_IO)"
)
set(ET_RTOL
    "0.01"
    CACHE STRING "Set atol to use for BundleIO testing (Requires ET_BUNDLE_IO)"
)

option(
  ET_DUMP_OUTPUTS
  "Collect and print outputs as a base64 buffer in the log (Requires EXECUTORCH_ENABLE_EVENT_TRACER)"
  OFF
)
option(
  ET_DUMP_INTERMEDIATE_OUTPUTS
  "Collect and print intermediate outputs as a base64 buffer in the log (Requires EXECUTORCH_ENABLE_EVENT_TRACER)"
  OFF
)
set(ET_DEBUG_BUFFER_SIZE
    "2097152"
    CACHE
      STRING
      "Size of buffer to collect intermediate outputs/outputs buffers (Requires EXECUTORCH_ENABLE_EVENT_TRACER and ET_DUMP_OUTPUTS or ET_DUMP_INTERMEDIATE_OUTPUTS)"
)

option(SEMIHOSTING "Enable semihosting" OFF)

set(ET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE
    ""
    CACHE STRING
          "Size of the SRAM-backed planned fast memory pool used for mem_id=3"
)

include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmRunnerUtils.cmake)

arm_runner_define_cache_options()
arm_runner_validate_model_source(ALLOW_SEMIHOSTING)
arm_runner_require_python()
arm_runner_require_baremetal_targets()

message(STATUS "SYSTEM_CONFIG is ${SYSTEM_CONFIG}")
message(STATUS "MEMORY_MODE is ${MEMORY_MODE}")
message(STATUS "ET_NUM_INFERENCES is ${ET_NUM_INFERENCES}")

# By default, use 2MB of temporary scratch buffer. For Dedicated_Sram, use 64MB
# for the temporary scratch buffer and derive the fast scratch buffer size from
# the memory mode suffix, e.g. Dedicated_Sram_384KB. Allow -D overrides.
if(MEMORY_MODE MATCHES "^Dedicated_Sram($|_)")
  if(NOT DEFINED ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set_ethosu_dedicated_sram_fast_scratch_size(
      ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE "${MEMORY_MODE}"
    )
  endif()
endif()
message(
  STATUS
    "ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE = ${ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}"
)
message(
  STATUS
    "ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE = ${ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}"
)
message(
  STATUS
    "ET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE = ${ET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE}"
)

arm_runner_configure_ethos_u_platform(
  SDK_PATH "${ETHOS_SDK_PATH}" SYSTEM_CONFIG "${SYSTEM_CONFIG}" MEMORY_MODE
  "${MEMORY_MODE}"
)

# The arm_executor_runner executable
add_executable(arm_executor_runner)

target_sources(
  arm_executor_runner PRIVATE arm_executor_runner.cpp arm_perf_monitor.cpp
                              ../common/arm_memory_allocator.cpp
)

arm_runner_configure_linker_script(
  TARGET arm_executor_runner SYSTEM_CONFIG "${SYSTEM_CONFIG}"
)
if(EXECUTORCH_OPTIMIZE_SIZE)
  arm_runner_link_minimal_specs(arm_executor_runner)
endif()

# Create ops libraries from included operators. Handle portable, cortex_m, and
# quantized ops.
set(_arm_runner_dtype_selective_build
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
)
if(NOT ET_PTE_FILE_PATH)
  set(_arm_runner_dtype_selective_build "")
endif()

arm_runner_create_default_selected_ops_libs(
  PREFIX
  arm
  SUFFIX
  _lib
  OP_LIST
  "${EXECUTORCH_SELECT_OPS_LIST}"
  OPS_FROM_MODEL
  "${ET_PTE_FILE_PATH}"
  DTYPE_SELECTIVE_BUILD
  "${_arm_runner_dtype_selective_build}"
  DEPS
  executorch_core
)

arm_runner_create_selected_ops_lib(
  PRIM_OPS
  LIB_NAME
  arm_prim_ops_lib
  OP_LIST
  "${EXECUTORCH_SELECT_OPS_LIST}"
  OPS_FROM_MODEL
  "${ET_PTE_FILE_PATH}"
  DEPS
  executorch_core
)

set(_arm_runner_selected_ops_libs arm_portable_ops_lib arm_prim_ops_lib
                                  arm_quantized_ops_lib arm_cortex_m_ops_lib
)
set(_arm_runner_normal_archive_libs
    portable_kernels kernels_util_all_deps arm_prim_ops_lib_impl
    quantized_kernels cortex_m_kernels cmsis-nn
)
arm_runner_link_registration_libraries(
  TARGET
  arm_executor_runner
  SCOPE
  PUBLIC
  BASE_LIBS
  extension_runner_util
  ethosu_target_init
  REGISTRATION_LIBS
  executorch_delegate_ethos_u
  ${_arm_runner_selected_ops_libs}
  NORMAL_LIBS
  executorch_core
  ${_arm_runner_normal_archive_libs}
  SUPPRESS_LIBS
  executorch
)

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  target_compile_definitions(arm_executor_runner PUBLIC ET_EVENT_TRACER_ENABLED)

  target_link_libraries(arm_executor_runner PUBLIC etdump flatccrt)
endif()

if(ET_BUNDLE_IO)
  if(TARGET bundled_program)
    target_link_libraries(arm_executor_runner PUBLIC bundled_program)
    target_link_directories(
      arm_executor_runner PRIVATE $<TARGET_FILE_DIR:bundled_program>
    )
  else()
    set(_bundled_program_library "")
    set(_bundled_program_search_paths "")
    if(ET_BUILD_DIR_PATH)
      list(
        APPEND
        _bundled_program_search_paths
        "${ET_BUILD_DIR_PATH}"
        "${ET_BUILD_DIR_PATH}/lib"
        "${ET_BUILD_DIR_PATH}/devtools/bundled_program"
        "${ET_BUILD_DIR_PATH}/devtools/bundled_program/lib"
      )
    endif()
    if(BUNDLED_PROGRAM_LIBRARY_DIR)
      list(APPEND _bundled_program_search_paths
           "${BUNDLED_PROGRAM_LIBRARY_DIR}"
      )
    endif()
    if(_bundled_program_search_paths)
      list(REMOVE_DUPLICATES _bundled_program_search_paths)
      # BundleIO can reuse a separate ExecuTorch build tree where
      # bundled_program is not part of this CMake graph. Restrict the fallback
      # lookup to the caller-provided build directories so we do not
      # accidentally pick up an unrelated library from the host system.
      find_library(
        _bundled_program_library
        NAMES bundled_program
        PATHS ${_bundled_program_search_paths}
        NO_DEFAULT_PATH
      )
    endif()
    if(_bundled_program_library)
      target_link_libraries(
        arm_executor_runner PUBLIC ${_bundled_program_library}
      )
    else()
      message(
        FATAL_ERROR
          "ET_BUNDLE_IO enabled but bundled_program is unavailable. Either configure this build with EXECUTORCH_BUILD_DEVTOOLS=ON so the target exists or set BUNDLED_PROGRAM_LIBRARY_DIR/ET_BUILD_DIR_PATH to a build directory that contains libbundled_program.a."
      )
    endif()
  endif()
endif()

# Ensure the ELF lands next to the CMake build tree so run.sh (and downstream
# tooling) can locate it deterministically regardless of multi-config vs
# single-config generators.
target_link_options(
  arm_executor_runner PUBLIC LINKER:-Map=arm_executor_runner.map
)
# Reuse a parent build's output directory if it already set one; otherwise keep
# the runner ELF next to this build tree so run.sh can find it predictably.
arm_runner_configure_runtime_output(arm_executor_runner "${CMAKE_BINARY_DIR}")

# Sanitizers
if(CMAKE_BUILD_TYPE MATCHES "UndefinedSanitizer")
  set(_et_runner_ubsan_flag -fsanitize=undefined)
  target_compile_options(arm_executor_runner PRIVATE ${_et_runner_ubsan_flag})
  target_link_options(arm_executor_runner PRIVATE ${_et_runner_ubsan_flag})
  if(NOT TARGET executorch_ubsan)
    add_subdirectory(
      ${EXECUTORCH_ROOT}/examples/arm/ubsan
      ${CMAKE_CURRENT_BINARY_DIR}/ubsan_runtime
    )
  endif()
  target_link_directories(
    arm_executor_runner PRIVATE $<TARGET_FILE_DIR:executorch_ubsan>
  )
  target_link_libraries(arm_executor_runner PRIVATE executorch_ubsan)
endif()

if(CMAKE_BUILD_TYPE MATCHES "AddressSanitizer")
  set(_et_runner_asan_flags -fsanitize=kernel-address -fasan-shadow-offset=0x0)
  target_compile_options(arm_executor_runner PRIVATE ${_et_runner_asan_flags})
  target_link_options(arm_executor_runner PRIVATE ${_et_runner_asan_flags})
  if(NOT TARGET executorch_asan)
    add_subdirectory(
      ${EXECUTORCH_ROOT}/examples/arm/asan
      ${CMAKE_CURRENT_BINARY_DIR}/asan_runtime
    )
  endif()
  target_link_libraries(arm_executor_runner PRIVATE executorch_asan)
  target_compile_definitions(
    arm_executor_runner PRIVATE EXECUTORCH_ENABLE_ADDRESS_SANITIZER
  )
endif()

# ET headers and generated headers includes
target_include_directories(
  arm_executor_runner
  PRIVATE ${ET_INCLUDE_PATH} ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
          ${EXECUTORCH_ROOT}/examples/arm/common ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(
  arm_executor_runner PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)

arm_runner_configure_model(
  PUBLIC
  TARGET
  arm_executor_runner
  PTE_FILE
  "${ET_PTE_FILE_PATH}"
  MODEL_PTE_ADDR
  "${ET_MODEL_PTE_ADDR}"
  MODEL_PTE_SIZE
  "${ET_MODEL_PTE_SIZE}"
)

if(ET_NUM_INFERENCES)
  target_compile_definitions(
    arm_executor_runner PUBLIC ET_NUM_INFERENCES=${ET_NUM_INFERENCES}
  )
endif()

if(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  if(NOT ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES MATCHES "^[1-9][0-9]*$")
    message(
      FATAL_ERROR
        "ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES must be a positive integer"
    )
  endif()
  target_compile_definitions(
    arm_executor_runner
    PRIVATE
      ET_ARM_ETHOSU_PER_DELEGATE_PROFILING
      ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES=${ET_ARM_ETHOSU_MAX_PROFILED_DELEGATES}
  )
  target_compile_definitions(
    executorch_delegate_ethos_u PRIVATE ET_ARM_ETHOSU_PER_DELEGATE_PROFILING
  )
endif()

if(ET_LOG_DUMP_INPUT)
  target_compile_definitions(arm_executor_runner PUBLIC ET_LOG_DUMP_INPUT)
endif()

if(ET_LOG_DUMP_OUTPUT)
  target_compile_definitions(arm_executor_runner PUBLIC ET_LOG_DUMP_OUTPUT)
endif()

# Devtool BundleIO: Use Bundle PTE with input and reference output included to
# check if it matches.

if(ET_BUNDLE_IO)
  target_compile_definitions(arm_executor_runner PUBLIC ET_BUNDLE_IO)
endif()

if(ET_ATOL)
  target_compile_definitions(arm_executor_runner PUBLIC ET_ATOL=${ET_ATOL})
endif()

if(ET_RTOL)
  target_compile_definitions(arm_executor_runner PUBLIC ET_RTOL=${ET_RTOL})
endif()

# Devtools ETDump: Speed and dumping output

if(ET_DUMP_OUTPUTS)
  target_compile_definitions(arm_executor_runner PUBLIC ET_DUMP_OUTPUTS)
endif()

if(ET_DUMP_INTERMEDIATE_OUTPUTS)
  target_compile_definitions(
    arm_executor_runner PUBLIC ET_DUMP_INTERMEDIATE_OUTPUTS
  )
endif()

if(ET_DEBUG_BUFFER_SIZE)
  target_compile_definitions(
    arm_executor_runner PUBLIC ET_DEBUG_BUFFER_SIZE=${ET_DEBUG_BUFFER_SIZE}
  )
endif()

# Semihosting FVP (FVP Simulator can access host filesystem)

if(SEMIHOSTING)
  target_compile_definitions(arm_executor_runner PUBLIC SEMIHOSTING)
endif()

# Memory buffer sizes for Executorch flow

if(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    arm_executor_runner
    PUBLIC
      ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE}
  )
endif()

if(DEFINED ET_ARM_BAREMETAL_SEMIHOSTING_FILE_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    arm_executor_runner
    PUBLIC
      ET_ARM_BAREMETAL_SEMIHOSTING_FILE_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_SEMIHOSTING_FILE_ALLOCATOR_POOL_SIZE}
  )
endif()

target_compile_definitions(
  arm_executor_runner
  PUBLIC
    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
)
if(DEFINED ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    arm_executor_runner
    PUBLIC
      ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
  )
endif()

if(ET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE)
  target_compile_definitions(
    arm_executor_runner
    PUBLIC
      ET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE=${ET_ARM_BAREMETAL_PLANNED_FAST_MEMORY_SIZE}
  )
endif()

# Fixup compilation of retarget.c
if(SEMIHOSTING)
  # Remove this when MLBEDSW-8910 is closed.
  set_source_files_properties(
    ${ETHOS_SDK_PATH}/core_platform/targets/${TARGET_BOARD}/retarget.c
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()
