# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_arm_executor_runner_standalone FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(arm_executor_runner LANGUAGES C CXX)
  set(_arm_executor_runner_standalone TRUE)
endif()

if(_arm_executor_runner_standalone)
  set(ARM_EXECUTOR_RUNNER_STANDALONE
      ${_arm_executor_runner_standalone}
      CACHE
        BOOL
        "Indicates arm_executor_runner was configured as a standalone project"
        FORCE
  )
endif()

# Derive the ExecuTorch checkout root by walking upward until we find the shared
# tooling directory rather than assuming a fixed relative depth.
set(_candidate_executorch_root "${CMAKE_CURRENT_LIST_DIR}")
set(_default_executorch_root "")
while(TRUE)
  if(EXISTS "${_candidate_executorch_root}/tools/cmake/common/preset.cmake")
    set(_default_executorch_root "${_candidate_executorch_root}")
    break()
  endif()
  get_filename_component(_parent_dir "${_candidate_executorch_root}" DIRECTORY)
  if(_parent_dir STREQUAL "${_candidate_executorch_root}")
    break()
  endif()
  set(_candidate_executorch_root "${_parent_dir}")
endwhile()

if(_default_executorch_root STREQUAL "")
  message(
    FATAL_ERROR
      "Unable to locate an ExecuTorch checkout near ${CMAKE_CURRENT_LIST_DIR}. Please rerun CMake with -DEXECUTORCH_ROOT=<path-to-executorch>."
  )
endif()

if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT
      "${_default_executorch_root}"
      CACHE PATH "Path to an ExecuTorch checkout"
  )
endif()

if(NOT EXISTS "${EXECUTORCH_ROOT}/CMakeLists.txt")
  if(EXISTS "${_default_executorch_root}/CMakeLists.txt")
    message(
      WARNING
        "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project. Falling back to ${_default_executorch_root}."
    )
    set(EXECUTORCH_ROOT
        "${_default_executorch_root}"
        CACHE PATH "Path to an ExecuTorch checkout" FORCE
    )
    set(ET_DIR_PATH
        "${EXECUTORCH_ROOT}"
        CACHE PATH
              "Kept for backward compatibility; synonym for EXECUTORCH_ROOT"
              FORCE
    )
  else()
    message(
      FATAL_ERROR
        "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project."
    )
  endif()
endif()

if(_arm_executor_runner_standalone)
  set(_executorch_preset_cmake
      "${EXECUTORCH_ROOT}/tools/cmake/common/preset.cmake"
  )
  if(EXISTS "${_executorch_preset_cmake}")
    include("${_executorch_preset_cmake}")
    if(NOT DEFINED EXECUTORCH_BUILD_PRESET_FILE)
      set(EXECUTORCH_BUILD_PRESET_FILE
          "${EXECUTORCH_ROOT}/tools/cmake/preset/arm_baremetal.cmake"
          CACHE PATH "Preset used when configuring the standalone runner"
      )
    endif()
    load_build_preset()
  endif()
  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
endif()

if(_arm_executor_runner_standalone)
  option(ARM_EXECUTOR_RUNNER_SKIP_INSTALL_RULES
         "Skip install() rules for standalone arm_executor_runner builds" ON
  )
  if(DEFINED CMAKE_SKIP_INSTALL_RULES)
    set(_arm_runner_skip_install_rules "${CMAKE_SKIP_INSTALL_RULES}")
  endif()
  if(ARM_EXECUTOR_RUNNER_SKIP_INSTALL_RULES)
    set(CMAKE_SKIP_INSTALL_RULES ON)
  endif()
  foreach(
    _opt
    EXECUTORCH_BUILD_ARM_BAREMETAL EXECUTORCH_BUILD_CORTEX_M
    EXECUTORCH_BUILD_KERNELS_QUANTIZED EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL
  )
    if(NOT DEFINED ${_opt})
      set(${_opt}
          ON
          CACHE BOOL "" FORCE
      )
    endif()
  endforeach()
  set(EXECUTORCH_SKIP_ARM_EXECUTOR_RUNNER
      ON
      CACHE BOOL "" FORCE
  )
  # When configured on its own, pull ExecuTorch in-tree so all required targets
  # (delegates, kernels, runner util, etc.) are built from the same checkout.
  # The preset above (or manual cache overrides) must configure any top-level
  # ExecuTorch options a superbuild would normally supply.
  add_subdirectory(
    ${EXECUTORCH_ROOT} ${CMAKE_BINARY_DIR}/executorch EXCLUDE_FROM_ALL
  )
  if(ARM_EXECUTOR_RUNNER_SKIP_INSTALL_RULES)
    if(DEFINED _arm_runner_skip_install_rules)
      set(CMAKE_SKIP_INSTALL_RULES "${_arm_runner_skip_install_rules}")
    else()
      unset(CMAKE_SKIP_INSTALL_RULES)
    endif()
  endif()
endif()

option(
  ET_MODEL_PTE_ADDR
  "Place in memory that the PTE file is located/flashed, if set to OFF the PTE is built into the code as a big data area."
  OFF
)

set(ET_NUM_INFERENCES
    "1"
    CACHE STRING "Number of inferences to run"
)

option(ET_LOG_DUMP_INPUT "Dump input in log" OFF)
option(ET_LOG_DUMP_OUTPUT "Dump output in log" ON)

option(ET_BUNDLE_IO "Set to compile in BundleIO support" OFF)
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

option(
  ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE
  "Set ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE to specify memory alloction pool size"
  OFF
)

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

include(${EXECUTORCH_ROOT}/backends/arm/scripts/corstone_utils.cmake)

set(ETHOS_SDK_PATH
    "${EXECUTORCH_ROOT}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)

set(_ethos_u_content_ready FALSE)
if(EXISTS "${ETHOS_SDK_PATH}/core_platform"
   AND EXISTS "${ETHOS_SDK_PATH}/core_software"
)
  set(_ethos_u_content_ready TRUE)
endif()

set(_fetch_ethos_u_default ON)
if(_ethos_u_content_ready)
  set(_fetch_ethos_u_default OFF)
endif()

option(FETCH_ETHOS_U_CONTENT
       "Fetch ethos_u dependencies instead of relying on pre-downloads"
       ${_fetch_ethos_u_default}
)

if(FETCH_ETHOS_U_CONTENT AND NOT _ethos_u_content_ready)
  fetch_ethos_u_content(${ETHOS_SDK_PATH} ${EXECUTORCH_ROOT})
  if(EXISTS "${ETHOS_SDK_PATH}/core_platform"
     AND EXISTS "${ETHOS_SDK_PATH}/core_software"
  )
    set(_ethos_u_content_ready TRUE)
  else()
    message(
      FATAL_ERROR
        "Failed to fetch Ethos-U content into ${ETHOS_SDK_PATH}. Inspect the fetch logs above."
    )
  endif()
endif()

if(NOT _ethos_u_content_ready)
  message(
    FATAL_ERROR
      "No Ethos-U content found at ${ETHOS_SDK_PATH}. Run examples/arm/setup.sh or enable FETCH_ETHOS_U_CONTENT=ON."
  )
endif()

if(NOT DEFINED ET_MODEL_PTE_ADDR
   AND NOT DEFINED ET_PTE_FILE_PATH
   AND NOT DEFINED SEMIHOSTING
)
  message(
    FATAL_ERROR
      "You must specify one of "
      "ET_MODEL_PTE_ADDR - .pte (or .bpte) on address, "
      "ET_PTE_FILE_PATH - .pte (or .bpte) built into the binary or "
      "SEMIHOSTING - pte and input via host filesystem (use by pytest)"
  )
endif()

set(ET_PTE_FILE_PATH
    ""
    CACHE PATH "Path to ExecuTorch model pte"
)

set(_arm_runner_placeholder_pte
    "${CMAKE_CURRENT_LIST_DIR}/data/placeholder.pte"
)

option(ARM_EXECUTOR_RUNNER_ALLOW_PLACEHOLDER_PTE
       "Stage a placeholder PTE when ET_PTE_FILE_PATH is missing"
       ${_arm_executor_runner_standalone}
)

set(_arm_runner_enabled TRUE)
if(NOT SEMIHOSTING
   AND NOT ET_MODEL_PTE_ADDR
   AND "${ET_PTE_FILE_PATH}" STREQUAL ""
)
  set(_arm_runner_enabled FALSE)
endif()

if(NOT SEMIHOSTING
   AND NOT ET_MODEL_PTE_ADDR
   AND NOT "${ET_PTE_FILE_PATH}" STREQUAL ""
)
  if(NOT EXISTS "${ET_PTE_FILE_PATH}")
    if(ARM_EXECUTOR_RUNNER_ALLOW_PLACEHOLDER_PTE
       AND EXISTS "${_arm_runner_placeholder_pte}"
    )
      get_filename_component(
        _arm_runner_pte_dir "${ET_PTE_FILE_PATH}" DIRECTORY
      )
      if(_arm_runner_pte_dir)
        file(MAKE_DIRECTORY "${_arm_runner_pte_dir}")
      endif()
      file(COPY_FILE "${_arm_runner_placeholder_pte}" "${ET_PTE_FILE_PATH}"
           ONLY_IF_DIFFERENT
      )
      message(
        STATUS
          "arm_executor_runner: staged placeholder model at ${ET_PTE_FILE_PATH}; run.sh will replace it with a real export."
      )
    else()
      message(
        FATAL_ERROR
          "ET_PTE_FILE_PATH is set to ${ET_PTE_FILE_PATH}, but no file was found. Provide a valid PTE or enable ARM_EXECUTOR_RUNNER_ALLOW_PLACEHOLDER_PTE=ON."
      )
    endif()
  endif()
endif()

# Ensure runner util is available even if the top-level build did not enable it.
if(NOT TARGET extension_runner_util)
  if(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL
     OR EXECUTORCH_BUILD_EXECUTOR_RUNNER
     OR DEFINED _executorch_extensions
  )
    message(
      WARNING
        "extension_runner_util target missing even though EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON. Falling back to a local build for backward compatibility."
    )
  endif()
  message(STATUS "arm_executor_runner enabling extension_runner_util locally")
  add_subdirectory(
    ${EXECUTORCH_ROOT}/extension/runner_util
    ${CMAKE_BINARY_DIR}/extension_runner_util_baremetal
  )
endif()

if(NOT TARGET quantized_ops_lib OR NOT TARGET quantized_kernels)
  if(EXECUTORCH_BUILD_KERNELS_QUANTIZED)
    message(
      WARNING
        "quantized_ops_lib target missing even though EXECUTORCH_BUILD_KERNELS_QUANTIZED=ON. Falling back to a local build for backward compatibility."
    )
  endif()
  message(STATUS "arm_executor_runner enabling quantized kernels locally")
  add_subdirectory(
    ${EXECUTORCH_ROOT}/kernels/quantized
    ${CMAKE_BINARY_DIR}/quantized_kernels_baremetal
  )
endif()

if(NOT TARGET cortex_m_ops_lib OR NOT TARGET cortex_m_kernels)
  if(EXECUTORCH_BUILD_CORTEX_M)
    message(
      FATAL_ERROR
        "cortex_m_ops_lib target missing even though EXECUTORCH_BUILD_CORTEX_M=ON. Please regenerate your build directory."
    )
  endif()
  message(STATUS "arm_executor_runner enabling cortex_m backend locally")
  add_subdirectory(
    ${EXECUTORCH_ROOT}/backends/cortex_m
    ${CMAKE_BINARY_DIR}/cortex_m_backend_baremetal
  )
endif()

if(NOT _arm_runner_enabled)
  add_subdirectory(${ETHOS_SDK_PATH}/core_software/core_driver core_driver_stub)
  message(
    STATUS
      "arm_executor_runner: built ethosu_core_driver for delegate builds but skipped the runner because no PTE file/addr or semihosting mode was configured."
  )
  return()
endif()

# Selects timing adapter values matching system_config. Default is
# Ethos_U55_High_End_Embedded, simulating optimal hardware for the
# Corestone-300.
set(SYSTEM_CONFIG
    "Ethos_U55_High_End_Embedded"
    CACHE STRING "System config"
)
set(MEMORY_MODE
    "Shared_Sram"
    CACHE STRING "Vela memory mode"
)

message(STATUS "SYSTEM_CONFIG is ${SYSTEM_CONFIG}")
message(STATUS "MEMORY_MODE is ${MEMORY_MODE}")
message(STATUS "ET_NUM_INFERENCES is ${ET_NUM_INFERENCES}")

# By default, use 2MB of temporary scratch buffer. For Dedicated_Sram, use 64MB
# for the temporary scratch buffer and 384KB for the fast scratch buffer (the
# cache, applicable only for Ethos-U65 and Ethos-U85). Allow -D overrides.
if(MEMORY_MODE MATCHES "Dedicated_Sram")
  if(NOT DEFINED ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x4000000)
  endif()
  if(NOT DEFINED ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set(ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x60000)
  endif()
else()
  if(NOT DEFINED ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x200000)
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

# Dependencies from the Ethos-U Core This is the platform target of
# Corstone-300, that includes ethosu_core_driver and bare-metal bringup
# libraries. We link against ethosu_target_init which includes all of these
# dependencies.
add_corstone_subdirectory(${SYSTEM_CONFIG} ${ETHOS_SDK_PATH})
configure_timing_adapters(${SYSTEM_CONFIG} ${MEMORY_MODE})

# Convert pte to header
if(NOT ${ET_MODEL_PTE_ADDR} AND NOT SEMIHOSTING)
  add_custom_target(
    gen_model_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pte_to_header.py
            --pte ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ET_PTE_FILE_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

# The arm_executor_runner executable
add_executable(arm_executor_runner)

target_sources(
  arm_executor_runner PRIVATE arm_executor_runner.cpp arm_perf_monitor.cpp
                              arm_memory_allocator.cpp
)

set(_arm_runner_required_targets
    executorch
    executorch_delegate_ethos_u
    extension_runner_util
    quantized_ops_lib
    cortex_m_ops_lib
    quantized_kernels
    cortex_m_kernels
    portable_kernels
)
foreach(_runner_target ${_arm_runner_required_targets})
  if(NOT TARGET ${_runner_target})
    message(
      FATAL_ERROR
        "Missing target '${_runner_target}' required by arm_executor_runner. Configure the ExecuTorch build with EXECUTORCH_BUILD_ARM_BAREMETAL=ON (and related options) before building this target."
    )
  endif()
endforeach()

# Check for "U55" in SYSTEM_CONFIG
string(FIND "${SYSTEM_CONFIG}" "U55" U55_FOUND)

# Check for "U85" in SYSTEM_CONFIG
string(FIND "${SYSTEM_CONFIG}" "U85" U85_FOUND)

# Check if neither "U55" nor "U85" was found
if(U55_FOUND EQUAL -1 AND U85_FOUND EQUAL -1)
  message(
    FATAL_ERROR
      "SYSTEM_CONFIG does not contain 'U55' or 'U85'. Configuration aborting."
  )
endif()

# Proceed with specific actions if either is found
if(NOT U55_FOUND EQUAL -1)
  message(STATUS "SYSTEM_CONFIG contains 'U55'.")
  set(LINK_FILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300.ld")
endif()

if(NOT U85_FOUND EQUAL -1)
  message(STATUS "SYSTEM_CONFIG contains 'U85'.")
  set(LINK_FILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-320.ld")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LINK_FILE_EXT ld)
  set(LINK_FILE_OPTION "-T")
  set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
endif()

get_filename_component(LINK_FILE_OUT_BASE ${LINK_FILE} NAME)
set(LINK_FILE_OUT
    ${CMAKE_CURRENT_BINARY_DIR}/${LINK_FILE_OUT_BASE}.${LINK_FILE_EXT}
)

execute_process(
  COMMAND ${CMAKE_C_COMPILER} ${COMPILER_PREPROCESSOR_OPTIONS} -o
          ${LINK_FILE_OUT} ${LINK_FILE_IN}
)
target_link_options(arm_executor_runner PRIVATE "-T" "${LINK_FILE_OUT}")

set(arm_executor_runner_link)
list(
  APPEND
  arm_executor_runner_link
  extension_runner_util
  ethosu_target_init
  executorch
  quantized_ops_lib
  cortex_m_ops_lib
  "-Wl,--whole-archive"
  executorch_delegate_ethos_u
  quantized_kernels
  cortex_m_kernels
  portable_kernels
  "-Wl,--no-whole-archive"
  -Xlinker
  -Map=arm_executor_runner.map
)

# Figure out which ops to include: For semihosting build, use
# (user-set)SELECT_OPS_MODEL variable. For normal build, use
# EXECUTORCH_SELECT_OPS_MODEL to include ops automatically. If the pte contains
# no undelegated ops, use neither.
set(FOUND_OPS_IN_FILE FALSE)
if(NOT SEMIHOSTING
   AND NOT ET_MODEL_PTE_ADDR
   AND NOT "${ET_PTE_FILE_PATH}" STREQUAL ""
   AND EXISTS "${ET_PTE_FILE_PATH}"
)
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} "${EXECUTORCH_ROOT}/codegen/tools/gen_oplist.py"
      --model_file_path=${ET_PTE_FILE_PATH}
      --output_path=${CMAKE_CURRENT_BINARY_DIR}/temp.yaml
    OUTPUT_VARIABLE CMD_RESULT
  )

  if(CMD_RESULT MATCHES "aten::" OR CMD_RESULT MATCHES "dim_order_ops::")
    set(FOUND_OPS_IN_FILE TRUE)
  endif()
endif()

if(SEMIHOSTING)
  set(EXECUTORCH_SELECT_OPS_MODEL "")
  message(
    "gen_oplist: Building with semihosting, no model is used to auto generate ops from will use EXECUTORCH_SELECT_OPS_LIST=${EXECUTORCH_SELECT_OPS_LIST}"
  )
elseif(FOUND_OPS_IN_FILE)
  set(EXECUTORCH_SELECT_OPS_LIST "")
  set(EXECUTORCH_SELECT_OPS_MODEL "${ET_PTE_FILE_PATH}")
  message(
    "gen_oplist:  EXECUTORCH_SELECT_OPS_MODEL=${ET_PTE_FILE_PATH} is used to auto generate ops from"
  )
else()
  set(EXECUTORCH_SELECT_OPS_LIST "")
  set(EXECUTORCH_SELECT_OPS_MODEL "")
  message(
    "gen_oplist: No non delagated ops was found in ${ET_PTE_FILE_PATH} no ops added to build"
  )
endif()

# Ensure that either executorch_select_ops_list or executorch_select_ops_model
# is set - otherwise assume no kernels needs to be registered
if(NOT ("${EXECUTORCH_SELECT_OPS_LIST}" STREQUAL ""
        AND "${EXECUTORCH_SELECT_OPS_MODEL}" STREQUAL "")
)
  gen_selected_ops(
    LIB_NAME
    "arm_portable_ops_lib"
    OPS_SCHEMA_YAML
    ""
    ROOT_OPS
    "${EXECUTORCH_SELECT_OPS_LIST}"
    INCLUDE_ALL_OPS
    ""
    OPS_FROM_MODEL
    "${EXECUTORCH_SELECT_OPS_MODEL}"
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )

  generate_bindings_for_kernels(
    LIB_NAME "arm_portable_ops_lib" FUNCTIONS_YAML
    ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )
  gen_operators_lib(
    LIB_NAME
    "arm_portable_ops_lib"
    KERNEL_LIBS
    portable_kernels
    DEPS
    executorch
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )
  list(APPEND arm_executor_runner_link arm_portable_ops_lib)
endif()

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  target_compile_options(arm_executor_runner PUBLIC -DET_EVENT_TRACER_ENABLED)

  list(APPEND arm_executor_runner_link etdump flatccrt)
endif()

if(ET_BUNDLE_IO)
  list(APPEND arm_executor_runner_link bundled_program)
endif()

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(arm_executor_runner PUBLIC ${arm_executor_runner_link})

target_link_options(
  arm_executor_runner PUBLIC LINKER:-Map=arm_executor_runner.map
)
set(ARM_EXECUTOR_RUNNER_OUTPUT_DIR
    ""
    CACHE PATH "Optional output directory for arm_executor_runner"
)
set(_arm_runner_output_dir "")
if(NOT "${ARM_EXECUTOR_RUNNER_OUTPUT_DIR}" STREQUAL "")
  set(_arm_runner_output_dir "${ARM_EXECUTOR_RUNNER_OUTPUT_DIR}")
elseif(_arm_executor_runner_standalone)
  set(_arm_runner_output_dir "${CMAKE_BINARY_DIR}")
endif()
if(_arm_runner_output_dir)
  set_target_properties(
    arm_executor_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   ${_arm_runner_output_dir}
  )
  if(CMAKE_CONFIGURATION_TYPES)
    foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${_cfg} _cfg_upper)
      set_target_properties(
        arm_executor_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}
                                       ${_arm_runner_output_dir}/${_cfg}
      )
    endforeach()
  endif()
endif()

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
      ${ET_DIR_PATH}/examples/arm/asan ${CMAKE_CURRENT_BINARY_DIR}/asan_runtime
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
  PRIVATE ${EXECUTORCH_ROOT} ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
          ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(
  arm_executor_runner PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)

if(NOT ${ET_MODEL_PTE_ADDR} AND NOT SEMIHOSTING)
  add_dependencies(arm_executor_runner gen_model_header)
endif()

if(ET_MODEL_PTE_ADDR)
  target_compile_definitions(
    arm_executor_runner PUBLIC -DET_MODEL_PTE_ADDR=${ET_MODEL_PTE_ADDR}
  )
endif()

if(ET_NUM_INFERENCES)
  target_compile_definitions(
    arm_executor_runner PUBLIC ET_NUM_INFERENCES=${ET_NUM_INFERENCES}
  )
endif()

if(ET_LOG_DUMP_INPUT)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_LOG_DUMP_INPUT)
endif()

if(ET_LOG_DUMP_OUTPUT)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_LOG_DUMP_OUTPUT)
endif()

# Devtool BundleIO: Use Bundle PTE with input and reference output included to
# check if it matches.

if(ET_BUNDLE_IO)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_BUNDLE_IO)
endif()

if(ET_ATOL)
  target_compile_definitions(arm_executor_runner PUBLIC ET_ATOL=${ET_ATOL})
endif()

if(ET_RTOL)
  target_compile_definitions(arm_executor_runner PUBLIC ET_RTOL=${ET_RTOL})
endif()

# Devtools ETDump: Speed and dumping output

if(ET_DUMP_OUTPUTS)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_DUMP_OUTPUTS)
endif()

if(ET_DUMP_INTERMEDIATE_OUTPUTS)
  target_compile_definitions(
    arm_executor_runner PUBLIC -DET_DUMP_INTERMEDIATE_OUTPUTS
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

# Fixup compilation of retarget.c
if(SEMIHOSTING)
  # Remove this when MLBEDSW-8910 is closed.
  set_source_files_properties(
    ${ETHOS_SDK_PATH}/core_platform/targets/${TARGET_BOARD}/retarget.c
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()
