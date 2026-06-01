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
if(NOT DEFINED ET_INCLUDE_PATH)
  set(ET_INCLUDE_PATH
      "${EXECUTORCH_ROOT}"
      CACHE
        PATH
        "Kept for backward compatibility; include root for ExecuTorch headers"
  )
endif()

if(NOT EXISTS "${EXECUTORCH_ROOT}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project."
  )
endif()

if(NOT COMMAND executorch_target_link_options_shared_lib)
  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
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
include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmEthosUSDK.cmake)
include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmRunnerUtils.cmake)

arm_runner_require_baremetal_targets()

# Keep the default scratch location aligned with the scratch tree used by
# setup.sh/run.sh so developers who just ran those scripts do not need extra
# CMake flags.
set(ETHOS_SDK_PATH
    "${EXECUTORCH_ROOT}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)

arm_ethos_u_default_fetch("${ETHOS_SDK_PATH}" _fetch_ethos_u_default)
option(FETCH_ETHOS_U_CONTENT
       "Fetch ethos_u dependencies instead of relying on pre-downloads"
       ${_fetch_ethos_u_default}
)
arm_ensure_ethos_u_content(
  "${ETHOS_SDK_PATH}" "${EXECUTORCH_ROOT}" ${FETCH_ETHOS_U_CONTENT}
)

set(ET_PTE_FILE_PATH
    ""
    CACHE PATH "Path to ExecuTorch model pte"
)

if(NOT ET_MODEL_PTE_ADDR
   AND "${ET_PTE_FILE_PATH}" STREQUAL ""
   AND NOT SEMIHOSTING
)
  message(
    FATAL_ERROR
      "You must specify one of "
      "ET_MODEL_PTE_ADDR - .pte (or .bpte) on address, "
      "ET_PTE_FILE_PATH - .pte (or .bpte) built into the binary or "
      "SEMIHOSTING - pte and input via host filesystem (use by pytest)"
  )
endif()

if(NOT SEMIHOSTING
   AND NOT ET_MODEL_PTE_ADDR
   AND NOT "${ET_PTE_FILE_PATH}" STREQUAL ""
)
  if(NOT EXISTS "${ET_PTE_FILE_PATH}")
    message(
      FATAL_ERROR
        "ET_PTE_FILE_PATH is set to ${ET_PTE_FILE_PATH}, but no file was found. Generate the model first or point ET_PTE_FILE_PATH at an existing .pte/.bpte."
    )
  endif()
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

if(NOT CMAKE_SKIP_INSTALL_RULES AND TARGET ethosu_core_driver)
  get_property(
    _et_ethosu_core_driver_exported GLOBAL
    PROPERTY ET_ETHOSU_CORE_DRIVER_EXPORTED
  )
  if(NOT _et_ethosu_core_driver_exported)
    install(
      TARGETS ethosu_core_driver
      EXPORT ExecuTorchTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    set_property(GLOBAL PROPERTY ET_ETHOSU_CORE_DRIVER_EXPORTED TRUE)
  endif()
endif()

# Convert pte to header
if(NOT "${ET_MODEL_PTE_ADDR}" AND NOT SEMIHOSTING)
  add_custom_target(
    gen_model_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pte_to_header.py
            --pte ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ET_PTE_FILE_PATH}
  )
endif()

# The arm_executor_runner executable
add_executable(arm_executor_runner)

target_sources(
  arm_executor_runner PRIVATE arm_executor_runner.cpp arm_perf_monitor.cpp
                              arm_memory_allocator.cpp
)

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
  set(LINK_FILE_IN "${CMAKE_CURRENT_LIST_DIR}/Corstone-300.ld")
endif()

if(NOT U85_FOUND EQUAL -1)
  message(STATUS "SYSTEM_CONFIG contains 'U85'.")
  set(LINK_FILE_IN "${CMAKE_CURRENT_LIST_DIR}/Corstone-320.ld")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LINK_FILE_EXT ld)
  set(LINK_FILE_OPTION "-T")
  set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
endif()

get_filename_component(LINK_FILE_OUT_BASE "${LINK_FILE}" NAME)
set(LINK_FILE_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/${LINK_FILE_OUT_BASE}.${LINK_FILE_EXT}"
)

execute_process(
  COMMAND ${CMAKE_C_COMPILER} ${COMPILER_PREPROCESSOR_OPTIONS} -o
          ${LINK_FILE_OUT} ${LINK_FILE_IN}
)
target_link_options(arm_executor_runner PRIVATE "-T" "${LINK_FILE_OUT}")

# Apply whole-archive helper where available to keep static initializers.
executorch_target_link_options_shared_lib(executorch_delegate_ethos_u)
executorch_target_link_options_shared_lib(quantized_kernels)
executorch_target_link_options_shared_lib(cortex_m_kernels)
executorch_target_link_options_shared_lib(portable_kernels)

set(arm_executor_runner_link)
list(
  APPEND
  arm_executor_runner_link
  extension_runner_util
  ethosu_target_init
  executorch
  quantized_ops_lib
  cortex_m_ops_lib
  executorch_delegate_ethos_u
  quantized_kernels
  cortex_m_kernels
  portable_kernels
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
  if(TARGET bundled_program)
    list(APPEND arm_executor_runner_link bundled_program)
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
      list(APPEND arm_executor_runner_link ${_bundled_program_library})
    else()
      message(
        FATAL_ERROR
          "ET_BUNDLE_IO enabled but bundled_program is unavailable. Either configure this build with EXECUTORCH_BUILD_DEVTOOLS=ON so the target exists or set BUNDLED_PROGRAM_LIBRARY_DIR/ET_BUILD_DIR_PATH to a build directory that contains libbundled_program.a."
      )
    endif()
  endif()
endif()

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(arm_executor_runner PUBLIC ${arm_executor_runner_link})

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
set(_arm_runner_include_dirs
    ${ET_INCLUDE_PATH} ${ET_INCLUDE_PATH}/runtime/core/portable_type/c10
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(
  arm_executor_runner PRIVATE ${_arm_runner_include_dirs}
)
target_compile_definitions(
  arm_executor_runner PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)

if(NOT "${ET_MODEL_PTE_ADDR}" AND NOT SEMIHOSTING)
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
