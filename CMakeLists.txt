# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for runtime components.
#
# ### One-time setup ###
#
# Configure the CMake build system. It's good practice to do this whenever
# cloning or pulling the upstream repo. Once this is done, you don't need to do
# it again until you pull from the upstream repo again.
#
# NOTE: Build options can be configured by passing arguments to cmake. For
# example, to enable the EXECUTORCH_BUILD_XNNPACK option, change the cmake
# command to 'cmake -DEXECUTORCH_BUILD_XNNPACK=ON ..'.
#[[
  (rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake ..)
]]
#
# ### Build ###
#
# NOTE: The `-j` argument specifies how many jobs/processes to use when
# building, and tends to speed up the build significantly. It's typical to use
# "core count + 1" as the `-j` value.
# ~~~
# cmake --build cmake-out -j9
# ~~~
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be checked with a linter via
# ~~~
# cmake-lint CMakeLists.txt
# ~~~
#

# TODO Lower to 3.24 when XNNPACK dependency is updated to include
# https://github.com/google/XNNPACK/commit/c690daa67f883e1b627aadf7684c06797e9a0684
cmake_minimum_required(VERSION 3.29)
project(executorch)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include(${PROJECT_SOURCE_DIR}/tools/cmake/common/preset.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/Codegen.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/Utils.cmake)
include(CMakeDependentOption)
include(ExternalProject)
include(GNUInstallDirs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
announce_configured_options(CMAKE_CXX_STANDARD)

if(NOT CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()
announce_configured_options(CMAKE_SYSTEM_PROCESSOR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
announce_configured_options(CMAKE_BUILD_TYPE)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()
announce_configured_options(PYTHON_EXECUTABLE)

announce_configured_options(CMAKE_CXX_COMPILER_ID)
announce_configured_options(CMAKE_TOOLCHAIN_FILE)
announce_configured_options(BUILD_TESTING)

load_build_preset()
include(${PROJECT_SOURCE_DIR}/tools/cmake/preset/default.cmake)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "ccache found and enabled for faster builds")
else()
  message(STATUS "ccache not found, builds will not be cached")
endif()
announce_configured_options(CCACHE_PROGRAM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setup RPATH. See
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH OFF)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
#
# TODO: Doesn't work for us right now because we are not installing .so's into
# the correct locations. For example we have libcustom_ops_aot_lib.so depending
# on _portable_lib.so, which was eventually put under
# <site-packages>/executorch/extension/pybindings/ but this rpath is not
# automatically added because at build time it seems `portable_lib` is being
# built under the same directory, so no extra rpath is being added. To properly
# fix this we need to install `portable_lib` into the correct path.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
# ------------------------------ OPTIONS -------------------------------------
# WARNING: Please don't add example specific options in this CMakeLists.txt.
# Instead please use `find_package(executorch REQUIRED)` in the example
# directory and add a new executable in the example `CMakeLists.txt`.

if(NOT EXECUTORCH_ENABLE_LOGGING)
  # Avoid pulling in the logging strings, which can be large. Note that this
  # will set the compiler flag for all targets in this directory, and for all
  # subdirectories included after this point.
  add_definitions(-DET_LOG_ENABLED=0)
endif()

add_definitions(-DET_MIN_LOG_LEVEL=${ET_MIN_LOG_LEVEL})

if(NOT EXECUTORCH_ENABLE_PROGRAM_VERIFICATION)
  # Avoid pulling in the flatbuffer data verification logic, which can add about
  # 20kB. Note that this will set the compiler flag for all targets in this
  # directory, and for all subdirectories included after this point.
  add_definitions(-DET_ENABLE_PROGRAM_VERIFICATION=0)
endif()

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  add_definitions(-DET_EVENT_TRACER_ENABLED)
endif()

# -ffunction-sections -fdata-sections: breaks function and data into sections so
# they can be properly gc'd. -s: strip symbol.
if(WIN32)
  set(CMAKE_CXX_FLAGS_RELEASE "/Gy /Gw ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  set(CMAKE_CXX_FLAGS_RELEASE
      "-ffunction-sections -fdata-sections ${CMAKE_CXX_FLAGS_RELEASE}"
  )
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

if(EXECUTORCH_OPTIMIZE_SIZE)
  # -Os: Optimize for size.
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os")
else()
  # -O2: Moderate opt.
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(EXECUTORCH_BUILD_TESTS)
  include(CTest)
else()
  # It looks like some of our third-party deps will try to turn this on if it's
  # not explicitly set, leading to confusing behavior.
  set(BUILD_TESTING OFF)
endif()

add_subdirectory(third-party)

if(NOT DEFINED FXDIV_SOURCE_DIR)
  set(ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG
      ${CMAKE_POSITION_INDEPENDENT_CODE}
  )
  set(FXDIV_SOURCE_DIR "backends/xnnpack/third-party/FXdiv")
  add_subdirectory("${FXDIV_SOURCE_DIR}")
  executorch_move_interface_include_directories_to_build_time_only(fxdiv)
  set(CMAKE_POSITION_INDEPENDENT_CODE
      ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
  )
endif()

if(EXECUTORCH_BUILD_CPUINFO)
  # --- cpuinfo
  set(ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG
      ${CMAKE_POSITION_INDEPENDENT_CODE}
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CPUINFO_SOURCE_DIR
      "${CMAKE_CURRENT_LIST_DIR}/backends/xnnpack/third-party/cpuinfo"
  )
  set(CPUINFO_BUILD_TOOLS
      OFF
      CACHE BOOL ""
  )
  set(CPUINFO_BUILD_UNIT_TESTS
      OFF
      CACHE BOOL ""
  )
  set(CPUINFO_BUILD_MOCK_TESTS
      OFF
      CACHE BOOL ""
  )
  set(CPUINFO_BUILD_BENCHMARKS
      OFF
      CACHE BOOL ""
  )
  set(CPUINFO_LIBRARY_TYPE
      "static"
      CACHE STRING ""
  )
  set(CPUINFO_LOG_LEVEL
      "error"
      CACHE STRING ""
  )
  set(CLOG_SOURCE_DIR "${CPUINFO_SOURCE_DIR}/deps/clog")
  add_subdirectory("${CPUINFO_SOURCE_DIR}")
  set(CMAKE_POSITION_INDEPENDENT_CODE
      ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
  )
  executorch_add_prefix_to_public_headers(cpuinfo "${CPUINFO_SOURCE_DIR}/")
  install(
    TARGETS cpuinfo
    EXPORT ExecuTorchTargets
    DESTINATION lib
    INCLUDES
    DESTINATION ${_common_include_directories}
  )
endif()

if(EXECUTORCH_BUILD_PTHREADPOOL)
  # --- pthreadpool
  set(ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG
      ${CMAKE_POSITION_INDEPENDENT_CODE}
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(PTHREADPOOL_SOURCE_DIR "backends/xnnpack/third-party/pthreadpool")
  set(PTHREADPOOL_BUILD_TESTS
      OFF
      CACHE BOOL ""
  )
  set(PTHREADPOOL_BUILD_BENCHMARKS
      OFF
      CACHE BOOL ""
  )
  set(PTHREADPOOL_LIBRARY_TYPE
      "static"
      CACHE STRING ""
  )
  set(PTHREADPOOL_ALLOW_DEPRECATED_API
      ON
      CACHE BOOL ""
  )
  if(APPLE)
    set(PTHREADPOOL_SYNC_PRIMITIVE
        "condvar"
        CACHE STRING ""
    )
  endif()
  add_subdirectory("${PTHREADPOOL_SOURCE_DIR}")
  executorch_move_interface_include_directories_to_build_time_only(pthreadpool)
  executorch_move_interface_include_directories_to_build_time_only(
    pthreadpool_interface
  )
  install(
    TARGETS pthreadpool pthreadpool_interface fxdiv
    EXPORT ExecuTorchTargets
    DESTINATION lib
    INCLUDES
    DESTINATION ${_common_include_directories}
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE
      ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
  )
endif()

if(EXECUTORCH_BUILD_TESTS)
  set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
  include(CTest)
endif()

# TODO(dbort): Fix these warnings and remove this flag.
set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
# TODO(#6475): This requires/assumes that the repo lives in a directory named
# exactly `executorch`. Check the assumption first. Remove this check once we
# stop relying on the assumption.
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _repo_dir_name)
if(NOT "${_repo_dir_name}" STREQUAL "executorch")
  message(
    FATAL_ERROR
      "The ExecuTorch repo must be cloned into a directory named exactly "
      "`executorch`; found `${_repo_dir_name}`. See "
      "https://github.com/pytorch/executorch/issues/6475 for progress on a "
      "fix for this restriction."
  )
endif()
set(_common_include_directories
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/core/portable_type/c10>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/core/portable_type/c10>
)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
if(EXECUTORCH_SRCS_FILE)
  message(
    WARNING
      "EXECUTORCH_SRCS_FILE is no longer necessary and will not affect the build."
  )
endif()
executorch_load_build_variables()

# Detect if an iOS toolchain is set.
if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  set(CMAKE_TOOLCHAIN_IOS ON)
else()
  set(CMAKE_TOOLCHAIN_IOS OFF)
endif()

# Detect if an Android toolchain is set.
if(CMAKE_TOOLCHAIN_FILE MATCHES ".*android\.toolchain\.cmake$")
  set(CMAKE_TOOLCHAIN_ANDROID ON)
  if(NOT ANDROID_PLATFORM)
    set(ANDROID_PLATFORM android-30)
  endif()
else()
  set(CMAKE_TOOLCHAIN_ANDROID OFF)
endif()

# Add code coverage flags to supported compilers
if(EXECUTORCH_USE_CPP_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    string(APPEND CMAKE_C_FLAGS " --coverage -fprofile-abs-path")
    string(APPEND CMAKE_CXX_FLAGS " --coverage -fprofile-abs-path")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
    string(APPEND CMAKE_CXX_FLAGS
           " -fprofile-instr-generate -fcoverage-mapping"
    )
  else()
    message(
      FATAL_ERROR
        "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported"
    )
  endif()
endif()

#
# program_schema: Generated .h files from schema/*.fbs inputs
#
add_subdirectory(schema)

#
# executorch_core: Minimal runtime library
#
# The bare-minimum runtime library, supporting the Program and Method
# interfaces. Does not contain any operators, including primitive ops. Does not
# contain any backends.
#

# Remove any PAL-definition files from the sources.
list(FILTER _executorch_core__srcs EXCLUDE REGEX
     "runtime/platform/default/[^/]*.cpp$"
)

# Add the source file that maps to the requested default PAL implementation.
list(APPEND _executorch_core__srcs ${EXECUTORCH_PAL_DEFAULT_FILE_PATH})

add_library(executorch_core ${_executorch_core__srcs})

# Legacy name alias.
add_library(executorch_no_prim_ops ALIAS executorch_core)

# A list of all configured backends.
set(_executorch_backends "")

# A list of all configured extensions.
set(_executorch_extensions "")

# A list of all configured kernel libraries.
set(_executorch_kernels "")

target_link_libraries(executorch_core PRIVATE program_schema)
if(ANDROID)
  target_link_libraries(executorch_core PUBLIC log)
endif()
if(EXECUTORCH_USE_DL)
  # Check if dl exists for this toolchain and only then link it.
  find_library(DL_LIBRARY_EXISTS NAMES dl)
  # Check if the library was found
  if(DL_LIBRARY_EXISTS)
    target_link_libraries(executorch_core PRIVATE dl) # For dladdr()
  endif()
endif()
target_include_directories(
  executorch_core PUBLIC ${_common_include_directories}
)
target_compile_definitions(
  executorch_core PUBLIC C10_USING_CUSTOM_GENERATED_MACROS
)
target_compile_options(executorch_core PUBLIC ${_common_compile_options})
if(MAX_KERNEL_NUM)
  target_compile_definitions(
    executorch_core PRIVATE MAX_KERNEL_NUM=${MAX_KERNEL_NUM}
  )
endif()

# Build devtools first if needed - some backends depend on protobuf from
# devtools
if(EXECUTORCH_BUILD_DEVTOOLS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/devtools)
endif()

if(EXECUTORCH_BUILD_PYBIND AND APPLE)
  # shared version
  add_library(executorch_core_shared SHARED ${_executorch_core__srcs})
  target_link_libraries(executorch_core_shared PRIVATE program_schema)
  if(DL_LIBRARY_EXISTS)
    # For dladdr()
    target_link_libraries(executorch_core_shared PRIVATE dl)
  endif()
  target_include_directories(
    executorch_core_shared PUBLIC ${_common_include_directories}
  )
  target_compile_definitions(
    executorch_core_shared PUBLIC C10_USING_CUSTOM_GENERATED_MACROS
  )
  target_compile_options(
    executorch_core_shared PUBLIC ${_common_compile_options}
  )
  if(MAX_KERNEL_NUM)
    target_compile_definitions(
      executorch_core_shared PRIVATE MAX_KERNEL_NUM=${MAX_KERNEL_NUM}
    )
  endif()
endif()

#
# executorch: Primary runtime library with primitive operators.
#
# Provides the Program and Method interfaces, along with primitive operators.
# Does not contain portable kernels or other full operators. Does not contain
# any backends.
#
add_library(executorch ${_executorch__srcs})
target_link_libraries(executorch PRIVATE executorch_core)
target_include_directories(executorch PUBLIC ${_common_include_directories})
target_compile_definitions(executorch PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_compile_options(executorch PUBLIC ${_common_compile_options})
executorch_target_link_options_shared_lib(executorch)

#
# portable_ops_lib: A library to register core ATen ops using portable kernels,
# see kernels/portable/CMakeLists.txt.
#
# Real integrations should supply their own YAML file that only lists the
# operators necessary for the models that will run.
#
if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  # find pytorch lib here to make it available to all sub-directories. Find it
  # before including portable so that optimized_portable_kernels can use it.
  find_package_torch_headers()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/portable/cpu/util)

if(EXECUTORCH_BUILD_PORTABLE_OPS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/portable)
endif()

if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/optimized)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/configurations)

# Install `executorch` library as well as `executorch-config.cmake` under
# ${CMAKE_INSTALL_PREFIX}/
install(
  DIRECTORY runtime/core/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/core
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY runtime/executor/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/executor
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY runtime/kernel/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/kernel
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY runtime/platform/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/platform
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY extension/kernel_util/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/kernel_util
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY extension/tensor/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/tensor
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY extension/threadpool/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/threadpool
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  TARGETS executorch executorch_core
  EXPORT ExecuTorchTargets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES tools/cmake/executorch-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ExecuTorch
)

if(EXECUTORCH_BUILD_ARM_BAREMETAL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/arm)
  list(APPEND _executorch_backends executorch_delegate_ethos_u)
endif()

if(EXECUTORCH_BUILD_CADENCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cadence)
endif()

if(EXECUTORCH_BUILD_NXP_NEUTRON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/nxp)
  list(APPEND _executorch_backends executorch_delegate_neutron)
endif()

if(EXECUTORCH_BUILD_COREML)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/apple/coreml)
  list(APPEND _executorch_backends coremldelegate)
endif()

if(EXECUTORCH_BUILD_MPS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/apple/mps)
  list(APPEND _executorch_backends mpsdelegate)
endif()

if(EXECUTORCH_BUILD_NEURON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/mediatek)
  list(APPEND _executorch_backends neuron_backend)
endif()

if(EXECUTORCH_BUILD_OPENVINO)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/openvino)
  list(APPEND _executorch_backends openvino_backend)
endif()

if(EXECUTORCH_BUILD_QNN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/qualcomm)
  list(APPEND _executorch_backends qnn_executorch_backend)
endif()

if(EXECUTORCH_BUILD_ENN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/samsung)
  list(APPEND _executorch_backends enn_backend)
endif()

if(EXECUTORCH_BUILD_XNNPACK)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/xnnpack)
  list(APPEND _executorch_backends xnnpack_backend)
endif()

if(EXECUTORCH_BUILD_CORTEX_M)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cortex_m)
endif()

if(EXECUTORCH_BUILD_EXTENSION_APPLE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/apple)
endif()

if(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/data_loader)
  install(
    DIRECTORY extension/data_loader/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/data_loader
    FILES_MATCHING
    PATTERN "*.h"
  )
  list(APPEND _executorch_extensions extension_data_loader)
endif()

if(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/evalue_util)
  install(
    DIRECTORY extension/evalue_util/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/evalue_util
    FILES_MATCHING
    PATTERN "*.h"
  )
endif()

if(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/flat_tensor)
  list(APPEND _executorch_extensions extension_flat_tensor)
endif()

if(EXECUTORCH_BUILD_EXTENSION_MODULE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/module)
  install(
    DIRECTORY extension/module/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/module
    FILES_MATCHING
    PATTERN "*.h"
  )
  list(APPEND _executorch_extensions extension_module_static)
endif()

if(EXECUTORCH_BUILD_EXTENSION_LLM)
  if(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER)
    set(SUPPORT_REGEX_LOOKAHEAD ON)
    # llama/runner/CMakeLists.txt builds a shared library libllama_runner.so
    # that transitively depends on tokenizers. Need to build tokenizers with
    # -fPIC.
    set(ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG
        ${CMAKE_POSITION_INDEPENDENT_CODE}
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  endif()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/tokenizers)
  if(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER)
    set(CMAKE_POSITION_INDEPENDENT_CODE
        ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
    )
  endif()
  list(APPEND _executorch_extensions tokenizers)
endif()

if(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/runner_util)
  install(
    DIRECTORY extension/runner_util/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/runner_util
    FILES_MATCHING
    PATTERN "*.h"
  )
  list(APPEND _executorch_extensions extension_runner_util)
endif()

if(EXECUTORCH_BUILD_EXTENSION_TENSOR)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/tensor)
  list(APPEND _executorch_extensions extension_tensor)
endif()

if(EXECUTORCH_BUILD_PTHREADPOOL AND EXECUTORCH_BUILD_CPUINFO)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/threadpool)
endif()

if(EXECUTORCH_BUILD_KERNELS_TORCHAO)
  if(NOT TARGET cpuinfo)
    message(
      FATAL_ERROR
        "EXECUTORCH_BUILD_KERNELS_TORCHAO requires EXECUTORCH_BUILD_CPUINFO be set ON"
    )
  endif()
  if(NOT TARGET pthreadpool)
    message(
      FATAL_ERROR
        "EXECUTORCH_BUILD_KERNELS_TORCHAO requires EXECUTORCH_BUILD_PTHREADPOOL be set ON"
    )
  endif()

  # Configure TorchAO kernels
  set(TORCHAO_BUILD_ATEN_OPS OFF)
  set(TORCHAO_BUILD_EXECUTORCH_OPS ON)
  set(TORCHAO_BUILD_CPU_AARCH64 ON)
  set(TORCHAO_ENABLE_ARM_NEON_DOT ON)
  set(TORCHAO_BUILD_KLEIDIAI ON)

  # TorchAO kernels look for EXECUTORCH_INCLUDE_DIRS
  if(DEFINED EXECUTORCH_INCLUDE_DIRS)
    message(FATAL_ERROR "EXECUTORCH_INCLUDE_DIRS is already defined")
  endif()
  set(EXECUTORCH_INCLUDE_DIRS
      ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
      ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
  )
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/ao/torchao/csrc/cpu)
  unset(EXECUTORCH_INCLUDE_DIRS)

  executorch_target_link_options_shared_lib(torchao_ops_executorch)
  list(APPEND _executorch_kernels torchao_ops_executorch)

  install(
    TARGETS torchao_ops_executorch torchao_kernels_aarch64
    EXPORT ExecuTorchTargets
    DESTINATION lib
    INCLUDES
    DESTINATION ${_common_include_directories}
  )
  # If using KleidiAI and XNNPACK has not installed it already, install it
  if(TORCHAO_BUILD_KLEIDIAI AND NOT (EXECUTORCH_BUILD_XNNPACK
                                     AND EXECUTORCH_XNNPACK_ENABLE_KLEIDI)
  )
    install(
      TARGETS kleidiai
      EXPORT ExecuTorchTargets
      DESTINATION lib
      INCLUDES
      DESTINATION ${_common_include_directories}
    )
  endif()

endif()

if(EXECUTORCH_BUILD_PYBIND)

  if(NOT EXECUTORCH_BUILD_EXTENSION_DATA_LOADER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/data_loader)
  endif()

  if(NOT EXECUTORCH_BUILD_DEVTOOLS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/devtools)
  endif()

  # Add codegen tools subdirectory for selective_build pybind module
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/codegen/tools)

  # Create bundled_module target only for pybindings when bundled_program exists
  # This target has hard dependencies on devtools generated headers
  if(TARGET bundled_program)
    add_library(
      bundled_module STATIC
      ${CMAKE_CURRENT_SOURCE_DIR}/extension/module/bundled_module.cpp
    )

    # Ensure bundled_module waits for bundled_program's generated headers
    add_dependencies(bundled_module bundled_program)

    target_link_libraries(bundled_module PRIVATE extension_data_loader)
    target_link_libraries(
      bundled_module PUBLIC extension_module_static bundled_program
    )

    target_include_directories(
      bundled_module PUBLIC ${_common_include_directories}
    )
    target_compile_options(
      bundled_module PUBLIC -Wno-deprecated-declarations -fPIC
    )
  endif()

  # find pytorch lib, to allow pybind to take at::Tensor as input/output
  find_package_torch()
  find_library(
    TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib"
  )

  set(_dep_libs
      ${TORCH_PYTHON_LIBRARY}
      bundled_program
      etdump
      flatccrt
      executorch
      extension_data_loader
      util
      torch
  )

  if(EXECUTORCH_BUILD_EXTENSION_MODULE)
    # Always use static linking for pybindings to avoid runtime symbol
    # resolution issues
    list(APPEND _dep_libs extension_module_static)
    # Add bundled_module if available
    if(TARGET bundled_module)
      list(APPEND _dep_libs bundled_module)
    endif()
  endif()

  if(EXECUTORCH_BUILD_TESTS)
    list(APPEND _dep_libs test_backend_compiler_lib)
  endif()

  if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
    list(APPEND _dep_libs optimized_native_cpu_ops_lib)
  else()
    list(APPEND _dep_libs portable_ops_lib)
  endif()

  if(EXECUTORCH_BUILD_COREML AND APPLE)
    list(APPEND _dep_libs coremldelegate)
  endif()

  if(EXECUTORCH_BUILD_MPS)
    list(APPEND _dep_libs mpsdelegate)
  endif()

  if(EXECUTORCH_BUILD_OPENVINO)
    list(APPEND _dep_libs openvino_backend)
  endif()

  if(EXECUTORCH_BUILD_QNN)
    list(APPEND _dep_libs qnn_executorch_backend)
  endif()

  if(EXECUTORCH_BUILD_ENN)
    list(APPEND _dep_libs enn_backend)
  endif()

  if(EXECUTORCH_BUILD_XNNPACK)
    # need to explicitly specify XNNPACK and xnnpack-microkernels-prod here
    # otherwise uses XNNPACK and microkernel-prod symbols from libtorch_cpu
    list(APPEND _dep_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  endif()

  if(EXECUTORCH_BUILD_VULKAN)
    list(APPEND _dep_libs vulkan_backend)
  endif()

  # compile options for pybind
  set(_pybind_compile_options -Wno-deprecated-declarations -fPIC -frtti
                              -fexceptions
  )

  # util lib
  add_library(
    util ${CMAKE_CURRENT_SOURCE_DIR}/extension/aten_util/aten_bridge.cpp
  )
  target_include_directories(
    util PUBLIC ${_common_include_directories} ${TORCH_INCLUDE_DIRS}
  )
  target_compile_definitions(util PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)

  target_compile_options(util PUBLIC ${_pybind_compile_options})
  target_link_libraries(util PRIVATE torch c10 executorch extension_tensor)

  # pybind portable_lib
  pybind11_add_module(portable_lib SHARED extension/pybindings/pybindings.cpp)
  # The actual output file needs a leading underscore so it can coexist with
  # portable_lib.py in the same python package.
  set_target_properties(portable_lib PROPERTIES OUTPUT_NAME "_portable_lib")
  target_compile_definitions(
    portable_lib PUBLIC EXECUTORCH_PYTHON_MODULE_NAME=_portable_lib
  )
  target_include_directories(portable_lib PRIVATE ${TORCH_INCLUDE_DIRS})
  target_compile_options(portable_lib PUBLIC ${_pybind_compile_options})
  target_link_libraries(portable_lib PRIVATE ${_dep_libs})

  # Set RPATH to find PyTorch libraries relative to the installation location
  # This goes from executorch/extension/pybindings up to site-packages, then to
  # torch/lib
  if(APPLE)
    set_target_properties(
      portable_lib PROPERTIES BUILD_RPATH "@loader_path/../../../torch/lib"
                              INSTALL_RPATH "@loader_path/../../../torch/lib"
    )
  else()
    set_target_properties(
      portable_lib PROPERTIES BUILD_RPATH "$ORIGIN/../../../torch/lib"
                              INSTALL_RPATH "$ORIGIN/../../../torch/lib"
    )
  endif()

  install(
    TARGETS portable_lib
    EXPORT ExecuTorchTargets
    LIBRARY DESTINATION executorch/extension/pybindings
  )
endif()

if(EXECUTORCH_BUILD_WASM)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/wasm)
endif()

if(EXECUTORCH_BUILD_TOKENIZERS_WASM)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/wasm/tokenizers)
endif()

if(EXECUTORCH_BUILD_EXTENSION_TRAINING)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/training)
  list(APPEND _executorch_extensions extension_training)
endif()

if(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/runner)
  list(APPEND _executorch_extensions extension_llm_runner)
endif()

if(EXECUTORCH_BUILD_EXTENSION_LLM_APPLE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/apple)
endif()

if(EXECUTORCH_BUILD_KERNELS_LLM)
  # TODO: move all custom kernels to ${CMAKE_CURRENT_SOURCE_DIR}/kernels/custom
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/custom_ops)
  list(APPEND _executorch_kernels custom_ops_aot_lib)
endif()

if(EXECUTORCH_BUILD_KERNELS_QUANTIZED)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantized)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  list(APPEND _executorch_kernels quantized_ops_lib)
endif()

if(EXECUTORCH_BUILD_VULKAN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan)
  list(APPEND _executorch_backends vulkan_backend vulkan_schema)
endif()

if(EXECUTORCH_BUILD_VGF)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/arm)
  list(APPEND _executorch_backends vgf_backend)
endif()

# Top-level interface targets.

# A target containing all configured backends.
add_library(executorch_backends INTERFACE)
add_library(executorch::backends ALIAS executorch_backends)
target_link_libraries(executorch_backends INTERFACE ${_executorch_backends})

# A target containing all configured extensions.
add_library(executorch_extensions INTERFACE)
add_library(executorch::extensions ALIAS executorch_extensions)
target_link_libraries(executorch_extensions INTERFACE ${_executorch_extensions})

# A target containing all configured kernels, with selective build, if enabled.
add_library(executorch_kernels INTERFACE)
add_library(executorch::kernels ALIAS executorch_kernels)
if(NOT EXECUTORCH_SELECT_OPS_YAML STREQUAL ""
   OR NOT EXECUTORCH_SELECT_OPS_LIST STREQUAL ""
   OR NOT EXECUTORCH_SELECT_OPS_MODEL STREQUAL ""
)
  # Use optimized kernels when available.
  if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
    set(_selected_kernel_functions_yaml
        ${CMAKE_CURRENT_BINARY_DIR}/configurations/merged.yaml
    )
    set(_selected_kernel_libs "optimized_kernels" "optimized_portable_kernels")
  else()
    set(_selected_kernel_functions_yaml
        ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
    )
    set(_selected_kernel_libs "portable_kernels")
  endif()

  gen_selected_ops(
    LIB_NAME
    "executorch_selected_kernels"
    OPS_SCHEMA_YAML
    "${EXECUTORCH_SELECT_OPS_YAML}"
    ROOT_OPS
    "${EXECUTORCH_SELECT_OPS_LIST}"
    INCLUDE_ALL_OPS
    FALSE
    OPS_FROM_MODEL
    "${EXECUTORCH_SELECT_OPS_MODEL}"
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )

  generate_bindings_for_kernels(
    LIB_NAME
    "executorch_selected_kernels"
    FUNCTIONS_YAML
    ${_selected_kernel_functions_yaml}
    CUSTOM_OPS_YAML
    ""
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )

  gen_operators_lib(
    LIB_NAME
    "executorch_selected_kernels"
    KERNEL_LIBS
    ${_selected_kernel_libs}
    DEPS
    executorch_core
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )
  list(APPEND _executorch_kernels executorch_selected_kernels)

  install(
    TARGETS executorch_selected_kernels
    EXPORT ExecuTorchTargets
    DESTINATION lib
  )
else()
  # No selective build - link the full library.
  if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
    list(APPEND _executorch_kernels optimized_native_cpu_ops_lib)
  else()
    list(APPEND _executorch_kernels portable_ops_lib)
  endif()
endif()
target_link_libraries(executorch_kernels INTERFACE ${_executorch_kernels})

install(TARGETS executorch_backends executorch_extensions executorch_kernels
        EXPORT ExecuTorchTargets
)

if(EXECUTORCH_BUILD_EXECUTOR_RUNNER)
  # Baseline libraries that executor_runner will link against.
  set(_executor_runner_libs executorch extension_evalue_util
                            extension_runner_util gflags executorch_backends
  )

  # Add all extensions
  list(APPEND _executor_runner_libs executorch_extensions)

  if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
    list(APPEND _executor_runner_libs optimized_native_cpu_ops_lib)
  elseif(EXECUTORCH_BUILD_CADENCE)
    list(APPEND _executor_runner_libs cadence_ops_lib)
  else()
    list(APPEND _executor_runner_libs portable_ops_lib)
  endif()

  # Generate lib to register quantized ops
  if(EXECUTORCH_BUILD_KERNELS_QUANTIZED)
    list(APPEND _executor_runner_libs quantized_ops_lib)
  endif()

  if(EXECUTORCH_BUILD_KERNELS_LLM)
    list(APPEND _executor_runner_libs $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>)
  endif()

  if(EXECUTORCH_ENABLE_EVENT_TRACER)
    list(APPEND _executor_runner_libs etdump flatccrt)
  endif()

  add_executable(executor_runner ${_executor_runner__srcs})
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options_gc_sections(executor_runner)
  endif()
  target_link_libraries(executor_runner ${_executor_runner_libs})
  target_compile_options(executor_runner PUBLIC ${_common_compile_options})

  # Automatically set when using `emcmake cmake` for Wasm build.
  if(EMSCRIPTEN)
    # Directory of model pte files to embed in the wasm binary.
    if(NOT DEFINED WASM_MODEL_DIR)
      set(WASM_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/")
    endif()

    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    target_link_options(
      executor_runner PUBLIC -sALLOW_MEMORY_GROWTH --embed-file
      "${WASM_MODEL_DIR}@/"
    )
  endif()
endif()

if(EXECUTORCH_BUILD_ANDROID_JNI)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/android)
endif()

include(Test.cmake)

install(
  EXPORT ExecuTorchTargets
  FILE ExecuTorchTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ExecuTorch
)

# Print all the configs that were called with announce_configured_options.
print_configured_options()
