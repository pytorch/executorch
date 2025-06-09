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

cmake_minimum_required(VERSION 3.24)
project(executorch)

# MARK: - Start EXECUTORCH_H12025_BUILD_MIGRATION

include(${PROJECT_SOURCE_DIR}/tools/cmake/common/preset.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/Utils.cmake)
include(CMakeDependentOption)
include(ExternalProject)

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

if(NOT BUCK2)
  resolve_buck2()
endif()
announce_configured_options(BUCK2)

announce_configured_options(CMAKE_CXX_COMPILER_ID)
announce_configured_options(CMAKE_TOOLCHAIN_FILE)

load_build_preset()
include(${PROJECT_SOURCE_DIR}/tools/cmake/preset/default.cmake)

# Print all the configs that were called with announce_configured_options.
print_configured_options()

# MARK: - End EXECUTORCH_H12025_BUILD_MIGRATION

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setup RPATH. See
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH OFF)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
# TODO: Doesn't work for us right now because we are
# not installing .so's into the correct locations. For example we have
# libcustom_ops_aot_lib.so depending on _portable_lib.so, which was eventually
# put under <site-packages>/executorch/extension/pybindings/ but this rpath is
# not automatically added because at build time it seems `portable_lib` is being
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
set(CMAKE_CXX_FLAGS_RELEASE
    "-ffunction-sections -fdata-sections ${CMAKE_CXX_FLAGS_RELEASE}"
)
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
endif()

add_subdirectory(third-party)

if(NOT DEFINED FXDIV_SOURCE_DIR)
  set(ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG
      ${CMAKE_POSITION_INDEPENDENT_CODE}
  )
  set(FXDIV_SOURCE_DIR "backends/xnnpack/third-party/FXdiv")
  add_subdirectory("${FXDIV_SOURCE_DIR}")
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
  set(CPUINFO_SOURCE_DIR "backends/xnnpack/third-party/cpuinfo")
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
  set(CMAKE_POSITION_INDEPENDENT_CODE
      ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
  )
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
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/core/portable_type/c10
)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#

if(NOT EXECUTORCH_SRCS_FILE)
  # A file wasn't provided. Run a script to extract the source lists from the
  # buck2 build system and write them to a file we can include.
  #
  # NOTE: This will only happen once during cmake setup, so it will not re-run
  # if the buck2 targets change.
  message(STATUS "executorch: Generating source lists")
  set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/executorch_srcs.cmake")
  extract_sources(${EXECUTORCH_SRCS_FILE})
endif()

# This file defines the `_<target>__srcs` variables used below.
message(STATUS "executorch: Using sources file ${EXECUTORCH_SRCS_FILE}")
include(${EXECUTORCH_SRCS_FILE})

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
    message(FATAL_ERROR
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
target_link_options_shared_lib(executorch)

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

#
# gflags: Commandline flag host library.
#

if(EXECUTORCH_BUILD_GFLAGS)
  add_subdirectory(third-party/gflags)
endif()

# Install `executorch` library as well as `executorch-config.cmake` under
# ${CMAKE_INSTALL_PREFIX}/
install(
  DIRECTORY runtime/core/
  DESTINATION include/executorch/runtime/core
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY runtime/kernel/
  DESTINATION include/executorch/runtime/kernel
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY runtime/platform/
  DESTINATION include/executorch/runtime/platform
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY extension/kernel_util/
  DESTINATION include/executorch/extension/kernel_util
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY extension/tensor/
  DESTINATION include/executorch/extension/tensor
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY extension/threadpool/
  DESTINATION include/executorch/extension/threadpool
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  TARGETS executorch executorch_core
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(FILES tools/cmake/executorch-config.cmake
        DESTINATION lib/cmake/ExecuTorch
)

if(EXECUTORCH_BUILD_ARM_BAREMETAL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/arm)
endif()

if(EXECUTORCH_BUILD_CADENCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cadence)
endif()

if(EXECUTORCH_BUILD_COREML)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/apple/coreml)
endif()

if(EXECUTORCH_BUILD_MPS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/apple/mps)
endif()

if(EXECUTORCH_BUILD_NEURON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/mediatek)
endif()

if(EXECUTORCH_BUILD_OPENVINO)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/openvino)
endif()

if(EXECUTORCH_BUILD_QNN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/qualcomm)
endif()

if(EXECUTORCH_BUILD_XNNPACK)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/xnnpack)
endif()

if(EXECUTORCH_BUILD_CORTEX_M)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cortex_m)
endif()

if(EXECUTORCH_BUILD_DEVTOOLS)
  if(NOT EXECUTORCH_BUILD_ARM_BAREMETAL)
    set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER
        ON
        CACHE BOOL "EXECUTORCH_BUILD_EXTENSION_DATA_LOADER" FORCE
    )
  else()
    set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER
        OFF
        CACHE BOOL "EXECUTORCH_BUILD_EXTENSION_DATA_LOADER" FORCE
    )
  endif()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/devtools)
endif()

if(EXECUTORCH_BUILD_EXTENSION_APPLE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/apple)
endif()

if(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/data_loader)
endif()

if(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/flat_tensor)
endif()

if(EXECUTORCH_BUILD_EXTENSION_MODULE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/module)
endif()

if(EXECUTORCH_BUILD_EXTENSION_LLM)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/tokenizers)
endif()

if(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/runner)
endif()

if(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/runner_util)
endif()

if(EXECUTORCH_BUILD_EXTENSION_TENSOR)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/tensor)
endif()

if(EXECUTORCH_BUILD_PTHREADPOOL AND EXECUTORCH_BUILD_CPUINFO)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/threadpool)
endif()

if(EXECUTORCH_BUILD_PYBIND)
  if(NOT EXECUTORCH_BUILD_EXTENSION_DATA_LOADER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/data_loader)
  endif()

  if(NOT EXECUTORCH_BUILD_DEVTOOLS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/devtools)
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

  if(EXECUTORCH_BUILD_XNNPACK)
    # need to explicitly specify XNNPACK and microkernels-prod here otherwise
    # uses XNNPACK and microkernel-prod symbols from libtorch_cpu
    list(APPEND _dep_libs xnnpack_backend XNNPACK microkernels-prod)
  endif()

  # compile options for pybind
  set(_pybind_compile_options -Wno-deprecated-declarations -fPIC -frtti
                              -fexceptions
  )

  # util lib
  add_library(
    util ${CMAKE_CURRENT_SOURCE_DIR}/extension/evalue_util/print_evalue.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/extension/aten_util/aten_bridge.cpp
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

  install(TARGETS portable_lib
          LIBRARY DESTINATION executorch/extension/pybindings
  )
endif()

if(EXECUTORCH_BUILD_EXTENSION_TRAINING)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/training)
endif()

if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  # TODO: move all custom kernels to ${CMAKE_CURRENT_SOURCE_DIR}/kernels/custom
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/llm/custom_ops)
endif()

if(EXECUTORCH_BUILD_KERNELS_QUANTIZED)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantized)
  target_link_options_shared_lib(quantized_ops_lib)
endif()

if(EXECUTORCH_BUILD_EXECUTOR_RUNNER)
  # Baseline libraries that executor_runner will link against.
  set(_executor_runner_libs executorch gflags)

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

  if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
    list(APPEND _executor_runner_libs $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>)
  endif()

  if(EXECUTORCH_BUILD_XNNPACK)
    list(APPEND _executor_runner_libs xnnpack_backend)
  endif()

  if(EXECUTORCH_ENABLE_EVENT_TRACER)
    list(APPEND _executor_runner_libs etdump flatccrt)
  endif()

  if(EXECUTORCH_BUILD_COREML AND APPLE)
    list(APPEND _executor_runner_libs coremldelegate)
  endif()

  add_executable(executor_runner ${_executor_runner__srcs})
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE)
      target_link_options(executor_runner PRIVATE "LINKER:-dead_strip")
    else()
      target_link_options(executor_runner PRIVATE "LINKER:--gc-sections")
    endif()
  endif()
  target_link_libraries(executor_runner ${_executor_runner_libs})
  target_compile_options(executor_runner PUBLIC ${_common_compile_options})
endif()

if(EXECUTORCH_BUILD_VULKAN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan)
endif()

include(Test.cmake)
