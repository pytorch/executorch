# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(executorch)
include(build/Utils.cmake)
include(CMakeDependentOption)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Setup RPATH.
# See https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH OFF)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
# TODO: Doesn't work for us right now because we are not installing .so's into the
# correct locations. For example we have libcustom_ops_aot_lib.so depending on
# _portable_lib.so, which was eventually put under <site-packages>/executorch/extension/pybindings/
# but this rpath is not automatically added because at build time it seems `portable_lib`
# is being built under the same directory, so no extra rpath is being added. To
# properly fix this we need to install `portable_lib` into the correct path.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
# ------------------------------ OPTIONS -------------------------------------
# WARNING: Please don't add example specific options in this CMakeLists.txt.
# Instead please use `find_package(executorch REQUIRED)` in the example
# directory and add a new executable in the example `CMakeLists.txt`.

# _default_release_disabled_options: default value for options that should be
# disabled in Release mode by default. Users can still manually enable them,
# though.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(_default_release_disabled_options OFF)
else()
  set(_default_release_disabled_options ON)
endif()

# Let users override which PAL defaults to use.
#
# TODO(dbort): Add another option that lets users point to a specific source
# file; if set, would override the default option.
set(EXECUTORCH_PAL_DEFAULT
    "posix"
    CACHE STRING
          "Which PAL default implementation to use: one of {posix, minimal}"
)

option(EXECUTORCH_ENABLE_LOGGING "Build with ET_LOG_ENABLED"
       ${_default_release_disabled_options}
)
if(NOT EXECUTORCH_ENABLE_LOGGING)
  # Avoid pulling in the logging strings, which can be large. Note that this
  # will set the compiler flag for all targets in this directory, and for all
  # subdirectories included after this point.
  add_definitions(-DET_LOG_ENABLED=0)
endif()

# Configure log level. Must be one of debug, info, error, fatal.
set(EXECUTORCH_LOG_LEVEL
    "Info"
    CACHE STRING "Build with the given ET_MIN_LOG_LEVEL value"
)
string(TOLOWER "${EXECUTORCH_LOG_LEVEL}" LOG_LEVEL_LOWER)
if(LOG_LEVEL_LOWER STREQUAL "debug")
  add_definitions(-DET_MIN_LOG_LEVEL=Debug)
elseif(LOG_LEVEL_LOWER STREQUAL "info")
  add_definitions(-DET_MIN_LOG_LEVEL=Info)
elseif(LOG_LEVEL_LOWER STREQUAL "error")
  add_definitions(-DET_MIN_LOG_LEVEL=Error)
elseif(LOG_LEVEL_LOWER STREQUAL "fatal")
  add_definitions(-DET_MIN_LOG_LEVEL=Fatal)
else()
  message(
    SEND_ERROR
      "Unknown log level \"${EXECUTORCH_LOG_LEVEL}\". Expected one of Debug, "
      + "Info, Error, or Fatal."
  )
endif()

option(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION
       "Build with ET_ENABLE_PROGRAM_VERIFICATION"
       ${_default_release_disabled_options}
)
if(NOT EXECUTORCH_ENABLE_PROGRAM_VERIFICATION)
  # Avoid pulling in the flatbuffer data verification logic, which can add about
  # 20kB. Note that this will set the compiler flag for all targets in this
  # directory, and for all subdirectories included after this point.
  add_definitions(-DET_ENABLE_PROGRAM_VERIFICATION=0)
endif()

option(EXECUTORCH_ENABLE_EVENT_TRACER "Build with ET_EVENT_TRACER_ENABLED=ON"
       OFF
)
if(EXECUTORCH_ENABLE_EVENT_TRACER)
  add_definitions(-DET_EVENT_TRACER_ENABLED)
endif()

# -ffunction-sections -fdata-sections: breaks function and data into sections so
# they can be properly gc'd. -s: strip symbol. -fno-exceptions -fno-rtti:
# disables exceptions and runtime type.
set(CMAKE_CXX_FLAGS_RELEASE
    "-ffunction-sections -fdata-sections -fno-exceptions -fno-rtti"
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

option(OPTIMIZE_SIZE "Build executorch runtime optimizing for binary size" OFF)
if(OPTIMIZE_SIZE)
  # -Os: Optimize for size
  set(CMAKE_CXX_FLAGS_RELEASE "-Os ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  # -O2: Moderate opt.
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

option(EXECUTORCH_BUILD_ANDROID_JNI "Build Android JNI" OFF)

option(EXECUTORCH_BUILD_ARM_BAREMETAL
       "Build the Arm Baremetal flow for Cortex-M and Ethos-U" OFF
)

option(EXECUTORCH_BUILD_COREML "Build the Core ML backend" OFF)

option(EXECUTORCH_BUILD_KERNELS_CUSTOM "Build the custom kernels" OFF)

option(EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT "Build the custom ops lib for AOT"
       OFF
)

option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "Build the Data Loader extension"
       OFF
)

option(EXECUTORCH_BUILD_EXTENSION_MODULE "Build the Module extension" OFF)

option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL "Build the Runner Util extension"
       OFF
)

option(EXECUTORCH_BUILD_EXTENSION_TENSOR "Build the Tensor extension" OFF)

option(EXECUTORCH_BUILD_EXTENSION_TRAINING "Build the training extension" OFF)

option(EXECUTORCH_BUILD_MPS "Build the MPS backend" OFF)

option(EXECUTORCH_BUILD_NEURON "Build the backends/mediatek directory" OFF)

option(EXECUTORCH_BUILD_PYBIND "Build the Python Bindings" OFF)

option(EXECUTORCH_BUILD_QNN "Build the Qualcomm backend" OFF)

option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build the optimized kernels" OFF)

option(EXECUTORCH_BUILD_KERNELS_QUANTIZED "Build the quantized kernels" OFF)

option(EXECUTORCH_BUILD_DEVTOOLS "Build the ExecuTorch Developer Tools")

option(EXECUTORCH_BUILD_TESTS "Build CMake-based unit tests" OFF)

option(EXECUTORCH_NNLIB_OPT "Build Cadence backend Hifi nnlib kernel" OFF)

option(EXECUTORCH_CADENCE_CPU_RUNNER "Build Cadence backend CPU runner" OFF)

option(EXECUTORCH_BUILD_SIZE_TEST "Build the size test" OFF)

option(EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" OFF)

option(EXECUTORCH_BUILD_VULKAN "Build the Vulkan backend" OFF)

option(BUILD_EXECUTORCH_PORTABLE_OPS "Build portable_ops library" ON)

option(EXECUTORCH_USE_DL "Use libdl library" ON)

option(EXECUTORCH_BUILD_CADENCE "Build the Cadence DSP backend" OFF)

#
# pthreadpool: build pthreadpool library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)

#
# cpuinfo: build cpuinfo library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)

if(EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT)
  set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
  set(EXECUTORCH_BUILD_KERNELS_CUSTOM ON)
endif()

if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
endif()

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

if(EXECUTORCH_BUILD_TESTS)
  include(CTest)
endif()

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()
message(STATUS "Using python executable '${PYTHON_EXECUTABLE}'")

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
set(_common_include_directories ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/runtime/core/portable_type)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#

if(NOT EXECUTORCH_SRCS_FILE)
  # Find or download buck2 binary.
  resolve_buck2()

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

#
# Modify default options when cross-compiling.
#
# The intent is for the EXECUTORCH_BUILD_HOST_TARGETS option to affect the
# default ON/OFF values of host targets around the tree. This way, a user can
# disable EXECUTORCH_BUILD_HOST_TARGETS to disable all host targets, and then
# optionally re-enable some of those targets. Or they could leave
# EXECUTORCH_BUILD_HOST_TARGETS enabled and then optionally disable any given
# host target.
#
# We can then use various cross-compilation hints to set the default value of
# EXECUTORCH_BUILD_HOST_TARGETS, which can still be overridden if desired.
#

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
    message(ERROR
            "Code coverage for compiler ${CMAKE_CXX_COMPILER_ID} is unsupported"
    )
  endif()
endif()

# EXECUTORCH_BUILD_HOST_TARGETS: Option to control the building of host-only
# tools like `flatc`, along with example executables like `executor_runner` and
# libraries that it uses, like `gflags`. Disabling this can be helpful when
# cross-compiling, but some required tools that would have been built need to be
# provided directly (via, for example, FLATC_EXECUTABLE).
cmake_dependent_option(
  EXECUTORCH_BUILD_HOST_TARGETS "Build host-only targets." ON
  "NOT CMAKE_TOOLCHAIN_IOS" OFF
)

#
# flatc: Flatbuffer commandline tool to generate .h files from .fbs files
#
cmake_dependent_option(
  EXECUTORCH_BUILD_FLATC "Build the flatc executable." ON
  "NOT FLATC_EXECUTABLE;EXECUTORCH_BUILD_HOST_TARGETS" OFF
)

if(EXECUTORCH_BUILD_FLATC)
  if(FLATC_EXECUTABLE)
    # We could ignore this, but it could lead to confusion about which `flatc`
    # is actually being used.
    message(
      FATAL_ERROR "May not set both EXECUTORCH_BUILD_FLATC and FLATC_EXECUTABLE"
    )
  endif()
  set(FLATC_EXECUTABLE flatc)
  set(FLATBUFFERS_BUILD_FLATC
      ON
      CACHE BOOL ""
  )
  set(FLATBUFFERS_BUILD_FLATHASH
      OFF
      CACHE BOOL ""
  )
  set(FLATBUFFERS_BUILD_FLATLIB
      OFF
      CACHE BOOL ""
  )
  set(FLATBUFFERS_BUILD_TESTS
      OFF
      CACHE BOOL ""
  )
  set(FLATBUFFERS_INSTALL
      OFF
      CACHE BOOL ""
  )
  add_subdirectory(third-party/flatbuffers)

  # exir lets users set the alignment of tensor data embedded in the flatbuffer,
  # and some users need an alignment larger than the default, which is typically
  # 32.
  target_compile_definitions(flatc PRIVATE FLATBUFFERS_MAX_ALIGNMENT=1024)
endif()
if(NOT FLATC_EXECUTABLE)
  message(
    FATAL_ERROR
      "FLATC_EXECUTABLE must be set when EXECUTORCH_BUILD_FLATC is disabled. "
      "Note that EXECUTORCH_BUILD_FLATC may be disabled implicitly when "
      "cross-compiling or when EXECUTORCH_BUILD_HOST_TARGETS is disabled."
  )
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
if(EXECUTORCH_PAL_DEFAULT MATCHES "^(posix|minimal)$")
  message(STATUS "executorch: Using PAL default '${EXECUTORCH_PAL_DEFAULT}'")
  list(APPEND _executorch_core__srcs
       "runtime/platform/default/${EXECUTORCH_PAL_DEFAULT}.cpp"
  )
else()
  message(
    FATAL_ERROR "Unknown EXECUTORCH_PAL_DEFAULT \"${EXECUTORCH_PAL_DEFAULT}\". "
                "Expected one of {posix, minimal}."
  )
endif()

add_library(executorch_core ${_executorch_core__srcs})

# Legacy name alias.
add_library(executorch_no_prim_ops ALIAS executorch_core)

target_link_libraries(executorch_core PRIVATE program_schema)
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
target_compile_definitions(executorch_core PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_compile_options(executorch_core PUBLIC ${_common_compile_options})
if(MAX_KERNEL_NUM)
  target_compile_definitions(
    executorch_core PRIVATE MAX_KERNEL_NUM=${MAX_KERNEL_NUM}
  )
endif()

if(EXECUTORCH_BUILD_PYBIND AND APPLE)
  # shared version
  add_library(
    executorch_core_shared SHARED ${_executorch_core__srcs}
  )
  target_link_libraries(executorch_core_shared PRIVATE program_schema)
  if(DL_LIBRARY_EXISTS)
    # For dladdr()
    target_link_libraries(executorch_core_shared PRIVATE dl)
  endif()
  target_include_directories(
    executorch_core_shared PUBLIC ${_common_include_directories}
  )
  target_compile_definitions(executorch_core_shared PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
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
if(BUILD_EXECUTORCH_PORTABLE_OPS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/portable)
endif()

if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/optimized)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/configurations)

#
# gflags: Commandline flag host library.
#
cmake_dependent_option(
  EXECUTORCH_BUILD_GFLAGS "Build the gflags library." ON
  EXECUTORCH_BUILD_HOST_TARGETS OFF
)
if(EXECUTORCH_BUILD_GFLAGS)
  add_subdirectory(third-party/gflags)
endif()

# Install `executorch` library as well as `executorch-config.cmake` under
# ${CMAKE_INSTALL_PREFIX}/
install(DIRECTORY runtime/core/  DESTINATION include/executorch/runtime/core FILES_MATCHING PATTERN "*.h")
install(DIRECTORY runtime/kernel/  DESTINATION include/executorch/runtime/kernel FILES_MATCHING PATTERN "*.h")
install(DIRECTORY runtime/platform/  DESTINATION include/executorch/runtime/platform FILES_MATCHING PATTERN "*.h")
install(DIRECTORY extension/kernel_util/  DESTINATION include/executorch/extension/kernel_util FILES_MATCHING PATTERN "*.h")
install(DIRECTORY extension/tensor/  DESTINATION include/executorch/extension/tensor FILES_MATCHING PATTERN "*.h")
install(DIRECTORY extension/threadpool/  DESTINATION include/executorch/extension/threadpool FILES_MATCHING PATTERN "*.h")
install(
  TARGETS executorch executorch_core
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
install(FILES build/executorch-config.cmake DESTINATION lib/cmake/ExecuTorch)

#
# executor_runner: Host tool that demonstrates program execution.
#
cmake_dependent_option(
  EXECUTORCH_BUILD_EXECUTOR_RUNNER "Build the executor_runner executable" ON
  EXECUTORCH_BUILD_HOST_TARGETS OFF
)

# Add googletest if any test targets should be built
if(BUILD_TESTING)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/googletest)
endif()

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

if(EXECUTORCH_BUILD_QNN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/qualcomm)
endif()

if(EXECUTORCH_BUILD_XNNPACK)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/xnnpack)
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

if(EXECUTORCH_BUILD_EXTENSION_MODULE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/module)
endif()

if(EXECUTORCH_BUILD_EXTENSION_TRAINING)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/training)
endif()

if(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/runner_util)
endif()

if(EXECUTORCH_BUILD_EXTENSION_TENSOR)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/tensor)
endif()

if(EXECUTORCH_BUILD_PTHREADPOOL
   AND EXECUTORCH_BUILD_CPUINFO
   AND CMAKE_CXX_STANDARD GREATER_EQUAL 14
)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/threadpool)
endif()

if(EXECUTORCH_BUILD_PYBIND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pybind11)

  if(NOT EXECUTORCH_BUILD_EXTENSION_DATA_LOADER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extension/data_loader)
  endif()

  if(NOT EXECUTORCH_BUILD_DEVTOOLS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/devtools)
  endif()

  # find pytorch lib, to allow pybind to take at::Tensor as input/output
  find_package(Torch CONFIG REQUIRED)
  find_library(
    TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib"
  )

  set(_dep_libs
      ${TORCH_PYTHON_LIBRARY}
      bundled_program
      etdump
      executorch
      extension_data_loader
      util
      torch
  )

  if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
    list(APPEND _dep_libs optimized_native_cpu_ops_lib)
  else()
    list(APPEND _dep_libs portable_ops_lib)
  endif()

  if(EXECUTORCH_BUILD_COREML)
    list(APPEND _dep_libs coremldelegate)
  endif()

  if(EXECUTORCH_BUILD_MPS)
    list(APPEND _dep_libs mpsdelegate)
  endif()

  if(EXECUTORCH_BUILD_XNNPACK)
    # need to explicitly specify XNNPACK and microkernels-prod
    # here otherwise uses XNNPACK and microkernel-prod symbols from libtorch_cpu
    list(APPEND _dep_libs xnnpack_backend XNNPACK microkernels-prod)
  endif()

  # compile options for pybind
  set(_pybind_compile_options
      -Wno-deprecated-declarations
      -fPIC
      -frtti
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

# Print all summary
executorch_print_configuration_summary()
