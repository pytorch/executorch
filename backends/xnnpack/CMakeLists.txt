# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

# NB: Enabling this will serialize execution of delegate instances Keeping this
# OFF by default to maintain existing behavior, to be revisited.
option(EXECUTORCH_XNNPACK_SHARED_WORKSPACE
       "Enable workspace sharing across different delegate instances" ON
)
# Keeping this OFF by default due to regressions in decode and model load with
# kleidi kernels
option(EXECUTORCH_XNNPACK_ENABLE_KLEIDI "Enable Arm Kleidi kernels" OFF)

# Turning this on cache weights between partitions and methods. If weights
# are shared across methods/partitions then this can reduce load time and
# memory usage

# Keeping this off maintains existing behavior. Turning this on serializes
# execution and initialization of delegates, to be revisited
option(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE
        "Enable weights cache to cache and manage all packed weights" OFF)

if(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE)
  add_definitions(-DENABLE_XNNPACK_WEIGHTS_CACHE)
endif()
if(EXECUTORCH_XNNPACK_SHARED_WORKSPACE)
  add_definitions(-DENABLE_XNNPACK_SHARED_WORKSPACE)
endif()
if(EXECUTORCH_XNNPACK_ENABLE_KLEIDI)
  add_definitions(-DENABLE_XNNPACK_KLEIDI)
endif()

set(_common_include_directories ${EXECUTORCH_ROOT}/..)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)

set(_xnnpack_schema__include_dir "${CMAKE_BINARY_DIR}/schema/include")
# Paths to headers generated from the .fbs files.
set(_xnnpack_flatbuffer__outputs)
foreach(fbs_file ${_xnnpack_schema__srcs})
  string(REGEX REPLACE "([^/]+)[.]fbs$" "\\1_generated.h" generated
                       "${fbs_file}"
  )
  list(APPEND _xnnpack_flatbuffer__outputs
       "${_xnnpack_schema__include_dir}/executorch/${generated}"
  )
endforeach()

set(_xnnpack_schema__outputs)
foreach(fbs_file ${_xnnpack_schema__srcs})
  string(REGEX REPLACE "runtime_([^/]+)[.]fbs$" "\\1_generated.h" generated
                       "${fbs_file}"
  )
  list(APPEND _xnnpack_schema__outputs
       "${_xnnpack_schema__include_dir}/executorch/${generated}"
  )
endforeach()

if(WIN32)
  set(MV_COMMAND powershell -Command "Move-Item -Path ${_xnnpack_flatbuffer__outputs} -Destination ${_xnnpack_schema__outputs}")
else()
  set(MV_COMMAND mv ${_xnnpack_flatbuffer__outputs} ${_xnnpack_schema__outputs})
endif()

# Generate the headers from the .fbs files.
add_custom_command(
  OUTPUT ${_xnnpack_schema__outputs}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
    "${_xnnpack_schema__include_dir}/executorch/backends/xnnpack/serialization"
    ${_xnnpack_schema__srcs}
  COMMAND ${MV_COMMAND}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS flatc
  COMMENT "Generating xnnpack_schema headers"
  VERBATIM
)

unset(MV_COMMAND)

add_library(xnnpack_schema INTERFACE ${_xnnpack_schema__outputs})
set_target_properties(xnnpack_schema PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(
  xnnpack_schema INTERFACE ${_xnnpack_schema__include_dir}
                           ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
)

set(xnnpack_third_party pthreadpool extension_threadpool cpuinfo)

include(cmake/Dependencies.cmake)

list(TRANSFORM _xnnpack_backend__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(xnnpack_backend STATIC ${_xnnpack_backend__srcs})
target_link_libraries(
  xnnpack_backend PUBLIC ${xnnpack_third_party} executorch_core xnnpack_schema extension_threadpool
)

target_include_directories(
  xnnpack_backend PUBLIC ${_common_include_directories}
)
target_include_directories(xnnpack_backend PUBLIC ${XNNPACK_INCLUDE_DIR})
target_include_directories(
  xnnpack_backend
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pthreadpool/include
)
target_include_directories(
  xnnpack_backend
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cpuinfo/include
)
target_compile_options(xnnpack_backend PUBLIC ${_common_compile_options})
target_link_options_shared_lib(xnnpack_backend)

if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  list(APPEND xnn_executor_runner_libs optimized_native_cpu_ops_lib)
else()
  list(APPEND xnn_executor_runner_libs portable_ops_lib)
endif()

if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  list(APPEND xnn_executor_runner_libs $<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>)
endif()

if(EXECUTORCH_BUILD_KERNELS_QUANTIZED)
  list(APPEND xnn_executor_runner_libs quantized_ops_lib)
endif()

list(APPEND xnn_executor_runner_libs xnnpack_backend executorch)

# ios can only build library but not binary
if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  #
  # xnn_executor_runner: Like executor_runner but with XNNPACK, the binary will
  # be at ${CMAKE_BINARY_DIR}/backends/xnnpack
  #
  list(TRANSFORM _xnn_executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")
  add_executable(xnn_executor_runner ${_xnn_executor_runner__srcs})

  if(EXECUTORCH_ENABLE_EVENT_TRACER)
    if(EXECUTORCH_BUILD_DEVTOOLS)
      list(APPEND xnn_executor_runner_libs etdump)
    else()
      message(
        SEND_ERROR
          "Use of 'EXECUTORCH_ENABLE_EVENT_TRACER' requires 'EXECUTORCH_BUILD_DEVTOOLS' to be enabled."
      )
    endif()
  endif()

  target_link_libraries(xnn_executor_runner gflags ${xnn_executor_runner_libs})
  target_compile_options(xnn_executor_runner PUBLIC ${_common_compile_options})
  if(EXECUTORCH_BUILD_PTHREADPOOL)
    target_link_libraries(xnn_executor_runner extension_threadpool pthreadpool)
    target_compile_definitions(xnn_executor_runner PRIVATE ET_USE_THREADPOOL)
  endif()
endif()

install(
  TARGETS xnnpack_backend
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
