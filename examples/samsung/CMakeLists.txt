# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(samsung_example)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  message(FATAL_ERROR "ios is not supported by Samsung AI system.")
endif()

# prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)
find_package(gflags REQUIRED)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)

get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)

add_compile_options(-Wall -Werror -fPIC)

message("Build Samsung Android Examples")

link_directories(${EXECUTORCH_ROOT}/build_samsung_android/lib)

add_library(lib_portable_ops STATIC IMPORTED)

set_target_properties(
  lib_portable_ops
  PROPERTIES
    IMPORTED_LOCATION
    "${EXECUTORCH_ROOT}/build_samsung_android/lib/libportable_ops_lib.a"
)

set(__enn_executor_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/executor_runner/enn_executor_runner.cpp
)

add_executable(enn_executor_runner ${__enn_executor_runner__srcs})

add_dependencies(enn_executor_runner enn_backend)

target_include_directories(
  enn_executor_runner PUBLIC ${_common_include_directories}
                             ${EXECUTORCH_ROOT}/..
)

target_link_libraries(
  enn_executor_runner PRIVATE enn_logging enn_backend gflags executorch
                              extension_data_loader portable_ops_lib
)

set_target_properties(
  enn_executor_runner PROPERTIES CXX_VISIBILITY_PRESET hidden
)
