# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# qnn_executor_runner: Like executor_runner but with QNN
if(NOT ${ANDROID})
    message(FATAL_ERROR "Not building Android, quitting...")
endif()
cmake_minimum_required(VERSION 3.19)
project(qualcomm_runner_example)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)
# Find prebuilt libraries. executorch package should contain
# portable_ops_lib, etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
)
#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(
  EXECUTORCH_SRCS_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

get_filename_component(EXECUTORCH_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    ABSOLUTE
)
set(_qnn_executor_runner__srcs ${_executor_runner__srcs})
list(
    TRANSFORM
    _qnn_executor_runner__srcs
    PREPEND
    "${EXECUTORCH_SOURCE_DIR}/"
)
list(
    FILTER
    _qnn_executor_runner__srcs
    EXCLUDE REGEX
    ".*executor_runner.cpp$"
)
list(
    PREPEND
    _qnn_executor_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/executor_runner/qnn_executor_runner.cpp
)

add_executable(qnn_executor_runner ${_qnn_executor_runner__srcs})

target_link_libraries(qnn_executor_runner
    ${_executor_runner_libs}
    qnn_executorch_backend
    gflags
)
target_compile_options(qnn_executor_runner
    PUBLIC
    ${_common_compile_options}
)
