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

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()
# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

# Find prebuilt libraries. executorch package should contain
# portable_ops_lib, etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
find_package(gflags REQUIRED)
find_library(android_log log)
set(qnn_lib_list
    qnn_backend
    qnn_backend_cache
    qnn_context
    qnn_device
    qnn_executorch_backend
    qnn_executorch_logging
    qnn_factory
    qnn_graph
    qnn_implementation
    qnn_logger
    qnn_manager
    qnn_sys_implementation
    utils
    wrappers
    ${CMAKE_DL_LIBS}
    ${android_log}
)
foreach(lib ${qnn_lib_list})
    set(lib_var "LIB_${lib}")
    find_library(${lib_var} ${lib} CMAKE_FIND_ROOT_PATH_BOTH)
    if(NOT ${lib_var})
        message("${lib} library is not found")
    else()
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(
            ${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}"
        )
        target_include_directories(${lib} INTERFACE ${_root})
    endif()
endforeach()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
set(EXECUTORCH_SRCS_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)
extract_sources(${EXECUTORCH_SRCS_FILE})
include(${EXECUTORCH_SRCS_FILE})

get_filename_component(EXECUTORCH_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    ABSOLUTE
)
set(_qnn_executor_runner__srcs ${_executor_runner__srcs})

# portable_ops_lib
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
gen_selected_ops("" "" "ON")
generate_bindings_for_kernels(
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml ""
)
gen_operators_lib("portable_ops_lib" portable_kernels executorch)
target_include_directories(portable_ops_lib
    PUBLIC
    ${_common_include_directories}
)

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

target_include_directories(qnn_executor_runner
    PUBLIC
    ${_common_include_directories}
)
target_link_libraries(qnn_executor_runner
    ${qnn_lib_list}
    portable_ops_lib
    gflags
)
target_compile_options(qnn_executor_runner
    PUBLIC
    ${_common_compile_options}
)
