# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

cmake_minimum_required(VERSION 3.19)
project(openvino_runner_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/../../../build/executorch_srcs.cmake")
extract_sources(${EXECUTORCH_SRCS_FILE})
include(${EXECUTORCH_SRCS_FILE})

set(_openvino_executor_runner__srcs 
    ${CMAKE_CURRENT_LIST_DIR}/../openvino/executor_runner/openvino_executor_runner.cpp
)

find_package(executorch CONFIG REQUIRED)
include_directories(${EXECUTORCH_INCLUDE_DIRS})

# Portable Ops Library
gen_selected_ops(LIB_NAME "openvino_portable_ops_lib" INCLUDE_ALL_OPS "ON")
generate_bindings_for_kernels(
  LIB_NAME "openvino_portable_ops_lib" FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/backends/openvino/openvino_functions.yaml
)
gen_operators_lib(
  LIB_NAME "openvino_portable_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
)
target_compile_options(openvino_portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED)
target_include_directories(openvino_portable_ops_lib PUBLIC ${_common_include_directories})

# Build Executor Runner
add_executable(openvino_executor_runner ${_openvino_executor_runner__srcs})

target_include_directories(
  openvino_executor_runner PUBLIC ${_common_include_directories} ${EXECUTORCH_ROOT}/cmake-openvino-out/third-party/gflags/include
)

# Set Library Directory
set(LIBRARY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../cmake-openvino-out/lib/;${CMAKE_CURRENT_LIST_DIR}/../../cmake-openvino-out/third-party/gflags")
message(STATUS "Library directory path: ${LIBRARY_DIR}")

# Locate OpenVINO Backend Library
find_library(OPENVINO_BACKEND_LIB NAMES openvino_backend PATHS ${LIBRARY_DIR} NO_DEFAULT_PATH)
if(NOT OPENVINO_BACKEND_LIB)
    message(FATAL_ERROR "OpenVINO backend library not found in ${LIBRARY_DIR}")
endif()

# Locate OpenVINO Backend Library
find_library(GFLAGS_LIB NAMES gflags_nothreads PATHS ${LIBRARY_DIR} NO_DEFAULT_PATH)
if(NOT GFLAGS_LIB)
    message(FATAL_ERROR "Gflags library not found in ${LIBRARY_DIR}")
endif()

# Link Libraries
target_link_libraries(openvino_executor_runner PRIVATE
    ${OPENVINO_BACKEND_LIB}
    ${GFLAGS_LIB}
    executorch
    executorch_core
    openvino_portable_ops_lib
    extension_data_loader
    extension_runner_util
    pthreadpool
)

# Ensure Proper RPATH Handling
set_target_properties(openvino_executor_runner PROPERTIES INSTALL_RPATH "$ORIGIN")

get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)


