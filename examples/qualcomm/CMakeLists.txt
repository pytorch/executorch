# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_STANDARD 17)
# qnn_executor_runner: Like executor_runner but with QNN

cmake_minimum_required(VERSION 3.19)
project(qualcomm_runner_example)

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

# Find prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)
find_package(gflags REQUIRED)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)
extract_sources(${EXECUTORCH_SRCS_FILE})
include(${EXECUTORCH_SRCS_FILE})

get_filename_component(
  EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)

# portable_ops_lib
gen_selected_ops(LIB_NAME "full_portable_ops_lib" INCLUDE_ALL_OPS "ON")
generate_bindings_for_kernels(
  LIB_NAME "full_portable_ops_lib" FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
)
gen_operators_lib(
  LIB_NAME "full_portable_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
)
target_compile_options(
  full_portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED
)
target_include_directories(
  full_portable_ops_lib PUBLIC ${_common_include_directories}
)

# build qnn_executor_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/executor_runner)

# build qnn_llama_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/oss_scripts/llama2)

# build qaihub_llama2_7b_runner and qaihub_llama3_8b_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qaihub_scripts/llama)

# build qaihub_stable_diffusion_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qaihub_scripts/stable_diffusion)
