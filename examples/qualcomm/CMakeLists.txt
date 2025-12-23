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

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# some toolchains might contain their own packages, skip them here
find_package(gflags REQUIRED NO_CMAKE_FIND_ROOT_PATH)
find_package(absl REQUIRED NO_CMAKE_FIND_ROOT_PATH)

# Find prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

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

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/..
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)

# build qnn_executor_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/executor_runner)

# build qnn_llama_runner for llama
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/oss_scripts/llama)

# build qnn_mimi_decoder_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/oss_scripts/moshi)

# build qnn_t5_runner for t5
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/oss_scripts/t5)

# build qnn_whisper_runner for whisper
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/oss_scripts/whisper)

# build qaihub_llama2_7b_runner and qaihub_llama3_8b_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qaihub_scripts/llama)

# build qaihub_stable_diffusion_runner
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qaihub_scripts/stable_diffusion)
