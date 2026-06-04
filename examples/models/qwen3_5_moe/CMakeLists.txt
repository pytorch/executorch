# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(qwen3_5_moe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# gflags
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

# executorch
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)

# CPU ops
list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Extensions
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Backend selection
if(EXECUTORCH_BUILD_METAL)
  list(APPEND link_libraries metal_backend)
  executorch_target_link_options_shared_lib(metal_backend)
elseif(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
  add_compile_definitions(EXECUTORCH_BUILD_CUDA)
else()
  message(
    FATAL_ERROR "Set EXECUTORCH_BUILD_CUDA=ON or EXECUTORCH_BUILD_METAL=ON"
  )
endif()

# Tokenizer
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(qwen3_5_moe_runner main.cpp qwen35_moe_engine.cpp)
target_include_directories(
  qwen3_5_moe_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(qwen3_5_moe_runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(qwen3_5_moe_runner)
  target_link_options(qwen3_5_moe_runner PRIVATE "LINKER:-s")
endif()

# Example-local serving module (_qwen35_moe) used by the worker subprocess
# (worker.py). Off by default; enable with -DEXECUTORCH_BUILD_PYBIND=ON. It
# builds a Qwen35MoEEngine via create_engine() and re-exposes the generic Engine
# / Session surface (no Qwen-specific Python class). The generic
# extension/llm/runner pybind is not touched, so the model-agnostic
# server/runner never depend on Qwen.
if(EXECUTORCH_BUILD_PYBIND)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/third-party/pybind11
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11
  )
  pybind11_add_module(
    _qwen35_moe SHARED qwen35_moe_pybindings.cpp qwen35_moe_engine.cpp
  )
  target_include_directories(_qwen35_moe PRIVATE ${_common_include_directories})
  target_link_libraries(_qwen35_moe PRIVATE ${link_libraries})
  set_target_properties(
    _qwen35_moe PROPERTIES CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON
  )
endif()
