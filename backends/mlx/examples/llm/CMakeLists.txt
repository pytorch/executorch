# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# C++ runner for off-graph-cache HF models (export_llm_hf --use-offgraph-cache).
# Unlike run_llm_hf.py (pybindings), this binds the cache via cache_key, so it
# is the run path for off-graph .pte files.

cmake_minimum_required(VERSION 3.24)
project(mlx_run_llm_hf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
executorch_get_build_include_dir(
  "${EXECUTORCH_ROOT}" EXECUTORCH_SOURCE_INCLUDE_DIR
)

set(_common_include_directories ${EXECUTORCH_SOURCE_INCLUDE_DIR})
set(_json_include
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)
# MLXExecutor.h reaches schema_generated.h, which needs flatbuffers. The MLX
# backend exposes that include under BUILD_INTERFACE, which does not reach a
# separate find_package() project like this one.
set(_flatbuffers_include ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

# As in examples/models/llama: point gflags_DIR at the copy the executorch build
# configured, since a separate find_package() project cannot see it.
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../third-party/gflags)
find_package(gflags REQUIRED)

set(link_libraries
    executorch
    extension_module
    extension_tensor
    extension_llm_cache
    extension_llm_runner
    extension_llm_sampler
    gflags
)

if(NOT TARGET mlxdelegate)
  message(FATAL_ERROR "mlx_run_llm_hf requires the MLX backend (mlxdelegate)")
endif()
list(APPEND link_libraries mlxdelegate mlx)
executorch_target_link_options_shared_lib(mlxdelegate)

# CPU kernels for the ops that stay outside the delegate (e.g. the cache
# bookkeeping copy_ the HF export wrappers emit).
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib)
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
endif()

list(APPEND link_libraries tokenizers::tokenizers)

add_executable(mlx_run_llm_hf run_llm_hf.cpp)
target_include_directories(
  mlx_run_llm_hf PUBLIC ${_common_include_directories} ${_json_include}
                        ${_flatbuffers_include}
)
target_link_libraries(mlx_run_llm_hf PUBLIC ${link_libraries})

# The copy helper is gated on EXECUTORCH_BUILD_MLX, which the installed config
# does not set; reaching here means mlxdelegate exists.
set(EXECUTORCH_BUILD_MLX ON)
executorch_target_copy_mlx_metallib(mlx_run_llm_hf)
