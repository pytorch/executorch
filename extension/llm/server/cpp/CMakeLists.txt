# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generic model-execution worker for standard .pte TextLLM models. One binary,
# no registry/factory: it constructs TextLLMEngine/TextLLMSession directly and
# speaks the JSONL worker protocol (worker_client.py). Model execution is C++
# only — the Python server is HTTP/control plane.
#
# Build like the example runners (standalone), e.g. from this directory: cmake
# -S . -B <executorch-cmake-out>/extension/llm/server/cpp \
# -DCMAKE_PREFIX_PATH=<executorch-cmake-out> -DEXECUTORCH_BUILD_XNNPACK=ON cmake
# --build <...>/extension/llm/server/cpp --target text_llm_worker

cmake_minimum_required(VERSION 3.24)
project(llm_server_workers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)
# Vendored single-include nlohmann/json for the worker protocol (no new dep).
set(_json_include
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)

# gflags
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../../third-party/gflags)
find_package(gflags REQUIRED)

# executorch
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)

# CPU ops
list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Custom + quantized kernels that export_llm models need, whole-archived so the
# static op registrations survive the linker: llama::custom_sdpa (from
# use_sdpa_with_kv_cache) and quantized_decomposed ops (from quantized exports).
# Without these the model loads but execution fails with "Missing operator".
if(TARGET custom_ops)
  executorch_target_link_options_shared_lib(custom_ops)
  list(APPEND link_libraries custom_ops)
endif()
if(TARGET quantized_ops_lib)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
endif()

# Extensions (Engine/Session lives in extension_llm_runner)
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# XNNPACK: the standard CPU backend for normal .pte TextLLM models.
list(APPEND link_libraries xnnpack_backend)
executorch_target_link_options_shared_lib(xnnpack_backend)

# Tokenizer
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(text_llm_worker text_llm_worker.cpp)
target_include_directories(
  text_llm_worker PUBLIC ${_common_include_directories} ${_json_include}
)
target_link_libraries(text_llm_worker PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(text_llm_worker)
  target_link_options(text_llm_worker PRIVATE "LINKER:-s")
endif()

# Pure unit test for the warm-resume prefill planner (worker_prefill_plan.h). No
# ET/model/tokenizer dependency, so it builds and runs standalone via ctest.
enable_testing()
add_executable(test_worker_prefill_plan test_worker_prefill_plan.cpp)
target_include_directories(
  test_worker_prefill_plan PUBLIC ${_common_include_directories}
)
add_test(NAME worker_prefill_plan COMMAND test_worker_prefill_plan)

# Worker-loop harness (worker_handle_request + WorkerSessions) driven by a
# scriptable fake LLMSession/Tokenizer/LLMEngine -- no model/GPU. It includes
# the full worker_loop.h, so it needs the JSON include + the runtime/tokenizer
# libs.
add_executable(test_worker_loop test_worker_loop.cpp)
target_include_directories(
  test_worker_loop PUBLIC ${_common_include_directories} ${_json_include}
)
target_link_libraries(test_worker_loop PUBLIC ${link_libraries})
add_test(NAME worker_loop COMMAND test_worker_loop)
