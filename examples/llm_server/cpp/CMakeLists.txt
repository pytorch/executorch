# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared worker-loop tests for the example LLM server. Model-specific workers
# live under examples/models/* and include worker_loop.h.

cmake_minimum_required(VERSION 3.24)
project(llm_server_workers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)
# Vendored single-include nlohmann/json for the worker protocol (no new dep).
set(_json_include
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)

# executorch
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(test_link_libraries executorch extension_llm_session tokenizers::tokenizers)

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
target_link_libraries(test_worker_loop PUBLIC ${test_link_libraries})
add_test(NAME worker_loop COMMAND test_worker_loop)
