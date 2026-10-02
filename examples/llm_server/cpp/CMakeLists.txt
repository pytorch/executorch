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
if(NOT TARGET executorch)
  find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
  # Older tokenizer exports expose their internal headers as top-level includes,
  # shadowing the POSIX regex.h used by GoogleTest. Use the public include root.
  get_target_property(
    _tokenizer_includes tokenizers::tokenizers INTERFACE_INCLUDE_DIRECTORIES
  )
  list(TRANSFORM _tokenizer_includes REPLACE "/pytorch/tokenizers/?$" "")
  set_target_properties(
    tokenizers::tokenizers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                      "${_tokenizer_includes}"
  )
endif()
find_package(Threads REQUIRED)
executorch_target_link_options_shared_lib(executorch)

# Older installed packages may omit this header-only interface target.
if(NOT TARGET extension_llm_session)
  add_library(extension_llm_session INTERFACE)
  target_link_libraries(extension_llm_session INTERFACE executorch_core)
endif()
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
target_link_libraries(
  test_worker_loop PUBLIC ${test_link_libraries} Threads::Threads
)
add_test(NAME worker_loop COMMAND test_worker_loop)

# New transport is deliberately separate from the legacy worker loop.
add_library(multiplexed_worker STATIC multiplexed_worker.cpp)
target_include_directories(
  multiplexed_worker
  PUBLIC ${_common_include_directories}
  PRIVATE ${_json_include}
)
target_link_libraries(
  multiplexed_worker PUBLIC extension_llm_serving Threads::Threads
)

if(NOT TARGET GTest::gtest_main)
  find_package(GTest CONFIG QUIET)
  if(NOT TARGET GTest::gtest_main)
    add_subdirectory(
      ${EXECUTORCH_ROOT}/third-party/googletest
      ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL
    )
  endif()
endif()
add_executable(test_multiplexed_worker test_multiplexed_worker.cpp)
target_include_directories(test_multiplexed_worker PRIVATE ${_json_include})
target_link_libraries(
  test_multiplexed_worker PRIVATE multiplexed_worker GTest::gtest_main
)
add_test(NAME multiplexed_worker COMMAND test_multiplexed_worker)
