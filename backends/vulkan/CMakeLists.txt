# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_VULKAN is ON

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT RUNTIME_PATH)
  set(RUNTIME_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

# Include this file to access target_link_options_shared_lib
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# ATen Vulkan Libs

set(PYTORCH_PATH ${EXECUTORCH_ROOT}/third-party/pytorch)
set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third-party)
include(cmake/ATenVulkan.cmake)

set(COMMON_INCLUDES ${VULKAN_API_HEADERS} ${EXECUTORCH_ROOT}/..)

# vulkan_graph_lib

file(GLOB_RECURSE vulkan_graph_cpp ${RUNTIME_PATH}/graph/*)

add_library(vulkan_graph_lib STATIC ${vulkan_graph_cpp})

target_include_directories(vulkan_graph_lib PRIVATE ${COMMON_INCLUDES})

target_link_libraries(vulkan_graph_lib vulkan_shader_lib)

target_compile_options(vulkan_graph_lib PRIVATE ${VULKAN_CXX_FLAGS})

# Generate Files from flatc

set(SCHEMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/schema/include)

set(GENERATED_HEADER
    ${SCHEMA_INCLUDE_DIR}/executorch/backends/vulkan/schema_generated.h)

add_custom_command(
  OUTPUT ${GENERATED_HEADER}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
    "${SCHEMA_INCLUDE_DIR}/executorch/backends/vulkan/" ${_vulkan_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  COMMENT "Generating vulkan_schema headers"
  VERBATIM)

# vulkan_schema library

add_library(vulkan_schema INTERFACE ${GENERATED_HEADER})
set_target_properties(vulkan_schema PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
  vulkan_schema INTERFACE ${SCHEMA_INCLUDE_DIR}
                          ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)

# vulkan_backend_lib

file(GLOB vulkan_backend_cpp ${RUNTIME_PATH}/*.cpp)

add_library(vulkan_backend ${vulkan_backend_cpp})

target_include_directories(vulkan_backend PRIVATE ${SCHEMA_INCLUDE_DIR})
target_include_directories(vulkan_backend PRIVATE ${COMMON_INCLUDES})

target_link_libraries(vulkan_backend PRIVATE vulkan_graph_lib)
target_link_libraries(vulkan_backend PRIVATE vulkan_schema)
target_link_libraries(vulkan_backend PRIVATE executorch)

target_compile_options(vulkan_backend PRIVATE ${VULKAN_CXX_FLAGS})

# This is required to ensure that vulkan_backend gets linked with
# --whole-archive since backends are registered via static variables that would
# otherwise be discarded
target_link_options_shared_lib(vulkan_backend)

# Executor Runner

if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*iOS\.cmake$")
  set(VULKAN_RUNNER_SRCS ${_executor_runner__srcs})
  list(TRANSFORM VULKAN_RUNNER_SRCS PREPEND "${EXECUTORCH_ROOT}/")
  add_executable(vulkan_executor_runner ${VULKAN_RUNNER_SRCS})
  target_link_libraries(vulkan_executor_runner ${_executor_runner_libs})
  target_link_libraries(vulkan_executor_runner vulkan_schema)
  target_link_libraries(vulkan_executor_runner vulkan_backend)
  target_compile_options(vulkan_executor_runner PUBLIC ${VULKAN_CXX_FLAGS})

  add_library(vulkan_executor_runner_lib STATIC ${VULKAN_RUNNER_SRCS})
  target_link_libraries(vulkan_executor_runner_lib ${_executor_runner_libs})
  target_link_libraries(vulkan_executor_runner_lib vulkan_schema)
  target_link_libraries(vulkan_executor_runner_lib vulkan_backend)
  target_compile_options(vulkan_executor_runner_lib PUBLIC ${VULKAN_CXX_FLAGS})
endif()

# Test targets

if(EXECUTORCH_BUILD_GTESTS)
  # vulkan_compute_api_test
  set(TEST_CPP ${CMAKE_CURRENT_SOURCE_DIR}/test/vulkan_compute_api_test.cpp)
  add_executable(vulkan_compute_api_test ${TEST_CPP})
  target_include_directories(vulkan_compute_api_test PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(vulkan_compute_api_test vulkan_api_lib)
  target_link_libraries(vulkan_compute_api_test vulkan_graph_lib)
  target_link_libraries(vulkan_compute_api_test gtest_main)
  target_compile_options(vulkan_compute_api_test PRIVATE ${VULKAN_CXX_FLAGS})
endif()
