# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True -i CMakeLists.txt
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

# Include this file to access target_link_options_shared_lib This is required to
# provide access to target_link_options_shared_lib which allows libraries to be
# linked with the --whole-archive flag. This is required for libraries that
# perform dynamic registration via static initialization.
include(${EXECUTORCH_ROOT}/build/Utils.cmake)

include(cmake/ShaderLibrary.cmake)

# Third party libs

set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third-party)

set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

# Compile settings

set(VULKAN_CXX_FLAGS "")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_WRAPPER")
list(APPEND VULKAN_CXX_FLAGS "-DUSE_VULKAN_VOLK")

# Vulkan Compute API

file(GLOB vulkan_api_cpp ${RUNTIME_PATH}/api/*.cpp)
list(APPEND vulkan_api_cpp ${VOLK_PATH}/volk.c)

set(COMMON_INCLUDES ${EXECUTORCH_ROOT}/.. ${VULKAN_HEADERS_PATH} ${VOLK_PATH}
                    ${VMA_PATH})

# vulkan_graph_lib

file(GLOB_RECURSE vulkan_graph_cpp ${RUNTIME_PATH}/graph/*)
list(APPEND vulkan_graph_cpp ${vulkan_api_cpp})

add_library(vulkan_graph_lib STATIC ${vulkan_graph_cpp})
target_include_directories(vulkan_graph_lib PRIVATE ${COMMON_INCLUDES})
target_compile_options(vulkan_graph_lib PRIVATE ${VULKAN_CXX_FLAGS})
# Link this library with --whole-archive due to dynamic operator registrations
target_link_options_shared_lib(vulkan_graph_lib)

# vulkan_standard_shaders

set(VULKAN_GRAPH_SHADERS_PATH ${RUNTIME_PATH}/graph/ops/glsl/)
# Declares a vulkan_standard_shaders library containing compiled SPIR-V shaders
# from the above path
vulkan_shader_library(${VULKAN_GRAPH_SHADERS_PATH} vulkan_standard_shaders)
# preserve the path of the generated cpp file
set(vulkan_standard_shaders_cpp
    ${CMAKE_BINARY_DIR}/vulkan_standard_shaders/spv.cpp)

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

# vulkan_backend

file(GLOB vulkan_backend_cpp ${RUNTIME_PATH}/*.cpp)
list(APPEND vulkan_backend_cpp ${vulkan_graph_cpp})
list(APPEND vulkan_backend_cpp ${vulkan_standard_shaders_cpp})

add_library(vulkan_backend STATIC ${vulkan_backend_cpp})
target_include_directories(vulkan_backend PRIVATE ${SCHEMA_INCLUDE_DIR}
                                                  ${COMMON_INCLUDES})
target_link_libraries(vulkan_backend PRIVATE vulkan_schema executorch)
target_compile_options(vulkan_backend PRIVATE ${VULKAN_CXX_FLAGS})
# Link this library with --whole-archive due to dynamic backend registration
target_link_options_shared_lib(vulkan_backend)

# Executor Runner

if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*iOS\.cmake$")
  set(VULKAN_RUNNER_SRCS ${_executor_runner__srcs})
  list(TRANSFORM VULKAN_RUNNER_SRCS PREPEND "${EXECUTORCH_ROOT}/")

  add_executable(vulkan_executor_runner ${VULKAN_RUNNER_SRCS})
  target_link_libraries(vulkan_executor_runner ${_executor_runner_libs}
                        vulkan_schema vulkan_backend)
  target_compile_options(vulkan_executor_runner PUBLIC ${VULKAN_CXX_FLAGS})

  add_library(vulkan_executor_runner_lib STATIC ${VULKAN_RUNNER_SRCS})
  target_link_libraries(vulkan_executor_runner_lib ${_executor_runner_libs}
                        vulkan_schema vulkan_backend)
  target_compile_options(vulkan_executor_runner_lib PUBLIC ${VULKAN_CXX_FLAGS})
endif()

# Test targets

if(EXECUTORCH_BUILD_GTESTS)
  set(TEST_UTILS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/test/utils)
  file(GLOB TEST_UTILS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/test/utils/*.cpp)

  set(TEST_SHADERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl)
  vulkan_shader_library(${TEST_SHADERS_PATH} test_shaderlib)

  # vulkan_compute_api_test
  set(COMPUTE_API_TEST_CPP
      ${CMAKE_CURRENT_SOURCE_DIR}/test/vulkan_compute_api_test.cpp)

  add_executable(vulkan_compute_api_test ${COMPUTE_API_TEST_CPP}
                                         ${TEST_UTILS_CPP})
  target_include_directories(vulkan_compute_api_test
                             PRIVATE ${COMMON_INCLUDES} ${TEST_UTILS_HEADERS})
  target_link_libraries(
    vulkan_compute_api_test PRIVATE gtest_main vulkan_graph_lib
                                    vulkan_standard_shaders test_shaderlib)
  target_compile_options(vulkan_compute_api_test PRIVATE ${VULKAN_CXX_FLAGS})
endif()

install(
  TARGETS vulkan_backend
  DESTINATION lib
  INCLUDES
  DESTINATION ${COMMON_INCLUDES})
