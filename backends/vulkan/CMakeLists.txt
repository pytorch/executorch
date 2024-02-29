# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

# ATen Vulkan Libs

set(PYTORCH_PATH ${EXECUTORCH_ROOT}/third-party/pytorch)
set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third-party)
include(cmake/ATenVulkan.cmake)

set(common_includes ${vulkan_api_headers} ${EXECUTORCH_ROOT}/..)

# vulkan_graph_lib

file(GLOB_RECURSE vulkan_graph_cpp ${RUNTIME_PATH}/graph/*)

add_library(vulkan_graph_lib STATIC ${vulkan_graph_cpp})

target_include_directories(vulkan_graph_lib PRIVATE ${common_includes})

target_link_libraries(vulkan_graph_lib vulkan_shader_lib)

target_compile_options(vulkan_graph_lib PRIVATE ${VULKAN_CXX_FLAGS})

# Generate Files from flatc

set(schema_include_dir ${CMAKE_BINARY_DIR}/schema/include)

set(generated_header
    ${schema_include_dir}/executorch/backends/vulkan/schema_generated.h)

add_custom_command(
  OUTPUT ${generated_header}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --scoped-enums -o
    "${schema_include_dir}/executorch/backends/vulkan/" ${_vulkan_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  COMMENT "Generating vulkan_schema headers"
  VERBATIM)

# vulkan_schema library

add_library(vulkan_schema INTERFACE ${generated_header})
set_target_properties(vulkan_schema PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
  vulkan_schema INTERFACE ${schema_include_dir}
                          ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)

# vulkan_backend_lib

file(GLOB vulkan_backend_cpp ${RUNTIME_PATH}/*.cpp)

add_library(vulkan_backend ${vulkan_backend_cpp})

target_include_directories(vulkan_backend PRIVATE ${schema_include_dir})
target_include_directories(vulkan_backend PRIVATE ${common_includes})

target_link_libraries(vulkan_backend PRIVATE vulkan_graph_lib)
target_link_libraries(vulkan_backend PRIVATE vulkan_schema)
target_link_libraries(vulkan_backend PRIVATE executorch)

target_compile_options(vulkan_backend PRIVATE ${VULKAN_CXX_FLAGS})

# Test targets

if(EXECUTORCH_BUILD_VULKAN_TESTS)
  # vulkan_compute_api_test
  set(test_cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/vulkan_compute_api_test.cpp)
  add_executable(vulkan_compute_api_test ${test_cpp})
  target_include_directories(vulkan_compute_api_test PRIVATE ${common_includes})
  target_link_libraries(vulkan_compute_api_test vulkan_api_lib)
  target_link_libraries(vulkan_compute_api_test vulkan_graph_lib)
  target_link_libraries(vulkan_compute_api_test gtest_main)
  target_compile_options(vulkan_compute_api_test PRIVATE ${VULKAN_CXX_FLAGS})
endif()
