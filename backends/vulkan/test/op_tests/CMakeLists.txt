# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_VULKAN is ON

cmake_minimum_required(VERSION 3.19)
project(executorch)

find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)
find_package(GTest CONFIG REQUIRED)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

# Include this file to access target_link_options_shared_lib This is required to
# provide access to target_link_options_shared_lib which allows libraries to be
# linked with the --whole-archive flag. This is required for libraries that
# perform dynamic registration via static initialization.
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

get_torch_base_path(TORCH_BASE_PATH)
message(STATUS "torch base path: ${TORCH_BASE_PATH}")

# Only build tests if Vulkan was compiled
find_library(LIB_VULKAN_BACKEND vulkan_backend)
find_library(LIB_TORCH torch ${TORCH_BASE_PATH}/lib)
find_library(LIB_TORCH_CPU torch_cpu ${TORCH_BASE_PATH}/lib)
find_library(LIB_C10 c10 ${TORCH_BASE_PATH}/lib)

message(STATUS "Vulkan backend lib ${LIB_VULKAN_BACKEND}")
message(STATUS "Torch ${LIB_TORCH}")

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# Third party include paths

set(VULKAN_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../third-party)

set(GTEST_INCLUDE_PATH
    ${EXECUTORCH_ROOT}/third-party/googletest/googletest/include
)
set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

set(COMMON_INCLUDES
    ${EXECUTORCH_ROOT}/..
    ${VULKAN_HEADERS_PATH}
    ${VOLK_PATH}
    ${VMA_PATH}
    ${GTEST_INCLUDE_PATH}
    ${TORCH_BASE_PATH}/include
    ${TORCH_BASE_PATH}/include/torch/csrc/api/include
)

target_link_options_shared_lib(vulkan_backend)

function(vulkan_op_test test_name test_src)
  set(extra_deps ${ARGN})

  add_executable(${test_name} ${test_src})
  target_include_directories(${test_name} PRIVATE ${COMMON_INCLUDES})
  target_link_libraries(
    ${test_name}
    PRIVATE GTest::gtest_main
            vulkan_backend
            executorch
            ${LIB_TORCH}
            ${LIB_TORCH_CPU}
            ${LIB_C10}
            ${extra_deps}
  )

  add_test(${test_name} ${test_name})
endfunction()

if(LIB_VULKAN_BACKEND AND LIB_TORCH)
  find_library(
    CUSTOM_OPS_LIB custom_ops_aot_lib
    HINTS ${CMAKE_INSTALL_PREFIX}/executorch/extension/llm/custom_ops
  )
  if(CUSTOM_OPS_LIB)
    vulkan_op_test(
      vulkan_sdpa_test ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_test.cpp
      ${CUSTOM_OPS_LIB}
    )
  else()
    message(
      STATUS "Skip building sdpa_test because custom_ops_aot_lib is not found"
    )
  endif()
  vulkan_op_test(
    vulkan_rope_test ${CMAKE_CURRENT_SOURCE_DIR}/rotary_embedding_test.cpp
  )
  vulkan_op_test(
    vulkan_linear_weight_int4_test
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_weight_int4_test.cpp
  )

  # Only build generated op tests if a path to tags.yaml and
  # native_functions.yaml is provided. These files are required for codegen.
  if(TORCH_OPS_YAML_PATH)
    set(GENERATED_VULKAN_TESTS_CPP_PATH ${CMAKE_CURRENT_BINARY_DIR}/vk_gen_cpp)

    # Generated operator correctness tests

    set(generated_test_cpp ${GENERATED_VULKAN_TESTS_CPP_PATH}/op_tests.cpp)

    add_custom_command(
      COMMENT "Generating Vulkan operator correctness tests"
      OUTPUT ${generated_test_cpp}
      COMMAND
        ${PYTHON_EXECUTABLE}
        ${EXECUTORCH_ROOT}/backends/vulkan/test/op_tests/generate_op_correctness_tests.py
        -o ${GENERATED_VULKAN_TESTS_CPP_PATH} --tags-path
        ${TORCH_OPS_YAML_PATH}/tags.yaml --aten-yaml-path
        ${TORCH_OPS_YAML_PATH}/native_functions.yaml
      DEPENDS ${EXECUTORCH_ROOT}/backends/vulkan/test/op_tests/**/*.py
    )

    vulkan_op_test(vulkan_op_correctness_tests ${generated_test_cpp})

    # Generated operator benchmarks (only built in google benchmark is
    # installed)
    find_package(benchmark CONFIG)

    if(benchmark_FOUND)
      set(generated_benchmark_cpp
          ${GENERATED_VULKAN_TESTS_CPP_PATH}/op_benchmarks.cpp
      )

      add_custom_command(
        COMMENT "Generating Vulkan operator benchmarks"
        OUTPUT ${generated_benchmark_cpp}
        COMMAND
          ${PYTHON_EXECUTABLE}
          ${EXECUTORCH_ROOT}/backends/vulkan/test/op_tests/generate_op_benchmarks.py
          -o ${GENERATED_VULKAN_TESTS_CPP_PATH} --tags-path
          ${TORCH_OPS_YAML_PATH}/tags.yaml --aten-yaml-path
          ${TORCH_OPS_YAML_PATH}/native_functions.yaml
        DEPENDS ${EXECUTORCH_ROOT}/backends/vulkan/test/op_tests/**/*.py
      )

      vulkan_op_test(vulkan_op_benchmarks ${generated_benchmark_cpp})
    endif()
  else()
    message(
      STATUS
        "Skipping generated operator correctness tests and benchmarks. Please specify TORCH_OPS_YAML_PATH to build these tests."
    )
  endif()
endif()
