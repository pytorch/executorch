# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(llm_sampler_cuda_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)
FetchContent_MakeAvailable(googletest)

# Get EXECUTORCH_ROOT
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

# Find installed ExecuTorch
find_package(executorch CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX})

# List of CUDA test files
set(LLM_SAMPLER_CUDA_TESTS test_argmax)

enable_testing()

foreach(test_name ${LLM_SAMPLER_CUDA_TESTS})
  add_executable(${test_name} ${test_name}.cu)

  target_include_directories(
    ${test_name} PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                         ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${test_name}
    PRIVATE GTest::gtest
            GTest::gtest_main
            executorch_core
            CUDA::cudart
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

