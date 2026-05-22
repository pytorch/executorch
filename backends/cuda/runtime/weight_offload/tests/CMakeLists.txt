# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(weight_offload_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)
FetchContent_MakeAvailable(googletest)

# Get EXECUTORCH_ROOT (this dir is backends/cuda/runtime/weight_offload/tests,
# five levels below the repo root -- same depth as shims/tests).
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

# Find installed ExecuTorch
find_package(executorch CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX})

enable_testing()

# ---------------------------------------------------------------------------
# Host-level: payload parser. No CUDA, no GPU. Builds + runs anywhere.
# ---------------------------------------------------------------------------
add_executable(payload_test payload_test.cpp)
target_include_directories(
  payload_test PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
)
target_link_libraries(
  payload_test PRIVATE GTest::gtest GTest::gtest_main executorch_core
)
add_test(NAME payload_test COMMAND payload_test)

# ---------------------------------------------------------------------------
# Session state machine. Requires a GPU; only built when CUDA is present.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_executable(session_test session_test.cpp)
  target_include_directories(
    session_test PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                         ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_compile_definitions(session_test PRIVATE CUDA_AVAILABLE=1)
  target_link_libraries(
    session_test
    PRIVATE GTest::gtest GTest::gtest_main aoti_cuda_backend aoti_cuda_shims
            executorch_core CUDA::cudart
  )
  add_test(NAME session_test COMMAND session_test)
else()
  message(STATUS "weight_offload_tests: no CUDA compiler; skipping session_test")
endif()
