# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(aoti_cuda_shim_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

# Find installed ExecuTorch
find_package(executorch CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX})

# List of test files
set(CUDA_SHIM_TESTS
    test_aoti_torch_create_tensor_from_blob_v2
    test_aoti_torch_empty_strided
    test_aoti_torch_delete_tensor_object
    test_aoti_torch__reinterpret_tensor
    test_aoti_torch_copy_
    test_aoti_torch_cuda_rand
    test_aoti_torch_new_tensor_handle
    test_aoti_torch_item_bool
    test_aoti_torch_assign_tensors_out
)

# CUDA-specific tests requiring GPU kernels
set(CUDA_KERNEL_TESTS
    test_aoti_torch_cuda__weight_int4pack_mm test_aoti_torch_cuda_int4_plain_mm
    test_aoti_torch_cuda_int5_plain_mm test_aoti_torch_cuda_int6_plain_mm
)

enable_testing()

foreach(test_name ${CUDA_SHIM_TESTS})
  add_executable(${test_name} ${test_name}.cpp)

  target_include_directories(
    ${test_name} PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                         ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_compile_definitions(${test_name} PRIVATE CUDA_AVAILABLE=1)

  target_link_libraries(
    ${test_name} PRIVATE GTest::gtest GTest::gtest_main aoti_cuda_shims
                         executorch_core CUDA::cudart
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

add_executable(benchmark_int4_plain_mm benchmark_int4_plain_mm.cu)
target_include_directories(
  benchmark_int4_plain_mm PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                                  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(benchmark_int4_plain_mm PRIVATE CUDA_AVAILABLE=1)
set_property(TARGET benchmark_int4_plain_mm PROPERTY CUDA_ARCHITECTURES 80)
target_compile_options(
  benchmark_int4_plain_mm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
target_link_libraries(
  benchmark_int4_plain_mm PRIVATE aoti_cuda_shims executorch_core CUDA::cudart
)
add_test(NAME benchmark_int4_plain_mm_m1
         COMMAND benchmark_int4_plain_mm --M=1 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int4_plain_mm_m2
         COMMAND benchmark_int4_plain_mm --M=2 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int4_plain_mm_m3
         COMMAND benchmark_int4_plain_mm --M=3 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int4_plain_mm_m4
         COMMAND benchmark_int4_plain_mm --M=4 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)

add_executable(benchmark_int6_plain_mm benchmark_int6_plain_mm.cu)
target_include_directories(
  benchmark_int6_plain_mm PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                                  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(benchmark_int6_plain_mm PRIVATE CUDA_AVAILABLE=1)
set_property(TARGET benchmark_int6_plain_mm PROPERTY CUDA_ARCHITECTURES 80)
target_compile_options(
  benchmark_int6_plain_mm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
target_link_libraries(
  benchmark_int6_plain_mm PRIVATE aoti_cuda_shims executorch_core CUDA::cudart
)
add_test(NAME benchmark_int6_plain_mm_m1
         COMMAND benchmark_int6_plain_mm --M=1 --N=1024 --K=8192 --gs=16
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int6_plain_mm_m2
         COMMAND benchmark_int6_plain_mm --M=2 --N=1024 --K=256 --gs=16
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int6_plain_mm_m3
         COMMAND benchmark_int6_plain_mm --M=3 --N=1024 --K=256 --gs=16
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int6_plain_mm_m4
         COMMAND benchmark_int6_plain_mm --M=4 --N=64 --K=256 --gs=16
                 --warmup=1 --iters=1 --seeds=3
)

add_executable(benchmark_int5_plain_mm benchmark_int5_plain_mm.cu)
target_include_directories(
  benchmark_int5_plain_mm PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                                  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(benchmark_int5_plain_mm PRIVATE CUDA_AVAILABLE=1)
set_property(TARGET benchmark_int5_plain_mm PROPERTY CUDA_ARCHITECTURES 80)
target_compile_options(
  benchmark_int5_plain_mm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
target_link_libraries(
  benchmark_int5_plain_mm PRIVATE aoti_cuda_shims executorch_core CUDA::cudart
)
add_test(NAME benchmark_int5_plain_mm_m1
         COMMAND benchmark_int5_plain_mm --M=1 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int5_plain_mm_m2
         COMMAND benchmark_int5_plain_mm --M=2 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int5_plain_mm_m3
         COMMAND benchmark_int5_plain_mm --M=3 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)
add_test(NAME benchmark_int5_plain_mm_m4
         COMMAND benchmark_int5_plain_mm --M=4 --N=64 --K=256 --gs=32
                 --warmup=1 --iters=1 --seeds=3
)

foreach(test_name ${CUDA_KERNEL_TESTS})
  add_executable(${test_name} ${test_name}.cpp)

  target_include_directories(
    ${test_name} PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                         ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_compile_definitions(${test_name} PRIVATE CUDA_AVAILABLE=1)

  target_link_libraries(
    ${test_name} PRIVATE GTest::gtest GTest::gtest_main aoti_cuda_shims
                         executorch_core CUDA::cudart
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
