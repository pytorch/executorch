# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build AOTI CUDA backend for runtime.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

find_package(CUDAToolkit REQUIRED)

# Use ExecutorTorch's standard way to find PyTorch libraries for AOTI
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

# CUDA tensor maker for backends that support incontiguous tensors
set(_tensor_maker_sources runtime/tensor/tensor_maker.cpp)
add_library(cuda_tensor_maker STATIC ${_tensor_maker_sources})
target_include_directories(
  cuda_tensor_maker
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)
target_compile_options(
  cuda_tensor_maker
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)
# Ensure symbols are exported properly
if(APPLE)
  target_link_options(cuda_tensor_maker PUBLIC -Wl,-export_dynamic)
else()
  target_link_options(
    cuda_tensor_maker PUBLIC
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--export-dynamic>
  )
endif()

# Link against ExecuTorch core libraries
target_link_libraries(cuda_tensor_maker PUBLIC executorch ${CMAKE_DL_LIBS})
executorch_target_link_options_shared_lib(cuda_tensor_maker)

install(
  TARGETS cuda_tensor_maker
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA-specific AOTI functionality
set(_aoti_cuda_sources
    runtime/cuda_backend.cpp
    runtime/shims/memory.cpp
    runtime/shims/tensor_attribute.cpp
    runtime/guard.cpp
    runtime/shims/cuda_guard.cpp
    runtime/shims/int4mm.cu
    runtime/platform/platform.cpp
)
add_library(aoti_cuda STATIC ${_aoti_cuda_sources})
target_include_directories(
  aoti_cuda
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
         # PyTorch AOTI headers from ExecutorTorch's torch detection
         ${TORCH_INCLUDE_DIRS}
)
target_compile_options(
  aoti_cuda PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
                   $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)
# Ensure symbols are exported properly
target_link_options(
  aoti_cuda PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--export-dynamic>
)

# Link against CUDA::cudart, common AOTI library, cuda_tensor_maker, and PyTorch
# CUDA libraries
target_link_libraries(
  aoti_cuda PUBLIC aoti_common cuda_tensor_maker CUDA::cudart ${CMAKE_DL_LIBS}
)
# If you need other CUDA libraries, link them similarly:
# target_link_libraries(aoti_cuda PUBLIC CUDA::cublas CUDA::cufft ...)
executorch_target_link_options_shared_lib(aoti_cuda)

if(BUILD_TESTING)
  add_executable(multimodal_benchmark tests/multimodal_benchmark.cpp)
  target_link_libraries(
    multimodal_benchmark PUBLIC aoti_cuda extension_module_static
                                extension_flat_tensor portable_ops_lib
  )
endif()

install(
  TARGETS aoti_cuda
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
