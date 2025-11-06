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
    ${EXECUTORCH_ROOT}/backends/aoti/common_shims.cpp
)
# Build as SHARED library (.dll) on Windows MSVC, otherwise STATIC
if(MSVC)
  add_library(aoti_cuda SHARED ${_aoti_cuda_sources})

  # Define export macros for Windows DLL
  target_compile_definitions(
    aoti_cuda PRIVATE EXPORT_AOTI_FUNCTIONS
  )

  # Ensure proper DLL import/export library naming on Windows with
  # config-specific paths
  set_target_properties(
    aoti_cuda
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF
  )
else()
  add_library(aoti_cuda STATIC ${_aoti_cuda_sources})
endif()

target_include_directories(
  aoti_cuda
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
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
  aoti_cuda PUBLIC extension_tensor cuda_tensor_maker CUDA::cudart ${CMAKE_DL_LIBS}
)

if(NOT MSVC)
  executorch_target_link_options_shared_lib(aoti_cuda)
endif()

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
