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

# Use dynamic linking for CUDA runtime
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

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
target_link_libraries(
  cuda_tensor_maker PRIVATE executorch_core ${CMAKE_DL_LIBS}
)
executorch_target_link_options_shared_lib(cuda_tensor_maker)

install(
  TARGETS cuda_tensor_maker
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# Platform utilities (load_library, close_library, etc.)
set(_cuda_platform_sources runtime/platform/platform.cpp)
add_library(cuda_platform STATIC ${_cuda_platform_sources})

target_include_directories(
  cuda_platform
  PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}> $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)

target_compile_options(
  cuda_platform
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)

# Link against ExecuTorch core libraries
target_link_libraries(cuda_platform PRIVATE executorch_core ${CMAKE_DL_LIBS})

install(
  TARGETS cuda_platform
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA-specific AOTI shim symbols (dynamically linked)
# Note: common_shims.h is header-only (all functions are inline)
set(_aoti_cuda_shim_sources
    runtime/shims/memory.cpp runtime/shims/tensor_attribute.cpp
    runtime/guard.cpp runtime/shims/cuda_guard.cpp runtime/shims/int4mm.cu
)

add_library(aoti_cuda_shims SHARED ${_aoti_cuda_shim_sources})

# Define CUDA_AVAILABLE to use SlimTensor in common_shims.h
target_compile_definitions(aoti_cuda_shims PRIVATE CUDA_AVAILABLE=1)

# Define export macros for shared library
if(MSVC)
  target_compile_definitions(aoti_cuda_shims PRIVATE EXPORT_AOTI_FUNCTIONS)

  # Ensure proper DLL import/export library naming on Windows
  set_target_properties(
    aoti_cuda_shims PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF
  )
endif()

target_include_directories(
  aoti_cuda_shims
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)

target_compile_options(
  aoti_cuda_shims
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)

# Ensure symbols are exported properly
target_link_options(
  aoti_cuda_shims PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--export-dynamic>
)

# Link against CUDA::cudart, common AOTI library, cuda_tensor_maker, and
# platform utilities
target_link_libraries(
  aoti_cuda_shims
  PRIVATE cuda_platform
  PUBLIC extension_tensor cuda_tensor_maker CUDA::cudart ${CMAKE_DL_LIBS}
)

if(NOT MSVC)
  executorch_target_link_options_shared_lib(aoti_cuda_shims)
endif()

install(
  TARGETS aoti_cuda_shims
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA backend implementation
set(_aoti_cuda_backend_sources runtime/cuda_backend.cpp)

# CUDA backend implementation
add_library(aoti_cuda_backend STATIC ${_aoti_cuda_backend_sources})

target_include_directories(
  aoti_cuda_backend
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)
target_compile_options(
  aoti_cuda_backend
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions -frtti -fPIC>
)
# Ensure symbols are exported properly
target_link_options(
  aoti_cuda_backend PUBLIC
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--export-dynamic>
)

# Link against shims library and other dependencies On Windows (MSVC), use
# PRIVATE linkage for aoti_cuda_shims since the DLL is copied to the executable
# directory. On other platforms, use PUBLIC so the dependency propagates to
# consumers.
target_link_libraries(
  aoti_cuda_backend PUBLIC cuda_platform extension_tensor cuda_tensor_maker
                           CUDA::cudart ${CMAKE_DL_LIBS}
)

if(MSVC)
  target_link_libraries(aoti_cuda_backend PRIVATE aoti_cuda_shims)
else()
  target_link_libraries(aoti_cuda_backend PUBLIC aoti_cuda_shims)
endif()

executorch_target_link_options_shared_lib(aoti_cuda_backend)

install(
  TARGETS aoti_cuda_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
