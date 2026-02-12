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

# Try to enable CUDA language when a working CUDA compiler toolchain is
# available. Some CI environments (notably Windows packaging jobs) provide
# CUDAToolkit headers/libs but cannot complete CUDA compiler identification. In
# those cases, keep configuration working and skip CUDA-only sources below.
if(NOT CMAKE_CUDA_COMPILER)
  include(CheckLanguage)
  check_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

# Centralize Windows/MSVC checks used throughout this file.
set(_cuda_is_msvc_toolchain OFF)
if(MSVC)
  set(_cuda_is_msvc_toolchain ON)
endif()

set(_cuda_is_windows_msvc OFF)
if(WIN32 AND _cuda_is_msvc_toolchain)
  set(_cuda_is_windows_msvc ON)
endif()

# Common C++ compile options for CUDA backend targets.
if(_cuda_is_msvc_toolchain)
  set(_cuda_cxx_compile_options /EHsc /GR)
else()
  set(_cuda_cxx_compile_options -fexceptions -frtti -fPIC)
endif()

# Platform-specific linker option for exporting symbols from shared libs.
set(_cuda_export_dynamic_option "")
if(NOT _cuda_is_msvc_toolchain)
  if(APPLE)
    set(_cuda_export_dynamic_option -Wl,-export_dynamic)
  else()
    set(_cuda_export_dynamic_option -Wl,--export-dynamic)
  endif()
endif()

# Use ExecuTorch's standard way to find PyTorch libraries for AOTI
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
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)
# Ensure symbols are exported properly
if(_cuda_export_dynamic_option)
  target_link_options(cuda_tensor_maker PUBLIC ${_cuda_export_dynamic_option})
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
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)

# Link against ExecuTorch core libraries
target_link_libraries(cuda_platform PRIVATE executorch_core ${CMAKE_DL_LIBS})

install(
  TARGETS cuda_platform
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

# CUDA-specific AOTI shim symbols (dynamically linked)
set(_aoti_cuda_shim_sources runtime/shims/memory.cpp
                            runtime/shims/cuda_guard.cpp
)

# Only build int4mm shim when CUDA language/toolchain is available.
if(CMAKE_CUDA_COMPILER)
  list(APPEND _aoti_cuda_shim_sources runtime/shims/int4mm.cu)
endif()

add_library(aoti_cuda_shims SHARED ${_aoti_cuda_shim_sources})

# Define CUDA_AVAILABLE to use SlimTensor on GPU in common_shims_slim.h
target_compile_definitions(aoti_cuda_shims PRIVATE CUDA_AVAILABLE=1)

# Define export macros for shared library. Use WIN32 (not just MSVC) so MinGW
# cross-compiles also emit dllexport symbols for AOTI shims.
if(WIN32)
  target_compile_definitions(aoti_cuda_shims PRIVATE EXPORT_AOTI_FUNCTIONS)
  if(_cuda_is_windows_msvc)
    # Ensure proper DLL import/export library naming on Windows
    set_target_properties(
      aoti_cuda_shims PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
  endif()
endif()

target_include_directories(
  aoti_cuda_shims
  PUBLIC ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
         $<INSTALL_INTERFACE:include>
)

target_compile_options(
  aoti_cuda_shims
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)

# Ensure symbols are exported properly
if(_cuda_export_dynamic_option)
  target_link_options(aoti_cuda_shims PUBLIC ${_cuda_export_dynamic_option})
endif()

# Link against CUDA::cudart, common AOTI library, cuda_tensor_maker, and
# platform utilities. On non-MSVC, use --whole-archive for
# aoti_common_shims_slim to force shim symbol retention.
if(_cuda_is_msvc_toolchain)
  target_link_libraries(
    aoti_cuda_shims PRIVATE cuda_platform cuda_tensor_maker CUDA::cudart
                            ${CMAKE_DL_LIBS}
  )
  # Link object library directly so symbols are pulled exactly once while
  # avoiding duplicate static/object inclusion and interface leakage.
  target_link_libraries(aoti_cuda_shims PRIVATE aoti_common_shims_slim_obj)
else()
  target_link_libraries(
    aoti_cuda_shims
    PRIVATE cuda_platform
    PUBLIC -Wl,--whole-archive aoti_common_shims_slim -Wl,--no-whole-archive
           cuda_tensor_maker CUDA::cudart ${CMAKE_DL_LIBS}
  )
endif()

if(NOT _cuda_is_msvc_toolchain)
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
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_cuda_cxx_compile_options}>"
)
# Ensure symbols are exported properly
if(_cuda_export_dynamic_option)
  target_link_options(aoti_cuda_backend PUBLIC ${_cuda_export_dynamic_option})
endif()

# Link against shims library and other dependencies On Windows (MSVC), use
# PRIVATE linkage for aoti_cuda_shims since the DLL is copied to the executable
# directory. On other platforms, use PUBLIC so the dependency propagates to
# consumers.
target_link_libraries(
  aoti_cuda_backend PUBLIC cuda_platform extension_tensor cuda_tensor_maker
                           CUDA::cudart ${CMAKE_DL_LIBS}
)

if(_cuda_is_msvc_toolchain)
  # cuda_backend.cpp uses SlimTensor CUDA utilities (e.g. getCurrentCUDAStream)
  # from aoti_common_shims_slim via headers; propagate the static lib so final
  # MSVC links (e.g. parakeet_runner) can resolve those C++ symbols.
  target_link_libraries(
    aoti_cuda_backend
    PRIVATE aoti_cuda_shims
    PUBLIC aoti_common_shims_slim
  )
else()
  target_link_libraries(aoti_cuda_backend PUBLIC aoti_cuda_shims)
endif()

executorch_target_link_options_shared_lib(aoti_cuda_backend)

install(
  TARGETS aoti_cuda_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
