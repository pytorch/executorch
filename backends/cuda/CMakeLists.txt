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
# Build as SHARED library (.dll) on Windows MSVC, otherwise STATIC
if(MSVC)
  add_library(aoti_cuda SHARED ${_aoti_cuda_sources} ${CMAKE_CURRENT_SOURCE_DIR}/aoti_cuda.def)
  # Define export macros for Windows DLL
  target_compile_definitions(aoti_cuda PRIVATE
    EXPORT_AOTI_FUNCTIONS
    BUILDING_CUDA_BACKEND
  )
  # Ensure proper DLL import/export library naming on Windows with config-specific paths
  set_target_properties(aoti_cuda PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS OFF  # We use explicit exports via AOTI_CUDA_EXPORT and .def file
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
  )
else()
  add_library(aoti_cuda STATIC ${_aoti_cuda_sources})
endif()
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

# Link against CUDA::cudart, common AOTI library, and PyTorch CUDA libraries
target_link_libraries(
  aoti_cuda PUBLIC aoti_common CUDA::cudart ${CMAKE_DL_LIBS}
)
# If you need other CUDA libraries, link them similarly:
# target_link_libraries(aoti_cuda PUBLIC CUDA::cublas CUDA::cufft ...)

# Only apply shared lib options on non-Windows platforms
if(NOT MSVC)
  executorch_target_link_options_shared_lib(aoti_cuda)
endif()

if(BUILD_TESTING)
  # Add runtime
  add_executable(voxtral_runner tests/voxtral_runner.cpp)
  target_link_libraries(
    voxtral_runner PUBLIC aoti_cuda extension_module_static
                          extension_flat_tensor portable_ops_lib
  )
endif()

install(
  TARGETS aoti_cuda
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
