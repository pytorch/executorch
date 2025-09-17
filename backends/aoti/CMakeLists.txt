# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build AOTI backend for runtime.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Determine which backend to use (mutually exclusive)
if(APPLE AND NOT DEFINED ENV{USE_CUDA})
  # On Apple platforms, prefer Metal unless CUDA is explicitly requested
  message(STATUS "Building with Metal support on macOS")
  set(AOTI_METAL TRUE)
  set(AOTI_CUDA FALSE)
else()
  # On non-Apple platforms or when CUDA is explicitly requested, use CUDA
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found, enabling CUDA support")
    set(AOTI_CUDA TRUE)
    set(AOTI_METAL FALSE)
  else()
    message(FATAL_ERROR "CUDA Toolkit not found. Install CUDA or use an Apple platform for Metal support.")
  endif()
endif()

# Use ExecutorTorch's standard way to find PyTorch libraries for AOTI
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
find_package_torch()

set(_aoti_sources
    runtime/aoti_backend.cpp
    runtime/aoti_model_container.cpp
    runtime/shims/memory.cpp
    runtime/shims/tensor_attribute.cpp
    runtime/shims/utils.cpp)

# Add Metal helper for macOS
if(AOTI_METAL)
  list(APPEND _aoti_sources runtime/shims/metal_helper.mm runtime/shims/shim_mps.mm)
endif()

add_library(aoti_backend STATIC ${_aoti_sources})
target_include_directories(
  aoti_backend
  PUBLIC
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
    $<INSTALL_INTERFACE:include>
    # PyTorch AOTI headers from ExecutorTorch's torch detection
    ${TORCH_INCLUDE_DIRS}
)

# Conditionally add CUDA includes
if(AOTI_CUDA)
  target_include_directories(
    aoti_backend
    PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
endif()

# Add Metal framework on macOS
if(AOTI_METAL)
  # Add Metal framework-specific compile definitions
  target_compile_definitions(aoti_backend PUBLIC AOTI_METAL)

  # Link Metal framework
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  target_link_libraries(aoti_backend PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
endif()

if(AOTI_CUDA)
  target_compile_definitions(aoti_backend PUBLIC AOTI_CUDA)
endif()
target_compile_options(aoti_backend PUBLIC -fexceptions -frtti -fPIC)

# Ensure symbols are exported properly (platform-specific)
if(APPLE)
  target_link_options(aoti_backend PUBLIC -Wl,-export_dynamic)
else()
  target_link_options(aoti_backend PUBLIC -Wl,--export-dynamic)
endif()

# Link against appropriate backends and standard libraries
target_link_libraries(
  aoti_backend
  PUBLIC
    extension_tensor
    ${CMAKE_DL_LIBS}
    # Link PyTorch libraries for AOTI functions
    ${TORCH_LIBRARIES}
)

# Conditionally link CUDA libraries
if(AOTI_CUDA)
  target_link_libraries(
    aoti_backend
    PUBLIC
    CUDA::cudart
  )
endif()
# If you need other CUDA libraries, link them similarly:
# target_link_libraries(aoti_backend PUBLIC CUDA::cublas CUDA::cufft ...)
# If you have a custom function, keep it
executorch_target_link_options_shared_lib(aoti_backend)
install(
  TARGETS aoti_backend
  EXPORT ExecuTorchTargets
  DESTINATION lib
)
