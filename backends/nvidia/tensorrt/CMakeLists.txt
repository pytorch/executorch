# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TensorRT backend for ExecuTorch
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
# The targets in this file will be built if EXECUTORCH_BUILD_TENSORRT is ON

cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

# When built as part of the main ExecuTorch CMake, Utils.cmake is already included.
# Only include it if executorch_target_link_options_shared_lib is not defined.
if(NOT COMMAND executorch_target_link_options_shared_lib)
  if(EXISTS ${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
    include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
  else()
    # Define a no-op fallback for standalone builds
    function(executorch_target_link_options_shared_lib target)
      # No-op: whole-archive linking is handled separately for the runner
    endfunction()
  endif()
endif()

if(EXECUTORCH_BUILD_TENSORRT)
  # Find TensorRT package
  find_package(TensorRT QUIET)

  if(NOT TensorRT_FOUND)
    # Try to find TensorRT manually via CMake variable or environment variable
    if(DEFINED TENSORRT_HOME)
      set(TENSORRT_ROOT ${TENSORRT_HOME})
    elseif(DEFINED ENV{TENSORRT_HOME})
      set(TENSORRT_ROOT $ENV{TENSORRT_HOME})
    elseif(DEFINED ENV{TENSORRT_DIR})
      set(TENSORRT_ROOT $ENV{TENSORRT_DIR})
    else()
      # Default to /usr for system installations (e.g., JetPack on Jetson)
      set(TENSORRT_ROOT "/usr")
    endif()

    message(STATUS "Looking for TensorRT in: ${TENSORRT_ROOT}")

    # Find TensorRT headers (supports both x86_64 and aarch64/Jetson)
    find_path(
      TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS
        ${TENSORRT_ROOT}/include
        ${TENSORRT_ROOT}/include/aarch64-linux-gnu
        ${TENSORRT_ROOT}/include/x86_64-linux-gnu
      PATH_SUFFIXES tensorrt
    )

    # Find TensorRT library (supports both x86_64 and aarch64/Jetson)
    find_library(
      TENSORRT_LIBRARY nvinfer
      PATHS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib/aarch64-linux-gnu
        ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
        ${TENSORRT_ROOT}/lib64
    )

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
      message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
      message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")

      # Get the directory containing TensorRT library and add to link directories
      get_filename_component(TENSORRT_LIB_DIR ${TENSORRT_LIBRARY} DIRECTORY)
      message(STATUS "TensorRT library directory: ${TENSORRT_LIB_DIR}")
      link_directories(${TENSORRT_LIB_DIR})
    endif()
  endif()

  # Verify TensorRT was found
  if(NOT TensorRT_FOUND AND NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR
      "TensorRT not found. Set TENSORRT_HOME or TENSORRT_DIR environment variable, "
      "or pass -DTENSORRT_HOME=/path/to/tensorrt to CMake.")
  endif()

  # Find CUDA
  find_package(CUDAToolkit REQUIRED)

  # Define common include directories (used by backend and runner/test binaries)
  set(TENSORRT_COMMON_INCLUDE_DIRS ${EXECUTORCH_ROOT}/..)

  # TensorRT backend static library
  add_library(tensorrt_backend STATIC)

  # Enable exceptions and RTTI for TensorRT backend
  target_compile_options(tensorrt_backend PRIVATE -frtti -fexceptions)

  target_include_directories(
    tensorrt_backend
    PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
           $<BUILD_INTERFACE:${TENSORRT_COMMON_INCLUDE_DIRS}>
           $<INSTALL_INTERFACE:include>
  )

  if(TENSORRT_INCLUDE_DIR)
    target_include_directories(
      tensorrt_backend PUBLIC $<BUILD_INTERFACE:${TENSORRT_INCLUDE_DIR}>
    )
  endif()

  # Add source files
  target_sources(
    tensorrt_backend
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/runtime/TensorRTBackend.cpp
            ${CMAKE_CURRENT_LIST_DIR}/runtime/TensorRTExecutor.cpp
  )

  # Link dependencies
  target_link_libraries(tensorrt_backend PUBLIC executorch_core CUDA::cudart)

  if(TENSORRT_LIBRARY)
    target_link_libraries(tensorrt_backend PUBLIC ${TENSORRT_LIBRARY})
  elseif(TensorRT_FOUND)
    target_link_libraries(tensorrt_backend PUBLIC TensorRT::nvinfer)
  endif()

  # Force link the whole library to ensure static registration works
  executorch_target_link_options_shared_lib(tensorrt_backend)

  # Install TensorRT backend library
  install(
    TARGETS tensorrt_backend
    EXPORT ExecuTorchTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

endif()
