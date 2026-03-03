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

# When built as part of the main ExecuTorch CMake, Utils.cmake is already
# included. Only include it if executorch_target_link_options_shared_lib is not
# defined.
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
      PATHS ${TENSORRT_ROOT}/include ${TENSORRT_ROOT}/include/aarch64-linux-gnu
            ${TENSORRT_ROOT}/include/x86_64-linux-gnu
      PATH_SUFFIXES tensorrt
    )

    # Find TensorRT library (supports both x86_64 and aarch64/Jetson)
    find_library(
      TENSORRT_LIBRARY nvinfer
      PATHS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu
            ${TENSORRT_ROOT}/lib/x86_64-linux-gnu ${TENSORRT_ROOT}/lib64
    )

    # Fallback: find TRT from pip-installed tensorrt_libs package
    if(NOT TENSORRT_LIBRARY AND PYTHON_EXECUTABLE)
      execute_process(
        COMMAND
          ${PYTHON_EXECUTABLE} -c
          "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))"
        OUTPUT_VARIABLE _TRT_PIP_LIBS_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        RESULT_VARIABLE _TRT_PIP_RC
      )
      if(_TRT_PIP_RC EQUAL 0 AND _TRT_PIP_LIBS_DIR)
        message(
          STATUS "Found pip-installed tensorrt_libs: ${_TRT_PIP_LIBS_DIR}"
        )
        find_library(
          TENSORRT_LIBRARY nvinfer
          PATHS ${_TRT_PIP_LIBS_DIR}
          NO_DEFAULT_PATH
        )
        # pip packages ship versioned .so (e.g. libnvinfer.so.10) without
        # symlink
        if(NOT TENSORRT_LIBRARY)
          file(GLOB _TRT_PIP_SO "${_TRT_PIP_LIBS_DIR}/libnvinfer.so*")
          if(_TRT_PIP_SO)
            list(GET _TRT_PIP_SO 0 TENSORRT_LIBRARY)
            message(
              STATUS "Using versioned TRT lib from pip: ${TENSORRT_LIBRARY}"
            )
          endif()
        endif()
      endif()
    endif()

    # Fallback header search: pip packages don't ship headers, but the TensorRT
    # Python package caches them in ~/.cache/tensorrt/include/ on first use.
    if(NOT TENSORRT_INCLUDE_DIR)
      find_path(
        TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS $ENV{HOME}/.cache/tensorrt/include
        NO_DEFAULT_PATH
      )
    endif()

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
      message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
      message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")

      # Get the directory containing TensorRT library and add to link
      # directories
      get_filename_component(TENSORRT_LIB_DIR ${TENSORRT_LIBRARY} DIRECTORY)
      message(STATUS "TensorRT library directory: ${TENSORRT_LIB_DIR}")
      link_directories(${TENSORRT_LIB_DIR})
    endif()
  endif()

  # Verify TensorRT was found
  if(NOT TensorRT_FOUND AND NOT (TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY))
    message(
      FATAL_ERROR
        "TensorRT not found. Set TENSORRT_HOME or TENSORRT_DIR environment variable, "
        "or pass -DTENSORRT_HOME=/path/to/tensorrt to CMake. "
        "For pip installations, ensure tensorrt is installed in the active Python environment "
        "and headers are available (typically cached in ~/.cache/tensorrt/include/)."
    )
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

  # TensorRT executor runner binary This binary links the TensorRT backend with
  # the ExecuTorch runtime to run exported .pte files on NVIDIA GPUs.
  #
  # Build: cmake -DEXECUTORCH_BUILD_TENSORRT=ON ... cmake --build . --target
  # tensorrt_executor_runner
  #
  # Usage: ./tensorrt_executor_runner --model_path=model_tensorrt.pte
  add_executable(
    tensorrt_executor_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/nvidia/tensorrt/tensorrt_executor_runner.cpp
  )

  target_include_directories(
    tensorrt_executor_runner
    PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
           $<BUILD_INTERFACE:${TENSORRT_COMMON_INCLUDE_DIRS}>
  )

  # Note: We need WHOLE_ARCHIVE for tensorrt_backend to ensure static
  # registration of the backend runs. This is necessary because the backend
  # registers itself via a global constructor.
  #
  # We use target_link_options to ensure the whole-archive flags are placed
  # correctly around the library in the link command. CMake's
  # target_link_libraries with string flags may not preserve the correct order.
  target_link_libraries(
    tensorrt_executor_runner PRIVATE executorch extension_data_loader
                                     extension_runner_util portable_kernels
  )

  # Apply whole-archive linking for tensorrt_backend via link options This
  # ensures the static backend registration is included
  if(APPLE)
    target_link_options(
      tensorrt_executor_runner PRIVATE
      "SHELL:LINKER:-force_load,$<TARGET_FILE:tensorrt_backend>"
    )
  else()
    target_link_options(
      tensorrt_executor_runner
      PRIVATE
      "SHELL:LINKER:--whole-archive $<TARGET_FILE:tensorrt_backend> LINKER:--no-whole-archive"
    )
  endif()

  # Add tensorrt_backend's transitive dependencies (CUDA, TensorRT) We can't use
  # tensorrt_backend directly in target_link_libraries because it would be
  # linked twice (once via whole-archive, once normally)
  target_link_libraries(tensorrt_executor_runner PRIVATE CUDA::cudart)
  if(TENSORRT_LIBRARY)
    target_link_libraries(tensorrt_executor_runner PRIVATE ${TENSORRT_LIBRARY})
  elseif(TensorRT_FOUND)
    target_link_libraries(tensorrt_executor_runner PRIVATE TensorRT::nvinfer)
  endif()

  # Ensure tensorrt_backend is built before the runner
  add_dependencies(tensorrt_executor_runner tensorrt_backend)

  target_compile_options(tensorrt_executor_runner PRIVATE -frtti -fexceptions)

  install(TARGETS tensorrt_executor_runner DESTINATION ${CMAKE_INSTALL_BINDIR})

endif()
