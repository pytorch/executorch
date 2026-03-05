# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TensorRT Examples - Benchmark runner
#
# Build:
#   cmake -DEXECUTORCH_BUILD_TENSORRT=ON ...
#   cmake --build . --target benchmark
#
# Usage:
#   ./benchmark                        # all .pte/.onnx in current dir
#   ./benchmark -m mv3                 # mv3 .pte and .onnx in current dir
#   ./benchmark -d /tmp/trt -n 200     # all models in /tmp/trt, 200 iterations

cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(COMMON_INCLUDE_DIRS ${EXECUTORCH_ROOT}/..)

if(EXECUTORCH_BUILD_TENSORRT)
  find_library(NVONNXPARSER_LIBRARY nvonnxparser
    HINTS ${TENSORRT_HOME}/lib ${TENSORRT_HOME}/lib64
          /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  )

  add_executable(benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp)

  target_include_directories(
    benchmark
    PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
           $<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>
  )

  # extension_module builds as extension_module_static in OSS CMake.
  if(TARGET extension_module_static)
    set(_extension_module extension_module_static)
  elseif(TARGET extension_module)
    set(_extension_module extension_module)
  else()
    message(FATAL_ERROR
      "extension_module not found. Build with -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON")
  endif()

  if(NOT TARGET extension_tensor)
    message(FATAL_ERROR
      "extension_tensor not found. Build with -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON")
  endif()

  target_link_libraries(
    benchmark
    PRIVATE executorch
            ${_extension_module}
            extension_tensor
            portable_kernels
  )

  target_link_options(
    benchmark
    PRIVATE
      "SHELL:LINKER:--whole-archive $<TARGET_FILE:tensorrt_backend> LINKER:--no-whole-archive"
  )
  target_link_libraries(benchmark PRIVATE CUDA::cudart)
  if(TENSORRT_LIBRARY)
    target_link_libraries(benchmark PRIVATE ${TENSORRT_LIBRARY})
  endif()
  if(NVONNXPARSER_LIBRARY)
    target_link_libraries(benchmark PRIVATE ${NVONNXPARSER_LIBRARY})
  endif()
  add_dependencies(benchmark tensorrt_backend)

  target_compile_options(benchmark PRIVATE -frtti -fexceptions)

  install(TARGETS benchmark DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
