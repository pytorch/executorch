# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(nanogpt_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set options for executorch build.
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON) # Build with Xnnpack backend

# Include the executorch subdirectory.
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/third-party/executorch
  ${CMAKE_BINARY_DIR}/executorch
)

add_executable(nanogpt_runner main.cpp)
target_link_libraries(
  nanogpt_runner
  PRIVATE executorch
          extension_module_static # Provides the Module class
          extension_tensor # Provides the TensorPtr class
          optimized_native_cpu_ops_lib # Provides baseline cross-platform
                                       # kernels
          xnnpack_backend # Provides the XNNPACK CPU acceleration backend
)
