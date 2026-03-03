# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NVIDIA Backend Examples
#
# This CMakeLists.txt includes the TensorRT examples subdirectory.
#
# Supported platforms: - Linux x86_64 with NVIDIA GPU (devgpu, workstations) -
# NVIDIA Jetson (Orin Nano, AGX Orin, etc.)
#
# Build instructions: cmake .. -DEXECUTORCH_BUILD_TENSORRT=ON cmake --build .
# --target benchmark_runner_tensorrt

cmake_minimum_required(VERSION 3.19)

project(nvidia_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure compile_commands.json is generated for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Include utility CMake scripts from ExecuTorch
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Find CUDA (optional - needed for TensorRT backend)
find_package(CUDAToolkit QUIET)

# Add TensorRT examples subdirectory
add_subdirectory(tensorrt)
