# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CMake configuration for building the multimodal example
# Usage: Include this in your main CMakeLists.txt or use it to build standalone

cmake_minimum_required(VERSION 3.19)
project(multimodal_example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
executorch_load_build_variables()

# Create the multimodal example executable
add_executable(multimodal_example multimodal_example.cpp)

# Link against the llm runner library and its dependencies
target_link_libraries(multimodal_example 
  PRIVATE 
    extension_llm_runner
    executorch_core
    extension_module
    extension_tensor
    tokenizers::tokenizers
)

# Include directories
target_include_directories(multimodal_example PRIVATE
  ${EXECUTORCH_ROOT}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set compilation properties
set_target_properties(multimodal_example PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)