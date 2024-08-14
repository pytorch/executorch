# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(nanogpt_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the executorch subdirectory.
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_BINARY_DIR}/executorch
)

executorch_runtime(
    et_runtime
    BACKEND xnnpack
    EXTENSION module dataloader
    OPTIMIZED
)

add_executable(nanogpt_runner main.cpp)
target_link_libraries(
  nanogpt_runner
  PRIVATE et_runtime
)
