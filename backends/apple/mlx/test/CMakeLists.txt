# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MLX backend tests

# Op test runner - generic test binary for testing individual ops
add_executable(op_test_runner op_test_runner.cpp)

target_link_libraries(
  op_test_runner
  PRIVATE
    extension_module
    extension_tensor
    executorch
    mlxdelegate
)

target_include_directories(
  op_test_runner
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
)
