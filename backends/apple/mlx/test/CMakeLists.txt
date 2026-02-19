# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MLX backend tests

# Test source directory - resolve symlinks to get the real path. This is needed
# because editable installs create symlinks (pytorch/executorch ->
# src/executorch), and CMAKE_CURRENT_SOURCE_DIR may resolve through those
# symlinks to a path where the .cpp files don't exist.
file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" _mlx_test_dir)

# Op test runner - generic test binary for testing individual ops
add_executable(op_test_runner ${_mlx_test_dir}/op_test_runner.cpp)

target_link_libraries(
  op_test_runner PRIVATE extension_module extension_tensor executorch
                         mlxdelegate
)

target_include_directories(
  op_test_runner PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..
)

# Multi-threaded inference test
include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

et_cxx_test(
  multi_thread_test_runner
  SOURCES
  ${_mlx_test_dir}/multi_thread_test_runner.cpp
  EXTRA_LIBS
  extension_module
  extension_tensor
  mlxdelegate
)
