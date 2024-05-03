# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

list(TRANSFORM _extension_runner_util__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_runner_util ${_extension_runner_util__srcs})
target_link_libraries(extension_runner_util executorch)
target_include_directories(extension_runner_util PUBLIC ${EXECUTORCH_ROOT}/..)
target_compile_options(extension_runner_util PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_runner_util
  DESTINATION ${CMAKE_BINARY_DIR}/lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
