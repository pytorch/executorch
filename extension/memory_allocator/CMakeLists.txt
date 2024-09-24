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

list(TRANSFORM _extension_memory_allocator__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_memory_allocator ${_extension_memory_allocator__srcs})
target_link_libraries(extension_memory_allocator executorch)
target_include_directories(extension_memory_allocator PUBLIC ${EXECUTORCH_ROOT}/..)
target_compile_options(extension_memory_allocator PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_memory_allocator
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
