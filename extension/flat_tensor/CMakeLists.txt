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

list(TRANSFORM _extension_flat_tensor__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(extension_flat_tensor ${_extension_flat_tensor__srcs})
target_link_libraries(extension_flat_tensor executorch_core)
target_include_directories(
  extension_flat_tensor
  PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/extension/flat_tensor/include>
         $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/flatbuffers/include>
         ${_common_include_directories}
)
target_compile_options(extension_flat_tensor PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_flat_tensor
  EXPORT ExecuTorchTargets
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

add_subdirectory(serialize)
add_dependencies(extension_flat_tensor flat_tensor_schema)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
