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

list(TRANSFORM _extension_training__srcs PREPEND "${EXECUTORCH_ROOT}/")

add_library(extension_training ${_extension_training__srcs})
target_include_directories(
  extension_training PUBLIC ${_common_include_directories}
)

target_include_directories(extension_training PUBLIC ${EXECUTORCH_ROOT}/..)
target_compile_options(extension_training PUBLIC ${_common_compile_options})
target_link_libraries(extension_training executorch_core
    extension_data_loader extension_module extension_tensor)


list(TRANSFORM _train_xor__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_executable(train_xor ${_train_xor__srcs})
target_include_directories(
  train_xor PUBLIC ${_common_include_directories}
)
target_link_libraries(
train_xor gflags executorch_core portable_ops_lib extension_tensor
    extension_training program_schema
)
target_compile_options(train_xor PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_training
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)
