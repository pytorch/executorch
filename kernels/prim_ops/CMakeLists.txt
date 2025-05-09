# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for portable kernels. Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

set(_common_compile_options -Wno-deprecated-declarations)
list(TRANSFORM _executorch__srcs PREPEND "${EXECUTORCH_ROOT}/")

add_library(prim_ops_lib ${_executorch__srcs})
target_link_libraries(prim_ops_lib PRIVATE executorch_core)
target_include_directories(prim_ops_lib PUBLIC ${_common_include_directories})
target_compile_definitions(prim_ops_lib PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_compile_options(prim_ops_lib PUBLIC ${_common_compile_options})
target_link_options_shared_lib(prim_ops_lib)

install(
  TARGETS prim_ops_lib
  DESTINATION lib
  PUBLIC_HEADER DESTINATION include/executorch/kernels/prim_ops/
)
