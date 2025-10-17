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

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

list(TRANSFORM _kernels_util_all_deps__srcs PREPEND "${EXECUTORCH_ROOT}/")

set(_common_compile_options
    $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-deprecated-declarations>
)

add_library(kernels_util_all_deps ${_kernels_util_all_deps__srcs})
target_link_libraries(kernels_util_all_deps PRIVATE executorch_core)
target_include_directories(
  kernels_util_all_deps PUBLIC ${_common_include_directories}
)
target_compile_definitions(
  kernels_util_all_deps PUBLIC C10_USING_CUSTOM_GENERATED_MACROS
)
target_compile_options(kernels_util_all_deps PUBLIC ${_common_compile_options})

install(
  TARGETS kernels_util_all_deps
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
