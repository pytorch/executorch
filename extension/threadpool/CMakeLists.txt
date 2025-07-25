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

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_library(
  extension_threadpool threadpool.cpp threadpool_guard.cpp thread_parallel.cpp
                       cpuinfo_utils.cpp
)
target_link_libraries(
  extension_threadpool PUBLIC executorch_core cpuinfo pthreadpool
)
target_include_directories(
  extension_threadpool PUBLIC ${_common_include_directories}
)
target_include_directories(
  extension_threadpool
  PUBLIC
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include>
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include>
)
target_compile_definitions(extension_threadpool PUBLIC ET_USE_THREADPOOL)
target_compile_options(extension_threadpool PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_threadpool
  EXPORT ExecuTorchTargets
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
