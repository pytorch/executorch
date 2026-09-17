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

# Threadpool size specifiers. Mutual exclusion is checking in default.cmake.
# Default to using performance cores if
# EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES isn't set.
set(_threadpool_size_flag)
if(EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES)
  set(_threadpool_size_flag "EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES")
else()
  set(_threadpool_size_flag "EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES")
endif()

# The thread pool is a process-wide singleton held in a function-local static,
# so every library that links it statically gets its own copy. Build it shared
# for the wheel, where several extensions are loaded into one interpreter, and
# keep it static everywhere else so no other build changes.
if(EXECUTORCH_BUILD_SHARED)
  set(_threadpool_library_type SHARED)
else()
  set(_threadpool_library_type STATIC)
endif()

add_library(
  extension_threadpool
  ${_threadpool_library_type} threadpool.cpp threadpool_guard.cpp
  thread_parallel.cpp cpuinfo_utils.cpp
)
if(EXECUTORCH_BUILD_SHARED)
  set_target_properties(
    extension_threadpool PROPERTIES OUTPUT_NAME executorch_threadpool
  )
  executorch_target_soname_policy(extension_threadpool)
  # Ships beside libexecutorch.so in the wheel's lib/ directory.
  executorch_target_shipped_runtime_path(extension_threadpool)
  # cpuinfo and pthreadpool are forced static, so bundle them inside this
  # library instead of making every consumer supply them.
  executorch_target_whole_archive(extension_threadpool cpuinfo)
  executorch_target_whole_archive(extension_threadpool pthreadpool)
  target_link_libraries(extension_threadpool PUBLIC executorch_shared)
else()
  target_link_libraries(
    extension_threadpool PUBLIC executorch_core cpuinfo pthreadpool
  )
endif()
target_include_directories(
  extension_threadpool PUBLIC ${_common_include_directories}
)
target_include_directories(
  extension_threadpool
  PUBLIC
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include>
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include>
)
target_compile_definitions(
  extension_threadpool PUBLIC ET_USE_THREADPOOL ${_threadpool_size_flag}
)
target_compile_options(extension_threadpool PUBLIC ${_common_compile_options})

# Install libraries
install(
  TARGETS extension_threadpool
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
