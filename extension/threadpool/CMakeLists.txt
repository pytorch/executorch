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

# get_threadpool() is a process-wide singleton, so in a shared (pip C++ SDK)
# build it must live in exactly one shared object that every consumer (the
# kernels lib, XNNPACK and other backends, _portable_lib) links dynamically.
# Otherwise each consumer that statically links extension_threadpool embeds its
# own pool. Build it SHARED for EXECUTORCH_BUILD_SHARED and keep it STATIC
# everywhere else (iOS/Android/embedded are unaffected). Mirrors iOS's
# standalone threadpool framework. cpuinfo and pthreadpool are static and
# bundled in; executorch_core resolves from libexecutorch.so at load.
if(EXECUTORCH_BUILD_SHARED)
  add_library(
    extension_threadpool SHARED threadpool.cpp threadpool_guard.cpp
                                thread_parallel.cpp cpuinfo_utils.cpp
  )
  set_target_properties(
    extension_threadpool
    PROPERTIES OUTPUT_NAME executorch_threadpool
               VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
  )
else()
  add_library(
    extension_threadpool threadpool.cpp threadpool_guard.cpp
                         thread_parallel.cpp cpuinfo_utils.cpp
  )
endif()
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
