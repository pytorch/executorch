# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(native_vision_runner LANGUAGES CXX)
  include(GNUInstallDirs)
  find_package(executorch_native CONFIG REQUIRED)
  set(EXECUTORCH_ROOT "${executorch_native_PREFIX}/include/executorch")
  add_subdirectory(../runner_utils runner_utils)
  set(_native_vision_runtime executorch_native::runtime)
else()
  set(_native_vision_runtime native_engine native_package native_runtime
                             native_vulkan_engine
  )
endif()

if(NOT TARGET gflags AND NOT TARGET gflags::gflags)
  find_package(gflags REQUIRED)
endif()
if(TARGET gflags::gflags)
  set(_native_vision_gflags gflags::gflags)
else()
  set(_native_vision_gflags gflags)
endif()

add_executable(run_classification run_classification.cpp)
target_link_libraries(
  run_classification PRIVATE ${_native_vision_gflags} native_runner_utils
                             ${_native_vision_runtime}
)
target_compile_options(
  run_classification PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                             $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

install(TARGETS run_classification RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
