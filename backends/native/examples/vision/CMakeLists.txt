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
  set(_native_vision_vulkan_engine executorch_native::runtime)
  set(EXECUTORCH_BUILD_NATIVE_VULKAN ON)

  if(NOT TARGET gflags AND NOT TARGET gflags::gflags)
    find_package(gflags REQUIRED)
  endif()
  if(TARGET gflags::gflags)
    set(_native_runner_gflags gflags::gflags)
  else()
    set(_native_runner_gflags gflags)
  endif()
else()
  set(_native_vision_runtime native_engine)
  set(_native_vision_vulkan_engine native_vulkan_engine)
endif()

add_library(native_classification_runner STATIC ClassificationRunner.cpp)
target_link_libraries(
  native_classification_runner
  PUBLIC ${_native_vision_runtime}
  PRIVATE ${_native_runner_gflags} native_runner_utils
)
target_compile_options(
  native_classification_runner
  PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
          $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

function(add_native_classification_runner target source engine)
  add_executable(${target} ${source})
  target_link_libraries(
    ${target} PRIVATE native_classification_runner ${_native_runner_gflags}
                      ${engine}
  )
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
  install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if(EXECUTORCH_BUILD_NATIVE_VULKAN)
  add_native_classification_runner(
    run_classification run_classification.cpp ${_native_vision_vulkan_engine}
  )
endif()

if(EXECUTORCH_BUILD_NATIVE_CPU)
  add_native_classification_runner(
    run_classification_cpu run_classification_cpu.cpp native_cpu_engine
  )
endif()
