# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT TARGET gflags AND NOT TARGET gflags::gflags)
  find_package(gflags REQUIRED)
endif()
if(TARGET gflags::gflags)
  set(_native_vision_gflags gflags::gflags)
else()
  set(_native_vision_gflags gflags)
endif()

add_library(native_classification_runner STATIC ClassificationRunner.cpp)
target_link_libraries(
  native_classification_runner
  PUBLIC native_engine
  PRIVATE ${_native_vision_gflags} native_runner_utils
)
target_compile_options(
  native_classification_runner
  PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
          $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

function(add_native_classification_runner target source engine)
  add_executable(${target} ${source})
  target_link_libraries(
    ${target}
    PRIVATE native_classification_runner ${_native_vision_gflags} ${engine}
  )
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
  install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if(EXECUTORCH_BUILD_NATIVE_VULKAN)
  add_native_classification_runner(
    run_classification run_classification.cpp native_vulkan_engine
  )
  add_native_classification_runner(
    run_classification_vulkan run_classification.cpp native_vulkan_engine
  )
endif()

if(EXECUTORCH_BUILD_NATIVE_CPU)
  add_native_classification_runner(
    run_classification_cpu run_classification_cpu.cpp native_cpu_engine
  )
endif()
