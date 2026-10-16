# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT EXECUTORCH_BUILD_NATIVE_VULKAN)
  message(
    FATAL_ERROR "run_classification requires EXECUTORCH_BUILD_NATIVE_VULKAN=ON"
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
  run_classification
  PRIVATE ${_native_vision_gflags} native_engine native_package
          native_runner_utils native_runtime native_vulkan_engine
)
target_compile_options(
  run_classification PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                             $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

install(TARGETS run_classification RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
