# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(native_whisper_runner STATIC WhisperRunner.cpp)
target_link_libraries(
  native_whisper_runner
  PUBLIC native_engine
  PRIVATE ${_native_runner_gflags} native_package native_runner_utils
          native_runtime
)
target_compile_options(
  native_whisper_runner PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                                $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)

set(_whisper_runners)
if(EXECUTORCH_BUILD_NATIVE_VULKAN)
  add_executable(run_whisper run_whisper.cpp)
  target_link_libraries(run_whisper PRIVATE native_vulkan_engine)
  list(APPEND _whisper_runners run_whisper)
endif()
if(EXECUTORCH_BUILD_NATIVE_GGML)
  add_executable(run_whisper_ggml run_whisper_ggml.cpp)
  target_link_libraries(run_whisper_ggml PRIVATE native_ggml_engine)
  list(APPEND _whisper_runners run_whisper_ggml)
endif()
foreach(target IN LISTS _whisper_runners)
  target_link_libraries(
    ${target} PRIVATE native_whisper_runner ${_native_runner_gflags}
  )
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
  install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()
