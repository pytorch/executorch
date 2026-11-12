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

add_executable(run_whisper_ggml_cpu run_whisper_ggml_cpu.cpp)
target_link_libraries(
  run_whisper_ggml_cpu PRIVATE native_whisper_runner native_ggml_engine
                               ${_native_runner_gflags}
)
foreach(target native_whisper_runner run_whisper_ggml_cpu)
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
endforeach()
install(TARGETS run_whisper_ggml_cpu
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
