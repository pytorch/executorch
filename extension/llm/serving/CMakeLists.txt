# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(extension_llm_serving INTERFACE)
target_compile_features(extension_llm_serving INTERFACE cxx_std_17)
target_include_directories(
  extension_llm_serving INTERFACE ${_common_include_directories}
)

target_link_libraries(extension_llm_serving INTERFACE executorch_core)

install(TARGETS extension_llm_serving EXPORT ExecuTorchTargets)
install(FILES types.h prompt_history.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/serving
)

# Public serving inputs do not require the legacy runner.
if(NOT EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER)
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../runner/multimodal_input.h
          ${CMAKE_CURRENT_SOURCE_DIR}/../runner/image.h
          ${CMAKE_CURRENT_SOURCE_DIR}/../runner/audio.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/llm/runner
  )
endif()
