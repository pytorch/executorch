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

enable_language(Swift)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

add_library(extension_llm_apple)

file(GLOB OBJC_SOURCES ExecuTorchLLM/Exported/*.m ExecuTorchLLM/Exported/*.mm)

target_sources(extension_llm_apple PRIVATE ${OBJC_SOURCES})

target_include_directories(extension_llm_apple PUBLIC ExecuTorchLLM/Exported)

if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "EXECUTORCH_BUILD_EXTENSION_LLM_APPLE requires EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER."
  )
endif()

find_library(FOUNDATION_FRAMEWORK Foundation)
target_link_libraries(
  extension_llm_apple PRIVATE extension_llm_runner ${FOUNDATION_FRAMEWORK}
)

set_source_files_properties(
  ${OBJC_SOURCES} PROPERTIES COMPILE_FLAGS "-fobjc-arc" "-fno-exceptions"
                                                        "-fno-rtti"
)

set_target_properties(
  extension_llm_apple PROPERTIES XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION
                                 YES
)
