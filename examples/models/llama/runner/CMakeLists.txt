# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for selective build demo.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# build llama_runner library
list(TRANSFORM _llama_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

target_include_directories(
  extension_module INTERFACE ${_common_include_directories}
)

if(CMAKE_TOOLCHAIN_IOS
   OR ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(llama_runner STATIC ${_llama_runner__srcs})
else()
  add_library(llama_runner SHARED ${_llama_runner__srcs})
endif()

# For extension_llm_runner
if(NOT TARGET extension_llm_runner)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/extension/llm/runner
    ${CMAKE_CURRENT_BINARY_DIR}/../../../../extension/llm/runner
  )
endif()

set(llama_runner_deps executorch_core extension_data_loader extension_module
                      extension_tensor extension_flat_tensor extension_llm_runner
)

target_link_libraries(llama_runner PUBLIC ${llama_runner_deps})

target_link_libraries(
  llama_runner PUBLIC tokenizers
)

target_include_directories(
  llama_runner
  PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_compile_options(llama_runner PUBLIC ${_preprocessor_flag})
