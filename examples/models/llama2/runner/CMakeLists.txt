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

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

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

list(
  APPEND _llama_runner__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/tokenizer/tiktoken.cpp
)
list(APPEND _llama_runner__srcs
     ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer/llama_tiktoken.cpp
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

# find RE2 for tokenizer, build tiktoken
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
  ${EXECUTORCH_ROOT}/extension/llm/third-party/abseil-cpp
  ${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp
)
add_subdirectory(
  ${EXECUTORCH_ROOT}/extension/llm/third-party/re2
  ${CMAKE_CURRENT_BINARY_DIR}/re2
)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

set(llama_runner_deps executorch extension_data_loader extension_module
                      extension_tensor re2::re2
)

target_link_libraries(llama_runner PUBLIC ${llama_runner_deps})

target_include_directories(
  llama_runner INTERFACE ${_common_include_directories} ${EXECUTORCH_ROOT}
)
target_compile_options(llama_runner PUBLIC ${_preprocessor_flag})
