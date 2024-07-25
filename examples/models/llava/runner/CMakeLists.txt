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

# extract_sources(${EXECUTORCH_SRCS_FILE})

# include(${EXECUTORCH_SRCS_FILE})

# build multimodal_runner library
set(_multimodal_runner__srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/multimodal_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../llama2/sampler/sampler.cpp"
    "${EXECUTORCH_ROOT}/extension/llm/tokenizer/bpe_tokenizer.cpp"
)

target_include_directories(
  extension_module INTERFACE ${_common_include_directories}
)

if(EXECUTORCH_USE_TIKTOKEN)
  list(APPEND _multimodal_runner__srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/tokenizer/tiktoken.cpp
  )
  list(APPEND _multimodal_runner__srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer/llama_tiktoken.cpp
  )
  set(_preprocessor_flag -DET_USE_TIKTOKEN)
endif()


add_library(multimodal_runner STATIC ${_multimodal_runner__srcs})

set(multimodal_runner_deps executorch extension_module extension_data_loader)

target_link_libraries(multimodal_runner PUBLIC ${multimodal_runner_deps})

target_include_directories(
    multimodal_runner INTERFACE ${_common_include_directories} ${EXECUTORCH_ROOT}
)
target_compile_options(multimodal_runner PUBLIC ${_preprocessor_flag})
