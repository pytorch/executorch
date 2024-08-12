# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for LLaVa runner.
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
# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# build llava_runner library
set(_llava_runner__srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/llava_runner.cpp"
    "${EXECUTORCH_ROOT}/extension/llm/sampler/sampler.cpp"
    "${EXECUTORCH_ROOT}/extension/llm/tokenizer/bpe_tokenizer.cpp"
)

# extension llm runner lib
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/runner
    ${CMAKE_CURRENT_BINARY_DIR}/../../../../extension/llm/runner
)

add_library(llava_runner STATIC ${_llava_runner__srcs})

set(llava_runner_deps executorch extension_module extension_data_loader extension_llm_runner)

target_link_libraries(llava_runner PUBLIC ${llava_runner_deps})

target_include_directories(
    llava_runner INTERFACE ${_common_include_directories} ${EXECUTORCH_ROOT}
)
