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

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# build llava_runner library
set(_llava_runner__srcs "${CMAKE_CURRENT_SOURCE_DIR}/llava_runner.cpp")

if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled."
  )
endif()

add_library(llava_runner STATIC ${_llava_runner__srcs})
target_include_directories(llava_runner PRIVATE ${_common_include_directories})

set(llava_runner_deps
    executorch_core extension_data_loader extension_llm_runner extension_module
    extension_tensor extension_flat_tensor
)

target_link_libraries(llava_runner PUBLIC ${llava_runner_deps})
