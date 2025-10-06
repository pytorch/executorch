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

# The buck-based executorch_srcs.cmake setup was crossing package boundaries and
# trying to build stuff from executorch/extension/llm/runner and tokenizers.
# Just set up sources manually.
set(llama_runner_srcs runner.cpp ../tokenizer/llama_tiktoken.cpp)

if(CMAKE_TOOLCHAIN_IOS
   OR ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(llama_runner STATIC ${llama_runner_srcs})
else()
  add_library(llama_runner SHARED ${llama_runner_srcs})
endif()

if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled."
  )
endif()

set(llama_runner_deps
    executorch_core extension_data_loader extension_module extension_tensor
    extension_flat_tensor extension_llm_runner
)

target_link_libraries(llama_runner PUBLIC ${llama_runner_deps})
target_link_libraries(llama_runner PUBLIC tokenizers::tokenizers)

target_include_directories(
  llama_runner PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
                      ${EXECUTORCH_ROOT}/..
)
target_compile_options(llama_runner PUBLIC ${_preprocessor_flag})
