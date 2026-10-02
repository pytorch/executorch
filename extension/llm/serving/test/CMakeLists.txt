# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

et_cxx_test(
  extension_llm_serving_test
  SOURCES
  serving_runtime_test.cpp
  generation_bridge_test.cpp
  prompt_preparer_test.cpp
  text_output_test.cpp
  text_generation_test.cpp
  EXTRA_LIBS
  extension_llm_serving
)

# Match the tokenizer-dependent runner test's sanitizer link workarounds.
if(APPLE)
  target_sources(
    extension_llm_serving_test
    PRIVATE ${EXECUTORCH_ROOT}/extension/llm/runner/test/lsan_stub.cpp
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_link_options(extension_llm_serving_test PRIVATE --rtlib=compiler-rt)
endif()
