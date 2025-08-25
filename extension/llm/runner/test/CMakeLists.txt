# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs
    test_generation_config.cpp test_text_llm_runner.cpp test_text_prefiller.cpp
    test_text_decoder_runner.cpp test_multimodal_input.cpp
)

# Add LSan stub for Apple platforms
if(APPLE)
  list(APPEND _test_srcs lsan_stub.cpp)
endif()

et_cxx_test(
  test_runner SOURCES ${_test_srcs} EXTRA_LIBS executorch extension_llm_runner
)

# Override sanitizer to this issue:
# https://github.com/abseil/abseil-cpp/issues/841 Root issue:
# https://github.com/llvm/llvm-project/issues/16778
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_link_options(test_runner PUBLIC --rtlib=compiler-rt)
endif()
