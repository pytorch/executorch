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

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(test_env "RESOURCES_PATH=${EXECUTORCH_ROOT}/extension/llm/tokenizer/test/resources")

set(_test_srcs test_bpe_tokenizer.cpp test_tiktoken.cpp)

et_cxx_test(
  extension_llm_tokenizer_test SOURCES ${_test_srcs} EXTRA_LIBS
  extension_llm_tokenizer
)

set_property(TEST extension_llm_tokenizer_test PROPERTY ENVIRONMENT ${test_env})
