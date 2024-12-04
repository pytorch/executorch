# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the LICENSE
# file in the root directory of this source tree.

#
# Build tokenizers.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

project(Tokenizers)

option(TOKENIZERS_BUILD_TEST "Build tests" OFF)

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third-party/sentencepiece)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

add_library(tokenizers STATIC src/sentencepiece.cpp)

# Using abseil from sentencepiece/third_party
target_include_directories(tokenizers PUBLIC third-party/sentencepiece/src
                                            third-party/sentencepiece include)

target_link_libraries(tokenizers PUBLIC sentencepiece-static)

# Build test
if(TOKENIZERS_BUILD_TEST)
  find_package(GTest REQUIRED)
  set(ENV{RESOURCES_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/test/resources)
  add_executable(sentencepiece_test test/test_sentencepiece.cpp)
  target_include_directories(
    sentencepiece_test PUBLIC third-party/sentencepiece/src
                              third-party/sentencepiece include)
  target_link_libraries(sentencepiece_test PUBLIC tokenizers GTest::GTest
                                                  GTest::Main)
endif()
