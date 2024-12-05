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

# Ignore weak attribute warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third-party/abseil-cpp)
add_subdirectory(third-party/re2)
add_subdirectory(third-party/sentencepiece)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

add_library(tokenizers STATIC src/sentencepiece.cpp src/tiktoken.cpp)

# Using abseil from sentencepiece/third_party
target_include_directories(
  tokenizers PUBLIC third-party/sentencepiece/src third-party/sentencepiece
                    include third-party/re2)

target_link_libraries(tokenizers PUBLIC sentencepiece-static re2::re2)

# Build test
if(TOKENIZERS_BUILD_TEST)
  include(FetchContent)
  # CMAKE
  FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
  )
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  set(ENV{RESOURCES_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/test/resources)
  add_executable(sentencepiece_test test/test_sentencepiece.cpp)
  target_include_directories(
    sentencepiece_test
    PUBLIC third-party/sentencepiece/src third-party/sentencepiece include
           GTEST_INCLUDE_PATH)
  target_link_libraries(sentencepiece_test PUBLIC tokenizers gtest_main)

  # tiktoken tests
  add_executable(tiktoken_test test/test_base64.cpp)
  target_include_directories(tiktoken_test PUBLIC include GTEST_INCLUDE_PATH)
  target_link_libraries(tiktoken_test PUBLIC gtest_main)
endif()
