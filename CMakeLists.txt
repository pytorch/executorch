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

add_library(tokenizers STATIC
  src/sentencepiece.cpp
  src/tiktoken.cpp
  src/bpe_tokenizer_base.cpp)

# Using abseil from sentencepiece/third_party
target_include_directories(
  tokenizers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  third-party/sentencepiece/src
  third-party/sentencepiece
  third-party/re2
  third-party/json/single_include)

target_link_libraries(tokenizers PUBLIC sentencepiece-static re2::re2)

# Build test
if(TOKENIZERS_BUILD_TEST)
    enable_testing()
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

  file(GLOB test_source_files test/test_*.cpp)
  foreach(test_source_file ${test_source_files})
      get_filename_component(test_name ${test_source_file} NAME_WE)
      message(STATUS "Configuring unit test ${test_name}")
      add_executable(${test_name} ${test_source_file})
      target_include_directories(${test_name} PRIVATE
        GTEST_INCLUDE_PATH
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third-party/sentencepiece
        ${CMAKE_SOURCE_DIR}/third-party/re2
        ${CMAKE_SOURCE_DIR}/third-party/json/single_include
        )
      target_link_libraries(${test_name} gtest_main tokenizers)
      target_compile_definitions(${test_name} PRIVATE RESOURCES_PATH="${CMAKE_SOURCE_DIR}/test/resources")
      add_test(${test_name} "${test_name}")
  endforeach()
endif()
