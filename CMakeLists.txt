# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
option(TOKENIZERS_BUILD_TOOLS "Build tools" OFF)
option(SUPPORT_REGEX_LOOKAHEAD
       "Support regex lookahead patterns (requires PCRE2)" OFF)

include(Utils.cmake)
# Ignore weak attribute warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)

set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/abseil-cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/re2)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/sentencepiece)

set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

file(GLOB tokenizers_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(tokenizers_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bpe_tokenizer_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hf_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llama2c_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pre_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/re2_regex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/regex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sentencepiece.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tiktoken.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/token_decoder.cpp)

file(GLOB unicode_source_files
     ${CMAKE_CURRENT_SOURCE_DIR}/third-party/llama.cpp-unicode/src/*.cpp)
add_library(tokenizers STATIC ${tokenizers_source_files}
                              ${unicode_source_files})

# Using abseil from sentencepiece/third_party
target_include_directories(
  tokenizers
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
         ${CMAKE_CURRENT_SOURCE_DIR}/third-party/sentencepiece
         ${CMAKE_CURRENT_SOURCE_DIR}/third-party/sentencepiece/src
         ${CMAKE_CURRENT_SOURCE_DIR}/third-party/re2
         ${CMAKE_CURRENT_SOURCE_DIR}/third-party/json/single_include
         ${CMAKE_CURRENT_SOURCE_DIR}/third-party/llama.cpp-unicode/include)
target_link_libraries(tokenizers PUBLIC sentencepiece-static re2::re2)

if(SUPPORT_REGEX_LOOKAHEAD OR TOKENIZERS_BUILD_TEST)
  set(PCRE2_BUILD_PCRE2_8 ON)
  set(PCRE2_BUILD_PCRE2_16 OFF)
  set(PCRE2_BUILD_PCRE2_32 OFF)
  set(PCRE2_BUILD_TESTS OFF)
  set(PCRE2_BUILD_PCRE2GREP OFF)
  set(PCRE2_BUILD_PCRE2TEST OFF)
  set(PCRE2_BUILD_PCRE2GPERF OFF)
  set(PCRE2_BUILD_DOCS OFF)
  set(PCRE2_BUILD_LIBPCRE2_PDB OFF)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pcre2)
  add_library(
    regex_lookahead STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pcre2_regex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/regex_lookahead.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/std_regex.cpp)
  target_link_libraries(regex_lookahead PUBLIC pcre2-8)
  target_include_directories(
    regex_lookahead PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pcre2/src)
  target_link_options_shared_lib(regex_lookahead)
endif()

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

  file(GLOB test_source_files ${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cpp)

  set(test_env "RESOURCES_PATH=${CMAKE_CURRENT_SOURCE_DIR}/test/resources")
  foreach(test_source_file ${test_source_files})
    get_filename_component(test_name ${test_source_file} NAME_WE)
    message(STATUS "Configuring unit test ${test_name}")
    add_executable(${test_name} ${test_source_file})
    target_include_directories(
      ${test_name}
      PRIVATE GTEST_INCLUDE_PATH
              ${CMAKE_CURRENT_SOURCE_DIR}/include
              ${CMAKE_CURRENT_SOURCE_DIR}/third-party/sentencepiece
              ${CMAKE_CURRENT_SOURCE_DIR}/third-party/re2
              ${CMAKE_CURRENT_SOURCE_DIR}/third-party/json/single_include)
    target_link_libraries(${test_name} gtest_main GTest::gmock tokenizers
                          regex_lookahead)
    add_test(${test_name} "${test_name}")
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT ${test_env})
  endforeach()
endif()

# Build tools
if(TOKENIZERS_BUILD_TOOLS)
  add_subdirectory(examples/tokenize_tool)
endif()
