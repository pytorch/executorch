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
       "Support regex lookahead patterns (requires PCRE2)" OFF
)

# Include CMakePackageConfigHelpers for configure_package_config_file
include(CMakePackageConfigHelpers)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/token_decoder.cpp
)

file(GLOB unicode_source_files
     ${CMAKE_CURRENT_SOURCE_DIR}/third-party/llama.cpp-unicode/src/*.cpp
)
add_library(
  tokenizers STATIC ${tokenizers_source_files} ${unicode_source_files}
)

# Using abseil from sentencepiece/third_party
target_include_directories(
  tokenizers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/sentencepiece>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/sentencepiece/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/re2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/json/single_include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/llama.cpp-unicode/include>
)
target_link_libraries(tokenizers PUBLIC sentencepiece-static re2::re2)

if(SUPPORT_REGEX_LOOKAHEAD)
  set(PCRE2_STATIC_PIC ON)
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

  # Set the INTERFACE_INCLUDE_DIRECTORIES property for pcre2-8-static
  set_target_properties(
    pcre2-8-static
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/pcre2/src>
  )
  add_library(
    regex_lookahead STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pcre2_regex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/regex_lookahead.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/std_regex.cpp
  )
  target_link_libraries(regex_lookahead PUBLIC pcre2-8)
  target_include_directories(
    regex_lookahead
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third-party/pcre2/src>
  )
  target_link_options_shared_lib(regex_lookahead)
  target_link_libraries(tokenizers PUBLIC regex_lookahead)
  install(
    TARGETS regex_lookahead pcre2-8-static
    EXPORT tokenizers-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

# Build tools
if(TOKENIZERS_BUILD_TOOLS)
  add_subdirectory(examples/tokenize_tool)
endif()

# Installation rules
include(GNUInstallDirs)

# Install the library and its dependencies
install(
  TARGETS tokenizers re2 sentencepiece-static
  EXPORT tokenizers-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install header files
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third-party/json/single_include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)

# Install the CMake config files
install(
  EXPORT tokenizers-targets
  FILE tokenizers-targets.cmake
  NAMESPACE tokenizers::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tokenizers
)

# Generate and install the config file
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tokenizers-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/tokenizers-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tokenizers
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tokenizers-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tokenizers
)
