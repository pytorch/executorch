# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

cmake_minimum_required(VERSION 3.19)
project(mediatek_example)

message(CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/third-party/json/single_include
)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

# Find prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)
find_package(gflags REQUIRED)

link_directories(${EXECUTORCH_ROOT}/cmake-android-out/lib)

if(${ANDROID})
  message("Build MTK Android Examples")

  get_filename_component(
    EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
  )
  set(_mtk_executor_runner__srcs ${_executor_runner__srcs})
  list(TRANSFORM _mtk_executor_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
  list(FILTER _mtk_executor_runner__srcs EXCLUDE REGEX ".*executor_runner.cpp$")
  list(PREPEND _mtk_executor_runner__srcs
       ${CMAKE_CURRENT_LIST_DIR}/executor_runner/mtk_executor_runner.cpp
  )

  add_executable(mtk_executor_runner ${_mtk_executor_runner__srcs})

  target_include_directories(
    mtk_executor_runner
    PUBLIC ${_common_include_directories}
           ${EXECUTORCH_ROOT}/cmake-android-out/third-party/gflags/include
  )

  target_link_libraries(
    mtk_executor_runner
    ${_executor_runner_libs}
    executorch
    neuron_backend
    executorch_core
    extension_evalue_util
    extension_runner_util
    gflags
  )
  target_compile_options(mtk_executor_runner PUBLIC ${_common_compile_options})
  add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

  set(_mtk_oss_executor_runner__srcs ${_executor_runner__srcs})
  list(TRANSFORM _mtk_oss_executor_runner__srcs
       PREPEND "${EXECUTORCH_SOURCE_DIR}/"
  )
  list(FILTER _mtk_oss_executor_runner__srcs EXCLUDE REGEX
       ".*executor_runner.cpp$"
  )
  list(PREPEND _mtk_oss_executor_runner__srcs
       ${CMAKE_CURRENT_LIST_DIR}/executor_runner/mtk_oss_executor_runner.cpp
  )

  add_executable(mtk_oss_executor_runner ${_mtk_oss_executor_runner__srcs})

  target_include_directories(
    mtk_oss_executor_runner
    PUBLIC ${_common_include_directories}
           ${EXECUTORCH_ROOT}/cmake-android-out/third-party/gflags/include
  )

  target_link_libraries(
    mtk_oss_executor_runner ${_executor_runner_libs} extension_module
    executorch neuron_backend gflags
  )
  target_compile_options(
    mtk_oss_executor_runner PUBLIC ${_common_compile_options}
  )

  set(_mtk_llama_executor_runner__srcs ${_mtk_executor_runner__srcs})
  list(FILTER _mtk_llama_executor_runner__srcs EXCLUDE REGEX
       ".*executor_runner.cpp$"
  )
  list(PREPEND _mtk_llama_executor_runner__srcs
       ${CMAKE_CURRENT_LIST_DIR}/executor_runner/mtk_llama_executor_runner.cpp
  )
  # Build ABSL and RE2
  set(EXTENSIONS_LLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../extension/llm)
  set(THIRD_PARTY_ABSL_DIR
      ${EXTENSIONS_LLM_DIR}/tokenizers/third-party/abseil-cpp
  )
  set(THIRD_PARTY_RE2_DIR ${EXTENSIONS_LLM_DIR}/tokenizers/third-party/re2)
  set(THIRD_PARTY_JSON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/json)
  set(THIRD_PARTY_UNICODE_DIR
      ${EXTENSIONS_LLM_DIR}/tokenizers/third-party/llama.cpp-unicode
  )
  set(THIRD_PARTY_PCRE2_DIR ${EXTENSIONS_LLM_DIR}/tokenizers/third-party/pcre2)
  set(ABSL_ENABLE_INSTALL ON)
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(
    ${THIRD_PARTY_ABSL_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/abseil
  )
  add_subdirectory(
    ${THIRD_PARTY_RE2_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/re2
  )
  add_subdirectory(
    ${THIRD_PARTY_JSON_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/json
  )
  add_subdirectory(
    ${THIRD_PARTY_UNICODE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/llama.cpp-unicode
  )
  add_subdirectory(
    ${THIRD_PARTY_PCRE2_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/pcre2
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

  # Build tokenizers
  set(SUPPORT_REGEX_LOOKAHEAD ON)
  set(LLAMA2_TOKENIZER_DIR ${EXTENSIONS_LLM_DIR}/tokenizers)
  add_library(tokenizer STATIC)
  target_include_directories(
    tokenizer
    PUBLIC ${_common_include_directories}
           ${THIRD_PARTY_ABSL_DIR}
           ${THIRD_PARTY_RE2_DIR}
           ${LLAMA2_TOKENIZER_DIR}/include
           ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/pcre2
           ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
           ${THIRD_PARTY_JSON_DIR}
           ${THIRD_PARTY_UNICODE_DIR}/include
           ${THIRD_PARTY_PCRE2_DIR}
  )
  target_link_libraries(tokenizer PRIVATE re2::re2)

  target_sources(
    tokenizer
    PRIVATE
      ${LLAMA2_TOKENIZER_DIR}/src/tiktoken.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/llama2c_tokenizer.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/regex.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/bpe_tokenizer_base.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/re2_regex.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/hf_tokenizer.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/pre_tokenizer.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/token_decoder.cpp
      ${LLAMA2_TOKENIZER_DIR}/src/normalizer.cpp
      ${LLAMA2_TOKENIZER_DIR}/third-party/llama.cpp-unicode/src/unicode.cpp
      ${LLAMA2_TOKENIZER_DIR}/third-party/llama.cpp-unicode/src/unicode-data.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../models/llama/tokenizer/llama_tiktoken.cpp
  )

  # Add support for regex_lookahead
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

  # Set the INTERFACE_INCLUDE_DIRECTORIES property for pcre2-8-static
  set_target_properties(
    pcre2-8-static
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/tokenizers/third-party/pcre2>
  )
  add_library(
    regex_lookahead STATIC
    ${LLAMA2_TOKENIZER_DIR}/src/pcre2_regex.cpp
    ${LLAMA2_TOKENIZER_DIR}/src/regex_lookahead.cpp
    ${LLAMA2_TOKENIZER_DIR}/src/std_regex.cpp
  )
  add_library(tokenizer::regex_lookahead ALIAS regex_lookahead)
  target_link_libraries(regex_lookahead PUBLIC pcre2-8-static)
  target_include_directories(
    regex_lookahead
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(tokenizer PUBLIC regex_lookahead)
  install(
    TARGETS regex_lookahead pcre2-8-static
    EXPORT tokenizers-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  # Include directory for neuron headers
  include_directories(
    BEFORE ${_common_include_directories}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/mediatek/runtime/include
  )

  # Build Llama Executor static library
  add_subdirectory(executor_runner/llama_runner)

  # Build Llama Executor Runner
  add_executable(mtk_llama_executor_runner ${_mtk_llama_executor_runner__srcs})

  target_link_libraries(
    mtk_llama_executor_runner ${_executor_runner_libs} neuron_backend gflags
    mtk_llama_executor_lib
  )
  target_link_libraries(
    mtk_llama_executor_runner tokenizer
    $<LINK_LIBRARY:WHOLE_ARCHIVE,regex_lookahead>
  )
  target_compile_options(
    mtk_llama_executor_runner PUBLIC ${_common_compile_options}
  )
endif()
