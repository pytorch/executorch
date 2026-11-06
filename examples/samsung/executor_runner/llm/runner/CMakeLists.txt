# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tools/cmake/Utils.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tools/cmake/Codegen.cmake)

# stb_image: same pinned revision as examples/models/gemma3. deprecated/ holds
# stb_image_resize.h.
include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_MakeAvailable(stb)

set(_enn_llm_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/decoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/prefiller.cpp
    ${CMAKE_CURRENT_LIST_DIR}/io_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rotary_embedding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vision_encoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stb_image_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../examples/models/llama/tokenizer/llama_tiktoken.cpp
)

target_include_directories(
  extension_module INTERFACE ${_common_include_directories}
)

add_library(lib_tokenizer STATIC IMPORTED)
set_target_properties(
  lib_tokenizer
  PROPERTIES
    IMPORTED_LOCATION
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../../../build_samsung_android/lib/libtokenizers.a"
)

add_library(lib_extension_llm_runner STATIC IMPORTED)
set_target_properties(
  lib_extension_llm_runner
  PROPERTIES
    IMPORTED_LOCATION
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../../../build_samsung_android/extension/llm/runner/libextension_llm_runner.a"
)

if(CMAKE_TOOLCHAIN_IOS
   OR ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(enn_llm_runner STATIC ${_enn_llm_runner__srcs})
else()
  add_library(enn_llm_runner SHARED ${_enn_llm_runner__srcs})
endif()

# For extension_llm_runner
if(NOT TARGET extension_llm_runner)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/runner
    ${CMAKE_CURRENT_BINARY_DIR}/../../../../../extension/llm/runner
  )
endif()

set(enn_llm_runner_deps
    executorch_core extension_data_loader extension_module extension_tensor
    extension_flat_tensor lib_extension_llm_runner
)

target_compile_options(enn_llm_runner PRIVATE ${_llm_compile_options})
target_link_libraries(enn_llm_runner PUBLIC ${enn_llm_runner_deps})
target_link_libraries(enn_llm_runner PUBLIC tokenizers::tokenizers)
target_link_libraries(enn_llm_runner PUBLIC lib_tokenizer)

target_include_directories(
  enn_llm_runner
  PUBLIC
    ${EXECUTORCH_ROOT}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/tokenizers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/tokenizers/third-party/re2
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/tokenizers/third-party/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/tokenizers/third-party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../examples/models/llama/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../examples/samsung/executor_runner/llm/runner
    ${stb_SOURCE_DIR}
    ${stb_SOURCE_DIR}/deprecated
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/runner
)
