# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# adapted from examples/models/llama/runner/CMakeLists.txt

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

list(TRANSFORM _llama_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

target_include_directories(
  extension_module INTERFACE ${_common_include_directories}
)

list(
  PREPEND
  _llama_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/io_manager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/io_manager.h
)

# build qnn llama runner
if(CMAKE_TOOLCHAIN_IOS
   OR ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(llama_runner STATIC ${_llama_runner__srcs})
else()
  add_library(llama_runner SHARED ${_llama_runner__srcs})
endif()

set(llama_runner_deps executorch extension_data_loader extension_module
                      extension_tensor qnn_executorch_backend
)

target_link_libraries(llama_runner PUBLIC ${llama_runner_deps})

target_include_directories(
  llama_runner
  INTERFACE ${_common_include_directories}
)

# Include tokenizers dependency
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
  ${EXECUTORCH_ROOT}/extension/llm/tokenizers
  ${CMAKE_CURRENT_BINARY_DIR}/tokenizers
)
target_link_libraries(
  llama_runner PUBLIC tokenizers
)

target_include_directories(
  llama_runner
  PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_compile_options(llama_runner PUBLIC ${_preprocessor_flag})
