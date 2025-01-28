# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# model sharding with custom op
set(CUSTOM_OP_SRCS_FILE 
  "${EXECUTORCH_SOURCE_DIR}/extension/llm/custom_ops/op_fallback.cpp"
)
add_library(custom_ops ${CUSTOM_OP_SRCS_FILE})
target_include_directories(custom_ops PUBLIC "${_common_include_directories}")
target_include_directories(
  custom_ops PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../include"
)
target_link_libraries(
  custom_ops PUBLIC full_portable_ops_lib
)
target_link_options_shared_lib(custom_ops)

# preprocess qnn runner src files for llama3.2
set(_llama3_2_runner__srcs ${_llama_runner__srcs})
list(TRANSFORM _llama3_2_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _llama3_2_runner__srcs EXCLUDE REGEX ".*(/runner/).*")
list(
  PREPEND
  _llama3_2_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama3_2_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/io_memory.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/io_memory.h
)

list(
  APPEND _llama3_2_runner__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/tokenizer/tiktoken.cpp
)
list(
  APPEND
  _llama3_2_runner__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../models/llama/tokenizer/llama_tiktoken.cpp
)

# build qnn llama3.2 1b runner
add_executable(qnn_llama3_2_runner ${_llama3_2_runner__srcs})
target_include_directories(
  qnn_llama3_2_runner PUBLIC ${_common_include_directories}
)

target_link_libraries(
  qnn_llama3_2_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_module
  extension_tensor
  gflags
  re2::re2
  custom_ops
)
target_compile_options(
  qnn_llama3_2_runner PUBLIC ${_common_compile_options}
)
set_target_properties(
  qnn_llama3_2_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
