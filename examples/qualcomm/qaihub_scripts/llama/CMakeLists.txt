# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# preprocess qaihub runner src files for llama2,3
set(_qaihub_llama_runner__srcs ${_llama_runner__srcs})
list(TRANSFORM _qaihub_llama_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _qaihub_llama_runner__srcs EXCLUDE REGEX ".*(/runner/).*")
list(
  PREPEND
  _qaihub_llama_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/io_memory.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/io_memory.h
)

# preprocess qaihub llama2 7b runner src files
set(_qaihub_llama2_7b_runner__srcs ${_qaihub_llama_runner__srcs})

list(PREPEND _qaihub_llama2_7b_runner__srcs
     ${CMAKE_CURRENT_LIST_DIR}/llama2/qaihub_llama2_7b_runner.cpp
)

# build qaihub llama2 7b runner
add_executable(qaihub_llama2_7b_runner ${_qaihub_llama2_7b_runner__srcs})
target_include_directories(
  qaihub_llama2_7b_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(
  qaihub_llama2_7b_runner
  qnn_executorch_backend
  executorch_no_prim_ops
  extension_data_loader
  extension_module
  extension_tensor
  gflags
  re2::re2
)
target_compile_options(
  qaihub_llama2_7b_runner PUBLIC ${_common_compile_options}
)
set_target_properties(
  qaihub_llama2_7b_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# preprocess qaihub llama3 8b runner src files
set(_qaihub_llama3_8b_runner__srcs ${_qaihub_llama_runner__srcs})

list(PREPEND _qaihub_llama3_8b_runner__srcs
     ${CMAKE_CURRENT_LIST_DIR}/llama3/qaihub_llama3_8b_runner.cpp
)

# Adding a compile option to differentiate llama2 with llama3 logic
list(APPEND _common_compile_options -DQAIHUB_LLAMA3_RUNNER)

# find RE2 for tokenizer
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/third-party/abseil-cpp
  ${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp
)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/third-party/re2
  ${CMAKE_CURRENT_BINARY_DIR}/re2
)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

list(
  APPEND _qaihub_llama3_8b_runner__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../extension/llm/tokenizer/tiktoken.cpp
)
list(
  APPEND
  _qaihub_llama3_8b_runner__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../models/llama2/tokenizer/llama_tiktoken.cpp
)

# build qaihub llama3 8b runner
add_executable(qaihub_llama3_8b_runner ${_qaihub_llama3_8b_runner__srcs})
target_include_directories(
  qaihub_llama3_8b_runner PUBLIC ${_common_include_directories}
)

target_link_libraries(
  qaihub_llama3_8b_runner
  qnn_executorch_backend
  executorch_no_prim_ops
  extension_data_loader
  extension_module
  extension_tensor
  gflags
  re2::re2
)
target_compile_options(
  qaihub_llama3_8b_runner PUBLIC ${_common_compile_options}
)
set_target_properties(
  qaihub_llama3_8b_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
