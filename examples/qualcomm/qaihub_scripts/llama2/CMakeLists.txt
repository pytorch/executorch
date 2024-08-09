# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# preprocess qaihub llama2 7b runner src files

set(_qaihub_llama2_7b_runner__srcs ${_llama_runner__srcs})

list(TRANSFORM _qaihub_llama2_7b_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _qaihub_llama2_7b_runner__srcs EXCLUDE REGEX ".*runner.cpp*$")
list(PREPEND _qaihub_llama2_7b_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qaihub_llama2_7b_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/io_memory.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/io_memory.h
)

# build qaihub llama2 7b runner
add_executable(qaihub_llama2_7b_runner ${_qaihub_llama2_7b_runner__srcs})
target_include_directories(qaihub_llama2_7b_runner
  PUBLIC ${_common_include_directories}
)
target_link_libraries(qaihub_llama2_7b_runner
  qnn_executorch_backend
  executorch_no_prim_ops
  extension_data_loader
  extension_module
  gflags
)
target_compile_options(qaihub_llama2_7b_runner
  PUBLIC ${_common_compile_options}
)
