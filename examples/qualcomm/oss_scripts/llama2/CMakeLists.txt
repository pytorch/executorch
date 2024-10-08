# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_qnn_llama_runner__srcs ${_llama_runner__srcs})

# preprocess qnn llama runner src files
list(TRANSFORM _qnn_llama_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _qnn_llama_runner__srcs EXCLUDE REGEX ".*(/runner/).*")
list(
  PREPEND
  _qnn_llama_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
)

# build qnn llama runner
add_executable(qnn_llama_runner ${_qnn_llama_runner__srcs})
target_include_directories(
  qnn_llama_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(
  qnn_llama_runner
  qnn_executorch_backend
  full_portable_ops_lib
  extension_data_loader
  extension_module
  extension_tensor
  gflags
  re2::re2
)
target_compile_options(qnn_llama_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_llama_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
