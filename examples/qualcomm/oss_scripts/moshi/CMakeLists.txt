# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_qnn_mimi_decoder_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/qnn_mimi_decoder_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
)

# build mimi decoder runner
add_executable(qnn_mimi_decoder_runner ${_qnn_mimi_decoder_runner__srcs})
target_include_directories(
  qnn_mimi_decoder_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(
  qnn_mimi_decoder_runner
  qnn_executorch_backend
  executorch_core
  extension_module
  extension_data_loader
  extension_flat_tensor
  gflags
)

target_compile_options(qnn_llama_runner PUBLIC ${_common_compile_options})

set_target_properties(
  qnn_mimi_decoder_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
