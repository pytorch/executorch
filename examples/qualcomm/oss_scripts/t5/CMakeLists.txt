# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# preprocess qnn runner src files for t5
set(_qnn_t5_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/qnn_t5_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/decoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/decoder.h
    ${CMAKE_CURRENT_LIST_DIR}/runner/encoder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/encoder.h
    ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
    ${EXECUTORCH_ROOT}/extension/llm/sampler/sampler.cpp
)

# build qnn t5 runner
add_executable(qnn_t5_runner ${_qnn_t5_runner__srcs})
target_include_directories(qnn_t5_runner PUBLIC ${_common_include_directories})

target_link_libraries(
  qnn_t5_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  tokenizers::tokenizers
)

target_compile_options(qnn_t5_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_t5_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
