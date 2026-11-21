# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# build qnn smolvla runner
add_executable(
  qnn_smolvla_runner ${CMAKE_CURRENT_LIST_DIR}/qnn_smolvla_runner.cpp
)
target_include_directories(
  qnn_smolvla_runner PUBLIC ${_common_include_directories}
)

target_link_libraries(
  qnn_smolvla_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_module
  extension_tensor
  full_portable_ops_lib
  gflags
)

target_compile_options(qnn_smolvla_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_smolvla_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
