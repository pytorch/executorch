# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared utilities (safetensors loader, json helpers, batch helpers).
add_library(qnn_toy_model_common STATIC common.cpp)
target_include_directories(
  qnn_toy_model_common PUBLIC ${_common_include_directories}
)
target_link_libraries(qnn_toy_model_common executorch_core extension_tensor)
target_compile_options(qnn_toy_model_common PUBLIC ${_common_compile_options})

function(_add_toy_model_binary target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PUBLIC ${_common_include_directories})
  target_link_libraries(
    ${target}
    qnn_toy_model_common
    qnn_executorch_backend
    executorch_core
    extension_module_static
    extension_tensor
    full_portable_ops_lib
  )
  target_compile_options(${target} PUBLIC ${_common_compile_options})
  set_target_properties(${target} PROPERTIES LINK_FLAGS "-Wl,-rpath,\\$ORIGIN")
endfunction()

_add_toy_model_binary(qnn_toy_model_trainer trainer.cpp)
_add_toy_model_binary(qnn_toy_model_inference inference.cpp)
