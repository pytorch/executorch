# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# preprocess qaihub_stable_diffusion_runner_src files
set(_qaihub_stable_diffusion_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/qaihub_stable_diffusion_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
)

# build qaihub_stable_diffusion_runner
add_executable(
  qaihub_stable_diffusion_runner ${_qaihub_stable_diffusion_runner__srcs}
)
target_include_directories(
  qaihub_stable_diffusion_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(
  qaihub_stable_diffusion_runner
  qnn_executorch_backend
  executorch_no_prim_ops
  extension_data_loader
  extension_module
  extension_tensor
  gflags
  re2::re2
)
target_compile_options(
  qaihub_stable_diffusion_runner PUBLIC ${_common_compile_options}
)
set_target_properties(
  qaihub_stable_diffusion_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
