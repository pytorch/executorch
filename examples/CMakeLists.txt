# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(bundled_executor_runner
               bundled_executor_runner/bundled_executor_runner.cpp)
target_compile_options(executorch PUBLIC -DET_EVENT_TRACER_ENABLED)
target_compile_options(portable_ops_lib PUBLIC -DET_EVENT_TRACER_ENABLED)
target_link_libraries(
  bundled_executor_runner
  executorch
  gflags
  etdump
  extension_data_loader
  flatcc
  bundled_program
  portable_ops_lib)
