# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(executorch_asan STATIC asan_runtime.c)

target_compile_features(executorch_asan PRIVATE c_std_11)

target_compile_options(
  executorch_asan PRIVATE -fno-sanitize=address -fno-sanitize=kernel-address
                          -fno-sanitize=undefined
)

set_target_properties(executorch_asan PROPERTIES OUTPUT_NAME "asan")

install(
  TARGETS executorch_asan
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION lib
)
