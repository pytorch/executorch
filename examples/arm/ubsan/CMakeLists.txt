# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(executorch_ubsan STATIC ubsan_runtime.c)

target_compile_features(executorch_ubsan PRIVATE c_std_11)

target_compile_options(executorch_ubsan PRIVATE -fno-sanitize=undefined)

set_target_properties(executorch_ubsan PROPERTIES OUTPUT_NAME "ubsan")

install(
  TARGETS executorch_ubsan
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION lib
)
