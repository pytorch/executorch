# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include("${CMAKE_CURRENT_LIST_DIR}/ExecuTorchNativeTargets.cmake")
get_filename_component(
  executorch_native_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
