# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

list(TRANSFORM _extension_tensor__srcs PREPEND "${PROJECT_SOURCE_DIR}/")

cc_library(
  NAME extension_tensor
  SRCS ${_extension_tensor__srcs}
  DEPS executorch_core
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
