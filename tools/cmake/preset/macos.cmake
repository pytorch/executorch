# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/preset/apple_common.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/preset/pybind.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/preset/llm.cmake)

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
set_overridable_option(EXECUTORCH_COREML_BUILD_EXECUTOR_RUNNER ON)
