# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/preset/pybind-macos.cmake)

set_overridable_option(EXECUTORCH_BUILD_COREML ON)
set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
set_overridable_option(EXECUTORCH_XNNPACK_SHARED_WORKSPACE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_APPLE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
