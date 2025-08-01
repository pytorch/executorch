# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Presets to enable profiling in executor runner

set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_LLM ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set_overridable_option(EXECUTORCH_BUILD_XNNPACK ON)
set_overridable_option(EXECUTORCH_BUILD_DEVTOOLS ON)

# Presets to build executor runner

set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
set_overridable_option(EXECUTORCH_ENABLE_EVENT_TRACER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
