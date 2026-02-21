# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MLX delegate preset - builds ExecuTorch with MLX backend for Apple Silicon

# Core ExecuTorch options
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_MODULE ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM ON)
set_overridable_option(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON)
set_overridable_option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON)

# Build the MLX delegate
set_overridable_option(EXECUTORCH_BUILD_MLX ON)
