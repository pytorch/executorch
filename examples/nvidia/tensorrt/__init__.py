# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT delegate examples for ExecuTorch.

This module provides utilities for exporting and running models
with the TensorRT delegate on NVIDIA GPUs.
"""

from executorch.examples.nvidia.tensorrt.export import (
    get_supported_models_list,
    TENSORRT_SUPPORTED_MODELS,
)

__all__ = [
    "get_supported_models_list",
    "TENSORRT_SUPPORTED_MODELS",
]
