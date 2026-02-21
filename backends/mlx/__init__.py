#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""MLX backend for ExecuTorch - executes models on Apple Silicon using MLX."""

# Import custom_ops module to register custom ATen ops (rope, etc.)
from executorch.backends.mlx import custom_ops as _custom_ops  # noqa: F401
from executorch.backends.mlx.partitioner import MLXPartitioner

from executorch.backends.mlx.preprocess import MLXBackend

__all__ = ["MLXBackend", "MLXPartitioner"]
