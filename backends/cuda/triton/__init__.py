# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import all kernels to ensure @triton_op decorators are executed
# and ops are registered to torch.ops.triton namespace
from executorch.backends.cuda.triton import kernels  # noqa: F401

from executorch.backends.cuda.triton.replacement_pass import (
    ReplaceEdgeOpWithTritonOpPass,
)

__all__ = [
    "ReplaceEdgeOpWithTritonOpPass",
]
