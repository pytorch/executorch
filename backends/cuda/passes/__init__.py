# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cuda.passes.fuse_int4_quant_matmul import (  # noqa: F401
    FuseInt4WeightOnlyQuantMatmulPass,
)

__all__ = [
    "FuseInt4WeightOnlyQuantMatmulPass",
]
