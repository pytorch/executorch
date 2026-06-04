# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.tosa.dialect.ops import (  # noqa F401
    activation,
    avg_pool2d,
    avg_pool2d_adaptive,
    cast_to_block_scaled,
    conv2d,
    conv3d,
    custom,
    depthwise_conv2d,
    gather,
    identity,
    matmul,
    matmul_t_block_scaled,
    max_pool2d,
    max_pool2d_adaptive,
    pad,
    reduction_ops,
    rescale,
    resize,
    scatter,
    shape_ops,
    slice,
    table,
    transpose_conv2d,
)
