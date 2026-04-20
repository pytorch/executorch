# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.tosa.dialect.ops import (  # noqa F401
    avg_pool2d,
    conv2d,
    conv3d,
    custom,
    depthwise_conv2d,
    gather,
    matmul,
    max_pool2d,
    pad,
    rescale,
    resize,
    scatter,
    shape_ops,
    slice,
    table,
    transpose,
    transpose_conv2d,
)
