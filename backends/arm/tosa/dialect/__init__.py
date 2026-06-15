# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.tosa.dialect.ops import (  # noqa F401
    activation,
    argmax,
    avg_pool2d,
    avg_pool2d_adaptive,
    conv2d,
    conv3d,
    custom,
    data_layout_ops,
    depthwise_conv2d,
    fft,
    gather,
    identity,
    matmul,
    max_pool2d,
    max_pool2d_adaptive,
    reduction_ops,
    rescale,
    resize,
    scatter,
    shape_ops,
    table,
    transpose_conv2d,
    unary_elementwise,
)
