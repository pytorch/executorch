# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom ops that XNNPACK passes introduce after partitioning. They never reach
the partitioner: a pass creates them inside an already lowered subgraph and a
node visitor serializes them, the reference implementation below only serves
tracing and tests.
"""

import torch.library

namespace = "et_xnnpack"
lib = torch.library.Library(namespace, "DEF")

#######################
## depth_to_space_2d ##
#######################


def depth_to_space_2d_impl(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Block-major depth to space, the channel order XNNPACK's (and TFLite's)
    DEPTH_TO_SPACE uses: input channel (by * block_size + bx) * C + c lands at
    output pixel (h * block_size + by, w * block_size + bx), channel c.
    aten.pixel_shuffle is the channel-major variant, c * block_size**2 + by *
    block_size + bx, and does not map onto XNNPACK's node directly.
    """
    n, c_in, h, w = x.shape
    bs = block_size
    c = c_in // (bs * bs)
    y = x.reshape(n, bs, bs, c, h, w)
    y = y.permute(0, 3, 4, 1, 5, 2)  # n, c, h, by, w, bx
    return y.reshape(n, c, h * bs, w * bs)


name = "depth_to_space_2d"
lib.define(f"{name}(Tensor x, int block_size) -> Tensor")
lib.impl(name, depth_to_space_2d_impl, "CompositeExplicitAutograd")
depth_to_space_2d_op = torch.ops.et_xnnpack.depth_to_space_2d
