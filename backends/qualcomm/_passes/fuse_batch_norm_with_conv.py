# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.qualcomm._passes.canonicalize_conv import ConvParamIdx
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.exir import ExportedProgram


class FuseBatchNormWithConv(FuseBatchNormWithConvPass):
    """Folds BatchNorm into the preceding convolution, except a transposed one.

    fuse_conv_bn_weights() scales dim 0 of the weight, which is the output
    channel for a regular convolution but the *input* channel for a transposed
    one, whose weight is [in, out/groups, *kernel]. Folding there scales the
    wrong axis: it raises when in != out, and silently returns wrong values
    when they happen to match. Transposed convolutions therefore keep their
    standalone BatchNorm.
    """

    @staticmethod
    def can_fuse(
        conv: torch.fx.Node, bn: torch.fx.Node, program: ExportedProgram
    ) -> bool:
        if (
            len(conv.args) > ConvParamIdx.TRANSPOSED
            and conv.args[ConvParamIdx.TRANSPOSED]
        ):
            return False
        return FuseBatchNormWithConvPass.can_fuse(conv, bn, program)
