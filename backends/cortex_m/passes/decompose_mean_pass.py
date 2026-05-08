# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue

from torch._ops import OpOverload
from torch.fx.node import Argument


class DecomposeMeanPass(ExportPass):
    """
    Decomposes AdaptiveAvgPool2d into AvgPool2d operation.
    """

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op == torch.ops.aten.adaptive_avg_pool2d.default:
            op = torch.ops.aten.avg_pool2d.default
            input_tensor = cast(torch.Tensor, args[0])
            shape = input_tensor.data.shape
            stride = [1, 1]
            kernel_size = [shape[-2], shape[-1]]
            args = (args[0], kernel_size, stride, [0, 0], 0, 0)

        return super().call_operator(op, args, kwargs, meta)
