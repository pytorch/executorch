# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple

import torch
from executorch.exir.pass_base import ExportPass
from torch._export.pass_base import Argument
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue


class ConvertBinaryOpsWithScalar(ExportPass):
    """
    Replace binary ops with scalar into binary ops with tensor.
    Since torch.ops.aten.xxx.Scalar will not generate a placeholder node
    for scalar after to_edge.
    """

    binary_ops_with_scalar = {
        torch.ops.aten.add.Scalar: torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Scalar: torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Scalar: torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Scalar: torch.ops.aten.mul.Tensor,
    }

    def __init__(self):
        super(ConvertBinaryOpsWithScalar, self).__init__()

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return super().call_operator(
            self.binary_ops_with_scalar.get(op, op), args, kwargs, meta
        )
