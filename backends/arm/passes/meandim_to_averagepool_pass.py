# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, cast, Dict, Tuple

import torch.fx

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue

Argument = Any


class ConvertMeanDimToAveragePool(ExportPass):
    """
    Replace a mean operation with dim = [-1, -2] and keep_dim = True with an average pool operation.
    """

    def call_operator(
        self,
        op: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op != exir_ops.edge.aten.mean.dim:
            return super().call_operator(op, args, kwargs, meta)

        input_value = cast(ProxyValue, args[0])
        dim = cast(list, args[1])
        keep_dim = cast(bool, args[2]) if len(args) > 2 else False

        # averagepool2d gets converted to a mean operation with dim = [-1, -2] and keep_dim = True
        # so check the dim argument for this case
        if dim == [-1, -2] and keep_dim is True:
            # Given the shape format of input is (N, C, H, W)
            kernel_size = [
                input_value.to_tensor().size()[2],
                input_value.to_tensor().size()[3],
            ]
            stride = [1, 1]
            return super().call_operator(
                exir_ops.edge.aten.avg_pool2d.default,
                (input_value, kernel_size, stride),
                {},
                meta,
            )
        else:
            return super().call_operator(op, args, kwargs, meta)
