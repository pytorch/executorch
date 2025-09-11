# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch._export.pass_base import Argument
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue


class ReplaceOpsWithScalar(ExportPass):
    # Replace binary ops with scalar into binary ops with tensor.
    # Ops list below.
    _ops_with_scalar = {
        exir_ops.edge.aten.add.Scalar: exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Scalar: exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.div.Scalar: exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.mul.Scalar: exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.pow.Tensor_Scalar: exir_ops.edge.aten.pow.Tensor_Tensor,
    }

    def __init__(self):
        super(ReplaceOpsWithScalar, self).__init__()

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in self._ops_with_scalar:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            op=self._ops_with_scalar.get(op, op),
            args=(args[0], torch.tensor(args[1])),
            kwargs=kwargs,
            meta=meta,
        )
