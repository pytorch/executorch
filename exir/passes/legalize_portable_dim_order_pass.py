# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import get_dim_order
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
import executorch.exir.passes.dim_order_ops_registry  # noqa: F401


class LegalizePortableDimOrderPass(ExportPass):
    """Insert contiguous dim-order copies before portable default-layout ops.

    Runs during ``edge -> executorch`` lowering on leftover edge ops whose
    portable kernels require default dim order on their primary input.
    """

    _copy_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    _target_ops = {
        exir_ops.edge.aten._adaptive_avg_pool2d.default,
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._target_ops:
            return super().call_operator(op, args, kwargs, meta)

        input_arg = args[0]
        if isinstance(input_arg, ProxyValue) and input_arg.is_tensor():
            input_tensor: Optional[torch.Tensor] = input_arg.to_tensor()
            input_meta = NodeMetadata(input_arg.node.meta)
        elif isinstance(input_arg, torch.Tensor):
            input_tensor = input_arg
            input_meta = meta.copy()
        else:
            input_tensor = None
            input_meta = meta.copy()

        if input_tensor is None or tuple(int(d) for d in input_tensor.dim_order()) == tuple(
            range(input_tensor.dim())
        ):
            return super().call_operator(op, args, kwargs, meta)

        contiguous_dim_order = get_dim_order(
            torch.contiguous_format, input_tensor.dim()
        )
        assert contiguous_dim_order is not None

        legalized_input = super().call_operator(
            self._copy_op,
            (input_arg,),
            {"dim_order": contiguous_dim_order},
            input_meta,
        )
        return super().call_operator(
            op,
            (legalized_input, *args[1:]),
            kwargs,
            meta,
        )
