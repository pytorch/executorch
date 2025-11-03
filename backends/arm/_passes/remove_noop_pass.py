# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Set, Type

from executorch.backends.arm._passes import ArmPass

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

logger = logging.getLogger(__name__)


class RemoveNoopPass(ArmPass):
    """Remove no-ops from graph_module"""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in (
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            exir_ops.edge.aten.copy.default,
        ):
            return super().call_operator(op, args, kwargs, meta)

        input_dtype = args[0].data.dtype
        output_dtype = kwargs.get("dtype", input_dtype)

        if input_dtype != output_dtype:
            return super().call_operator(op, args, kwargs, meta)

        if op == exir_ops.edge.aten.copy.default:
            return args[1]
        return args[0]
