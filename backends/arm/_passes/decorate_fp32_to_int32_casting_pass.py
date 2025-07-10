# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_node_arg
from executorch.exir.dialects._ops import ops as exir_ops


def _get_decorated_ops(op):
    if op in DecorateFp32toInt32CastingPass.targets:
        return (
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.ge.Tensor,
            exir_ops.edge.aten.floor.default,
            exir_ops.edge.aten.ceil.default,
            exir_ops.edge.aten.where.self,
        )
    else:
        raise RuntimeError(f"Can't get decorated ops for op {op}")


class DecorateFp32toInt32CastingPass(ArmPass):
    """
    To lower pytorch fp32 -> int32 casting to TOSA,
    we need to transform the value with Ceil, Floor, and Where.
    Before:
        output = to_copy(x, dtype=torch.int32)
    After:
        %zero = full((1,), 0.0, dtype=torch.float32)
        is_non_negative = x >= %zero
        floor_x = floor(x)
        ceil_x = ceil(x)
        decorated_x = where(is_non_negative, floor_x, ceil_x)
        output = to_copy(decorated_x, dtype=torch.int32)
    """

    targets = [
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    ]

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.targets:
            return super().call_operator(op, args, kwargs, meta)

        input = get_node_arg(args, 0)
        input_dtype = input.node.meta["val"].dtype
        output_dtype = meta["val"].dtype

        if not (input_dtype == torch.float32 and output_dtype == torch.int32):
            return super().call_operator(op, args, kwargs, meta)

        op_full, op_ge, op_floor, op_ceil, op_where = _get_decorated_ops(op)

        zero = super().call_operator(
            op_full,
            args=((1,) * len(meta["val"].size()), 0.0),
            kwargs={"dtype": torch.float32},
            meta=meta,
            updated=True,
        )

        is_non_negative = super().call_operator(
            op_ge, (input, zero), {}, meta, updated=True
        )
        floor_x = super().call_operator(op_floor, (input,), {}, meta, updated=True)
        ceil_x = super().call_operator(op_ceil, (input,), {}, meta, updated=True)
        decorated_x = super().call_operator(
            op_where, (is_non_negative, floor_x, ceil_x), {}, meta, updated=True
        )

        return super().call_operator(op, (decorated_x,), kwargs, meta, updated=True)
