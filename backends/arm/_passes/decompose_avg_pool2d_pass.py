# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_div_ops = (exir_ops.edge.aten.avg_pool2d.default,)
aten_div_ops = (torch.ops.aten.avg_pool2d.default,)


def get_decomposition(op) -> tuple:
    if op in edge_div_ops:
        return (
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.mul.Tensor,
        )
    if op in aten_div_ops:
        return (
            torch.ops.aten.full.default,
            torch.ops.aten.cat.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.mul.Tensor,
        )
    raise RuntimeError(f"Can't get avg_pool2d decomposition for op {op}")


class DecomposeAvgPool2dPass(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = {ComputeConstantOpsAOTPass}

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_div_ops + aten_div_ops):
            return super().call_operator(op, args, kwargs, meta)

        full_op, cat_op, avgpool_op, mul_op = get_decomposition(op)

        x = args[0]
        kernel_h, kernel_w = args[1]
        kernel_size = kernel_h * kernel_w
        if len(args) > 2 and args[2] is not None:
            stride_h, stride_w = args[2]
        else:
            stride_h, stride_w = kernel_h, kernel_w
        pad_h, pad_w = new_pad_h, new_pad_w = args[3] if len(args) > 3 else (0, 0)
        ceil_mode = args[4] if len(args) > 4 else False
        count_include_pad = args[5] if len(args) > 5 else True
        divisor_override = args[6] if len(args) > 6 else None

        n, c, h, w = x.data.shape
        post_pad_w, post_pad_h = (0, 0)

        # Count_include_pad == False means that we use a different divisor for edge elements
        # When divisor_override is set, this will be overriden anyways.
        # It is easier to replace a constant divisor, so set count_include_pad == True
        if divisor_override is not None:
            count_include_pad = True

        # Add width padding manually if count_include_pad
        if count_include_pad and pad_w > 0:
            pre_pad_shape = [n, c, h, pad_w]
            pre_pad = super().call_operator(
                full_op, (pre_pad_shape, 0.0), kwargs, meta, updated=True
            )

            if ceil_mode and divisor_override is None:
                post_pad_w = pad_w
            else:
                post_pad_w = adjust_pooling_pad_if_needed(
                    w, kernel_w, stride_w, pad_w, ceil_mode
                )

            if post_pad_w > 0:
                post_pad_shape = [n, c, h, post_pad_w]
                post_pad = super().call_operator(
                    full_op, (post_pad_shape, 0.0), kwargs, meta, updated=True
                )
                cat_nodes = [pre_pad, x, post_pad]
            else:
                cat_nodes = [pre_pad, x]

            x = super().call_operator(
                cat_op, (cat_nodes, 3), kwargs, meta, updated=True
            )
            new_pad_w = 0

        # Add height padding manually if count_include_pad
        if count_include_pad and pad_h > 0:
            pre_pad_shape = [n, c, pad_h, w + pad_w + post_pad_w]
            pre_pad = super().call_operator(
                full_op, (pre_pad_shape, 0.0), kwargs, meta, updated=True
            )

            if ceil_mode and divisor_override is None:
                post_pad_h = pad_h
            else:
                post_pad_h = adjust_pooling_pad_if_needed(
                    h, kernel_h, stride_h, pad_h, ceil_mode
                )

            if post_pad_h > 0:
                post_pad_shape = [n, c, post_pad_h, w + pad_w + post_pad_w]
                post_pad = super().call_operator(
                    full_op, (post_pad_shape, 0.0), kwargs, meta, updated=True
                )
                cat_nodes = [pre_pad, x, post_pad]
            else:
                cat_nodes = [pre_pad, x]

            x = super().call_operator(
                cat_op, (cat_nodes, 2), kwargs, meta, updated=True
            )
            new_pad_h = 0

        avgpool_args = (
            x,
            args[1],
            [stride_h, stride_w],
            [new_pad_h, new_pad_w],
            ceil_mode,
            False,
        )
        x = super().call_operator(avgpool_op, avgpool_args, kwargs, meta, updated=True)

        # Multiply by factor (kernel_size / divisor_override) if divisor_override
        if divisor_override is not None and divisor_override != kernel_size:
            override_multiplier = super().call_operator(
                full_op,
                ([1, 1, 1, 1], kernel_size / divisor_override),
                kwargs,
                meta,
                updated=True,
            )
            x = super().call_operator(
                mul_op, (x, override_multiplier), kwargs, meta, updated=True
            )

        return x
