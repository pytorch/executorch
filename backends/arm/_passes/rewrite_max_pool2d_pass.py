# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_max_pool2d_ops = (exir_ops.edge.aten.max_pool2d.default,)


def _to_2tuple(value):
    if isinstance(value, int):
        return (value, value)
    if len(value) == 1:
        return (value[0], value[0])
    return tuple(value)


class RewriteMaxPool2dPass(ArmPass):
    """Rewrite max_pool2d ops to TOSA MAX_POOL2D."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in edge_max_pool2d_ops:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        kernel = _to_2tuple(args[1])

        if len(args) > 2 and args[2] is not None:
            stride = _to_2tuple(args[2])
        else:
            stride = kernel

        padding = _to_2tuple(args[3]) if len(args) > 3 else (0, 0)
        dilation = _to_2tuple(args[4]) if len(args) > 4 else (1, 1)
        ceil_mode = args[5] if len(args) > 5 else False

        if dilation != (1, 1):
            return super().call_operator(op, args, kwargs, meta)

        # TOSA MAX_POOL2D pad order is [top, bottom, left, right]
        pad = [padding[0], padding[0], padding[1], padding[1]]
        pad[1] = adjust_pooling_pad_if_needed(
            x.data.shape[2], kernel[0], stride[0], pad[1], ceil_mode
        )
        pad[3] = adjust_pooling_pad_if_needed(
            x.data.shape[3], kernel[1], stride[1], pad[3], ceil_mode
        )

        pre_permute = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x, list(NHWC_ORDER)),
            {},
            meta,
            updated=True,
        )
        tosa_pool = super().call_operator(
            exir_ops.backend.tosa.MAX_POOL2D.default,
            (
                pre_permute,
                list(kernel),
                list(stride),
                pad,
            ),
            {},
            meta,
            updated=True,
        )
        return super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (tosa_pool, list(NHWC_INVERSE_ORDER)),
            {},
            meta,
            updated=True,
        )
