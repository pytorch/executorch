# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm.tosa.dialect.shape import is_shape_op_node

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue


class InsertDynamicPaddingPass(ArmOpTargetedPass):
    """This pass rewrites conv operations with padding to use an explicit pad
    operator before the conv2d operation and setting the padding to zero in the
    conv2d operator. E.g. conv2d(x, weight, bias, stride, padding, dilation)
    becomes: x_padded = pad(x, explicit_padding) conv2d(x_padded,
    weight, bias, stride, (0,0,0,0), dilation) where explicit_padding is
    calculated based on the original padding value.

    To be used with dynamic shapes only.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (
        exir_ops.backend.tosa.CONV2D.default,
        exir_ops.backend.tosa.DEPTHWISE_CONV2D.default,
        exir_ops.backend.tosa.MAX_POOL2D.default,
    )

    def _is_dynamic_padding(
        self, padding: ProxyValue | list[int] | tuple[int, ...]
    ) -> bool:
        return (isinstance(padding, ProxyValue) and is_shape_op_node(padding.node)) or (
            (
                isinstance(padding, (list, tuple))
                and any(isinstance(p, torch.SymInt) for p in padding)
            )
        )

    def call_operator(self, op, args, kwargs, meta, updated=False) -> ProxyValue:
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)
        if op == exir_ops.backend.tosa.MAX_POOL2D.default:
            padding_index = 3
        else:
            padding_index = 4
        padding = args[padding_index]
        if not self._is_dynamic_padding(padding):
            return super().call_operator(op, args, kwargs, meta, updated)

        # Create a pad op before conv2d
        input_tensor = args[0]

        zero_padding_pair = [0, 0]
        zero_spatial_padding = [0, 0, 0, 0]
        N_padding = super().call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default,
            (zero_padding_pair,),
            {},
            meta,
            True,
        )
        C_padding = N_padding

        padding_shape_args = [N_padding, padding, C_padding]

        padding_shape = super().call_shape_operator(
            exir_ops.backend.tosa.CONCAT_SHAPE.default,
            (padding_shape_args,),
            {},
            meta,
            True,
        )

        pad_res = super().call_operator(
            exir_ops.backend.tosa.PAD.default,
            (
                input_tensor,
                padding_shape,
            ),
            {
                "value": 0,
            },
            meta,
            True,
        )
        new_conv2d_args = list(args)
        new_conv2d_args[0] = pad_res
        new_conv2d_args[padding_index] = zero_spatial_padding
        return super().call_operator(op, tuple(new_conv2d_args), kwargs, meta, updated)
