# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class RewritePadPass(ArmPass):
    """Rewrite constant_pad_nd operator to TOSA Pad operator with constant
    mode.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    targeted_ops = {
        exir_ops.edge.aten.constant_pad_nd.default,
    }

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.targeted_ops:
            return super().call_operator(op, args, kwargs, meta)

        if len(args) == 3:
            input_tensor, pad, value = args
        else:
            input_tensor, pad = args
            value = 0

        output_dtype = meta["val"].dtype
        if output_dtype in (torch.int8, torch.int16):
            input_qparams = meta.data.get("input_qparams", {})
            if len(input_qparams) == 0:
                raise ValueError(
                    f"No input quantization parameters found in metadata for constant_pad_nd with output dtype {output_dtype}"
                )
            value = input_qparams[0].quantize_value(value).item()

        # Each dim needs 2 padding values. For example, to pad the last dimension, the pad has the form
        # (padding_left, padding_right); to pad the last two dimensions, the pad has the form
        # (padding_left, padding_right, padding_top, padding_bottom), and so on. We want to reverse the padding
        # so that we get (N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after) for a 4D
        # input tensor.
        pad_pairs = [[pad[i], pad[i + 1]] for i in range(0, len(pad), 2)]
        input_pad = []
        for pair in reversed(pad_pairs):
            input_pad.extend(pair)
        input_rank = len(input_tensor.data.shape)
        # Place spatial dimensions last and pad non-spatial dimensions with 0 padding
        shape = [0] * ((input_rank * 2 - len(pad))) + input_pad

        pad_shape = super().call_shape_operator(
            exir_ops.backend.tosa.CONST_SHAPE.default, (shape,), {}, meta, True
        )

        return super().call_operator(
            exir_ops.backend.tosa.PAD.default,
            (input_tensor, pad_shape),
            {"value": value},
            meta,
            True,
        )
