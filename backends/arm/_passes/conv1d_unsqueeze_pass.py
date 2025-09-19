# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class Conv1dUnsqueezePass(ExportPass):
    """
    This pass is used to change conv1d ops into conv2d since TOSA only
    supports 2d and 3d convolution. This is done by modifying the graph to do the
    following:
    1a) unsqueeze the convolution's input from 3d to 4d
    1b) unsqueeze the convolution's weight from 3d to 4d
    2) perform a conv2d (with a modified version of the original conv1d args)
    3) squeeze the output back down to 3d.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.convolution.default:
            return super().call_operator(op, args, kwargs, meta)
        stride = list(args[3])
        if len(stride) != 1:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        x_unsqueezed_shape = list(x.data.shape) + [1]
        x = super().call_operator(
            exir_ops.edge.aten.view_copy.default, (x, x_unsqueezed_shape), {}, meta
        )

        w_meta = meta.copy()
        w_meta.data["input_qparams"] = {}
        w_meta.data["output_qparams"] = {}

        w = args[1]
        w_unsqueezed_shape = list(w.data.shape) + [1]
        w = super().call_operator(
            exir_ops.edge.aten.view_copy.default, (w, w_unsqueezed_shape), {}, w_meta
        )

        new_args = (
            x,
            w,
            args[2],
            args[3] + [1],  # stride
            args[4] + [0],  # padding
            args[5] + [1],  # dilation
            args[6],
            args[7] + [0],
            args[8],
        )
        x = super().call_operator(
            exir_ops.edge.aten.convolution.default, new_args, kwargs, meta
        )

        x_squeezed_shape = list(x.data.shape)[:-1]
        x = super().call_operator(
            exir_ops.edge.aten.view_copy.default, (x, x_squeezed_shape), {}, meta
        )

        return x
