# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

# Support both aten and edge dialects
edge_max_pool1d_ops = (exir_ops.edge.aten.max_pool1d.default,)
aten_max_pool1d_ops = (torch.ops.aten.max_pool1d.default,)


def get_ops_for_dialect(op) -> tuple:
    """Get the appropriate ops for the given dialect."""
    if op in edge_max_pool1d_ops:
        return (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.max_pool2d.default,
        )
    if op in aten_max_pool1d_ops:
        return (
            torch.ops.aten.view_copy.default,
            torch.ops.aten.max_pool2d.default,
        )
    raise RuntimeError(f"Can't get decomposition ops for {op}")


class DecomposeMaxPool1dPass(ArmPass):
    """
    This pass decomposes max_pool1d ops into max_pool2d by unsqueezing the input
    from 3D to 4D, calling max_pool2d, and squeezing the output back to 3D.

    This is needed to avoid issues with quantization metadata not propagating
    correctly when max_pool1d decomposes naturally after quantization.

    The transformation is:
    1. Unsqueeze input from (N, C, L) to (N, C, 1, L) by adding dim at position 2
    2. Call max_pool2d with adapted kernel_size, stride, padding
    3. Squeeze output from (N, C, 1, L_out) back to (N, C, L_out)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_max_pool1d_ops + aten_max_pool1d_ops):
            return super().call_operator(op, args, kwargs, meta)

        # Get the appropriate ops for this dialect
        view_copy_op, max_pool2d_op = get_ops_for_dialect(op)

        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else kernel_size
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        # Convert scalar values to lists if needed
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        if isinstance(stride, int):
            stride = [stride]
        if isinstance(padding, int):
            padding = [padding]
        if isinstance(dilation, int):
            dilation = [dilation]

        # Create metadata for intermediate operations (without qparams)
        intermediate_meta = meta.copy()
        intermediate_meta.data["input_qparams"] = {}
        intermediate_meta.data["output_qparams"] = {}

        # Step 1: Unsqueeze input from 3D to 4D (add dimension at position 2)
        # (N, C, L) -> (N, C, 1, L)
        x_shape = list(x.data.shape)
        x_unsqueezed_shape = x_shape[:2] + [1] + x_shape[2:]
        x_unsqueezed = super().call_operator(
            view_copy_op,
            (x, x_unsqueezed_shape),
            {},
            intermediate_meta,
            updated=True,
        )

        # Step 2: Call max_pool2d with 2D parameters
        # kernel_size: [k] -> [1, k]
        # stride: [s] -> [1, s]
        # padding: [p] -> [0, p]
        # dilation: [d] -> [1, d]
        kernel_2d = [1] + kernel_size
        stride_2d = [1] + stride
        padding_2d = [0] + padding
        dilation_2d = [1] + dilation

        pooled = super().call_operator(
            max_pool2d_op,
            (x_unsqueezed, kernel_2d, stride_2d, padding_2d, dilation_2d, ceil_mode),
            {},
            meta,
            updated=True,
        )

        # Step 3: Squeeze output back to 3D
        # (N, C, 1, L_out) -> (N, C, L_out)
        pooled_shape = list(pooled.data.shape)
        output_shape = pooled_shape[:2] + pooled_shape[3:]
        output = super().call_operator(
            view_copy_op,
            (pooled, output_shape),
            {},
            intermediate_meta,
            updated=True,
        )

        return output
