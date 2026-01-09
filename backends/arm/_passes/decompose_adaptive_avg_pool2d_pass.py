# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil, floor
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.decompose_avg_pool2d_pass import (
    DecomposeAvgPool2dPass,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata

edge_ops = (exir_ops.edge.aten._adaptive_avg_pool2d.default,)
aten_ops = (torch.ops.aten.adaptive_avg_pool2d.default,)


def _get_decomposition(op) -> tuple:
    if op in edge_ops:
        return (
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.cat.default,
        )
    if op in aten_ops:
        return (
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.slice_copy.Tensor,
            torch.ops.aten.cat.default,
        )
    raise RuntimeError(f"Unable to get decomposition for op {op}")


class DecomposeAdaptiveAvgPool2dPass(ArmPass):
    """
    Decomposes AdaptiveAvgPool2d into AvgPool2d operations.

    An input tensor of shape (N, C, H, W) is transformed into an output tensor
    of shape (N, C, output_size_h, output_size_w).

    The output is of size output_size_h x output_size_w for any input.
    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeAvgPool2dPass}

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in (edge_ops + aten_ops):
            return super().call_operator(op, args, kwargs, meta, updated)

        avg_pool2d_op, slice_op, cat_op = _get_decomposition(op)

        x = args[0]
        _, _, input_size_h, input_size_w = x.data.shape

        (output_size_h, output_size_w) = args[1]

        # Vela currently only allows a stride in the interval of [1,3] for AvgPool2d.
        # To accommodate this, the AvgPool2d op is applied to pooling regions and the results are concatenated.

        # Slices and concats does not require quantization parameters
        metadata_dict = dict(meta.data)
        metadata_dict["input_qparams"] = {}
        metadata_dict["output_qparams"] = {}
        meta_with_no_qparams = NodeMetadata(metadata_dict)
        res = []
        for out_i in range(output_size_h):
            row = []
            for out_j in range(output_size_w):
                # Calculate pooling regions
                start_h = floor(out_i * input_size_h / output_size_h)
                end_h = ceil((out_i + 1) * input_size_h / output_size_h)
                start_w = floor(out_j * input_size_w / output_size_w)
                end_w = ceil((out_j + 1) * input_size_w / output_size_w)

                # Slice along H
                x_h = super().call_operator(
                    slice_op, (x, 2, start_h, end_h), kwargs, meta_with_no_qparams, True
                )
                # Slice along W
                x_hw = super().call_operator(
                    slice_op,
                    (x_h, 3, start_w, end_w),
                    kwargs,
                    meta_with_no_qparams,
                    True,
                )

                # Apply avg pooling with kernel size equal to the pooling region
                kernel_h = end_h - start_h
                kernel_w = end_w - start_w
                pool_args = (x_hw, (kernel_h, kernel_w), (1, 1), (0, 0))
                pooled = super().call_operator(
                    avg_pool2d_op, pool_args, kwargs, meta, True
                )
                row.append(pooled)

            # Concatenate row results along width (dim=3)
            row_tensor = super().call_operator(
                cat_op, (row, 3), kwargs, meta_with_no_qparams, True
            )
            res.append(row_tensor)

        # Concatenate all rows along height (dim=2)
        out = super().call_operator(
            cat_op, (res, 2), kwargs, meta_with_no_qparams, True
        )
        return out
