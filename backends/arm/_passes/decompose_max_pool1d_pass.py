# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Set, Type, Union

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.pass_base import ExportPass


def _normalize_to_list(
    value: Optional[Union[int, List[int], tuple]],
    default: Optional[List[int]] = None,
) -> List[int]:
    """Normalize parameter to list: handle None, int, tuple, list."""
    if value is None:
        if default is None:
            raise ValueError("Value cannot be None without a default")
        return default
    if isinstance(value, int):
        return [value]
    return list(value)


class DecomposeMaxPool1dPass(ArmPass):
    """
    Decomposes max_pool1d into max_pool2d via unsqueeze_copy/squeeze_copy operations.

    This pass runs in transform_for_annotation (TFA) pipeline before quantization,
    ensuring proper quantization annotation for the decomposed ops.

    Transformation:
        max_pool1d(x, kernel, stride, padding, dilation, ceil_mode)
            → unsqueeze_copy(x, dim=2)           # (N,C,L) → (N,C,1,L)
            → max_pool2d(..., [1,k], [1,s], [0,p], [1,d], ceil_mode)
            → squeeze_copy(..., dims=[2])        # (N,C,1,L') → (N,C,L')
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op != torch.ops.aten.max_pool1d.default or not self.allowed_to_transform(
            meta
        ):
            return super().call_operator(op, args, kwargs, meta)

        # Extract and normalize arguments
        x = args[0]
        kernel_size = _normalize_to_list(args[1])
        stride = _normalize_to_list(
            args[2] if len(args) > 2 else None,
            default=kernel_size,  # stride defaults to kernel_size
        )
        padding = _normalize_to_list(args[3] if len(args) > 3 else 0)
        dilation = _normalize_to_list(args[4] if len(args) > 4 else 1)
        ceil_mode = args[5] if len(args) > 5 else False

        # Step 1: Unsqueeze input from 3D to 4D at dim=2
        # (N, C, L) → (N, C, 1, L)
        x_4d = super().call_operator(
            torch.ops.aten.unsqueeze_copy.default,
            (x, 2),
            {},
            meta,
            updated=True,
        )

        # Step 2: Call max_pool2d with 2D parameters
        # kernel: [k] → [1, k], stride: [s] → [1, s]
        # padding: [p] → [0, p], dilation: [d] → [1, d]
        pooled = super().call_operator(
            torch.ops.aten.max_pool2d.default,
            (
                x_4d,
                [1] + kernel_size,
                [1] + stride,
                [0] + padding,
                [1] + dilation,
                ceil_mode,
            ),
            {},
            meta,
            updated=True,
        )

        # Step 3: Squeeze output back to 3D at dims=[2]
        # (N, C, 1, L') → (N, C, L')
        output = super().call_operator(
            torch.ops.aten.squeeze_copy.dims,
            (pooled, [2]),
            {},
            meta,
            updated=True,
        )

        return output
