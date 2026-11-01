# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Decompositions into core ATen ops for ops that PyTorch keeps whole.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import statically_known_true


def _cubic_filter(x: torch.Tensor) -> torch.Tensor:
    # Keys' cubic with a = -0.5, the one ATen uses when it antialiases.
    a = -0.5
    x = x.abs()
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = (((x - 5) * x + 8) * x - 4) * a
    return torch.where(x < 1, near, torch.where(x < 2, far, 0))


def _antialias_taps(
    in_size: int,
    out_size: int,
    align_corners: bool,
    scale: Optional[float],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Taps of an antialiased resize along one dim, as ATen computes them: output
    o reads input index[o, k] with weight[o, k]. When shrinking, the window
    widens by the shrink factor. Taps outside the window read index in_size.
    """
    # _cubic_filter is 0 from a distance of 2 on.
    support = 2
    if align_corners or scale is None or scale <= 0:
        if align_corners:
            # ATen uses a ratio of 0 for an output size of 1. min and max give
            # that without a branch, which an exported program would fix to
            # the sizes it was traced with.
            num = (in_size - 1) * torch.sym_min(out_size - 1, 1)
            den = torch.sym_max(out_size - 1, 1)
        else:
            num, den = in_size, out_size
        ratio = num / den
        # The runtime takes sym_max of ints only, so max(ratio, 1) is
        # max(num, den) / den.
        widen = torch.sym_max(num, den) / den
        # The tap count is a shape, so count it exactly, in integers. With
        # dynamic sizes, memory is planned for its upper bound, which pairs the
        # largest num with the smallest den even when the two move together.
        reach = torch.sym_max((support * num + den - 1) // den, support)
    else:
        ratio = 1.0 / scale
        widen = max(ratio, 1.0)
        reach = math.ceil(support * widen)
    center = ratio * (torch.arange(out_size, dtype=dtype, device=device) + 0.5)
    # ATen truncates. floor gives the same values here (first is clamped at 0
    # and end is positive), and more backends support it.
    first = torch.clamp(torch.floor(center - support * widen + 0.5), min=0)
    end = torch.clamp(torch.floor(center + support * widen + 0.5), max=in_size)
    # A window never holds more than in_size pixels, which also bounds the
    # memory planned for dynamic sizes.
    tap_count = torch.sym_min(2 * reach + 1, in_size)
    position = first[:, None] + torch.arange(tap_count, dtype=dtype, device=device)
    inside = position < end[:, None]
    weights = _cubic_filter((position - center[:, None] + 0.5) * (1.0 / widen))
    weights = torch.where(inside, weights, 0)
    total = weights.sum(1, keepdim=True)
    weights = torch.where(total != 0, weights / total, weights)
    index = torch.where(inside, position.to(torch.int64), in_size)
    if not statically_known_true(in_size != out_size):
        # With dynamic sizes, whether the dim keeps its size, and so is copied,
        # is only known at run time. A copy reads the output's own pixel in tap
        # 0 and the pad in the other taps.
        same = torch.scalar_tensor(in_size == out_size, dtype=torch.bool, device=device)
        own = torch.arange(index.shape[1], device=device) == 0
        copy = torch.where(own, torch.arange(out_size, device=device)[:, None], in_size)
        index = torch.where(same, copy, index)
        weights = torch.where(same, own.to(dtype), weights)
    return index, weights


def upsample_bicubic2d_aa(
    input: torch.Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """
    ``aten._upsample_bicubic2d_aa`` in core ATen ops. It holds every output's
    window at once, which takes several times the memory of the larger of the
    input and the output. Returns NotImplemented for integer inputs, which
    keeps the op.
    """
    if not input.is_floating_point():
        # ATen rounds integer results in fixed point. Keep the op rather than
        # give different values.
        return NotImplemented  # pyre-ignore[7]: keeps the op undecomposed.
    memory_format = suggest_memory_format(input)
    dims = [
        (dim, scale)
        for dim, scale in ((3, scales_w), (2, scales_h))
        # ATen's CPU kernel copies a dim whose size does not change, whatever
        # its scale.
        if not statically_known_true(input.shape[dim] == output_size[dim - 2])
    ]
    if not dims:
        return input.clone(memory_format=memory_format)
    # Half precision is computed in float32 and rounded once at the end. The
    # default dim order is used throughout, because kernels such as the
    # portable index_select need their input and output in the same one.
    dtype = torch.promote_types(input.dtype, torch.float32)
    result = input.to(dtype, memory_format=torch.contiguous_format)
    for dim, scale in dims:
        index, weights = _antialias_taps(
            input.shape[dim],
            output_size[dim - 2],
            align_corners,
            scale,
            dtype,
            input.device,
        )
        # A tap outside its window has weight 0, but NaN * 0 is NaN. So it reads
        # a 0 padded after the input, which keeps a NaN or inf in the outputs
        # whose window has it.
        padded = torch.nn.functional.pad(result, [0, 0] * (3 - dim) + [0, 1])
        taps = padded.index_select(dim, index.flatten()).unflatten(dim, index.shape)
        result = (taps * weights.view(index.shape + (1,) * (3 - dim))).sum(dim + 1)
    return result.to(input.dtype).contiguous(memory_format=memory_format)
