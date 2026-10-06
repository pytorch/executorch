# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatch of quantized ``F.linear`` to a ``QuantizedGemmFamily`` op.

Shared by the per-format dispatchers (``int{4,5,6,8}_dispatch.py``):

1. ``can_launch_triton``: is a Triton kernel possible here at all?
2. ``select_bucket``: the first bucket, in ascending order, whose op supports
   the arguments (the family decides, through its ``supports``).
3. Otherwise the format's dequantize + ``F.linear`` fallback, chunked along N
   by ``chunked_dequant_linear`` so huge weights (an lm_head) never build one
   full-size dequantized copy.

Unsupported inputs always fall back; dispatch never raises for them.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import is_fake

# Dequantizing a whole large weight (e.g. a 262144-row lm_head) in one shot
# materializes several full-size temporaries (~10 GiB). Above this row count the
# weight is dequantized and multiplied in N chunks instead; output rows are
# independent, so the result is identical. Only weights above the threshold
# take this path, and only when no kernel serves them.
_DEQUANT_N_THRESHOLD = 65536
_DEQUANT_N_CHUNK = 32768


def can_launch_triton(x: torch.Tensor) -> bool:
    """Whether a Triton kernel can serve ``x``.

    Export and torch.compile trace with fake tensors, whatever device the
    example inputs are on, and the CUDA backend compiles the kernel; only real
    CPU eager cannot launch Triton.
    """
    return x.device.type == "cuda" or is_fake(x)


def select_bucket(family, *args) -> Optional[int]:
    """The smallest bucket whose op supports ``args``, or None."""
    return next((b for b in family.buckets if family.supports(b, *args)), None)


def chunked_dequant_linear(
    x: torch.Tensor,
    n: int,
    dequant_linear_rows: Callable[[int, int], torch.Tensor],
) -> torch.Tensor:
    """``F.linear`` of ``x`` with an N-row quantized weight, via the format's
    ``dequant_linear_rows(start, end)``, which dequantizes weight rows
    ``[start, end)`` and returns ``F.linear(x, rows)``."""
    if n <= _DEQUANT_N_THRESHOLD:
        return dequant_linear_rows(0, n)
    return torch.cat(
        [
            dequant_linear_rows(i, min(i + _DEQUANT_N_CHUNK, n))
            for i in range(0, n, _DEQUANT_N_CHUNK)
        ],
        dim=-1,
    )


def quantized_linear(
    family,
    input: torch.Tensor,
    weight_args: Sequence,
    bias: Optional[torch.Tensor],
    dequant_linear: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``F.linear(input, weight, bias)`` for a quantized weight.

    ``weight_args`` are the family op's arguments after the activation;
    ``dequant_linear(x_2d)`` is the format's fallback on the 2-D activation.
    """
    orig_shape = input.shape
    x_2d = input.reshape(-1, orig_shape[-1])
    args = (x_2d, *weight_args)
    bucket = select_bucket(family, *args) if can_launch_triton(x_2d) else None
    if bucket is not None:
        out = family.op(bucket)(*args)
    else:
        out = dequant_linear(x_2d)
    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out


__all__ = [
    "can_launch_triton",
    "chunked_dequant_linear",
    "quantized_linear",
    "select_bucket",
]
