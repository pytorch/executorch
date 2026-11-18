# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Custom ops for the native backend.

Registers ``native::rope``, the fused rotary position embedding op produced by
``FuseRoPEPass``. Registration happens as an import side effect, so this module is
imported from the backend's entry points (``passes`` / ``partitioner.py``) to
guarantee the op exists before lowering. The reference ``CompositeExplicitAutograd``
body is only used for eager execution; AOT export/serialization uses the fake
(meta) impl.
"""

import torch
from torch import Tensor

lib: torch.library.Library = torch.library.Library("native", "DEF")


lib.define(
    "rope(Tensor input, Tensor position_ids, Tensor inv_freq, "
    "bool interleaved=False, float attention_scale=1.0) -> Tensor"
)


def _check_rope_shapes(input: Tensor, position_ids: Tensor, inv_freq: Tensor) -> None:
    torch._check(input.is_floating_point(), lambda: "rope input must be floating point")
    torch._check(input.dim() == 4, lambda: "rope input must have shape [B, H, T, D]")
    torch._check(
        position_ids.dim() == 2, lambda: "rope position_ids must have shape [Bpos, T]"
    )
    torch._check(inv_freq.dim() == 1, lambda: "rope inv_freq must have shape [R / 2]")
    torch._check(
        (position_ids.shape[0] == 1) | (position_ids.shape[0] == input.shape[0]),
        lambda: "rope position batch must be 1 or match the input batch",
    )
    torch._check(
        position_ids.shape[1] == input.shape[2],
        lambda: "rope position sequence length must match the input",
    )
    torch._check(
        2 * inv_freq.shape[0] <= input.shape[3],
        lambda: "rope rotary width must not exceed the input width",
    )


@torch.library.register_fake("native::rope", lib=lib)
def _rope_fake(
    input: Tensor,
    position_ids: Tensor,
    inv_freq: Tensor,
    interleaved: bool = False,
    attention_scale: float = 1.0,
) -> Tensor:
    _check_rope_shapes(input, position_ids, inv_freq)
    return input.new_empty(input.shape, dtype=input.dtype)


@torch.library.impl("native::rope", "CompositeExplicitAutograd", lib=lib)
def _rope_impl(
    input: Tensor,
    position_ids: Tensor,
    inv_freq: Tensor,
    interleaved: bool = False,
    attention_scale: float = 1.0,
) -> Tensor:
    """Rotate the leading R channels of [B, H, T, D], preserving the tail.

    Positions [Bpos, T] broadcast over heads and, when Bpos is 1, batches.
    inv_freq [R / 2] determines the rotary width. Phase, trig and attention
    scaling use FP32; the tables are then cast to the input dtype.
    """
    _check_rope_shapes(input, position_ids, inv_freq)
    # Elementwise multiplication keeps phase construction in FP32 under autocast,
    # unlike the equivalent batched matmul used by HF's source pattern.
    phase = position_ids.float().unsqueeze(-1) * inv_freq.float()
    if interleaved:
        phase = phase.repeat_interleave(2, dim=-1)
    else:
        phase = torch.cat((phase, phase), dim=-1)
    cos = (phase.cos() * attention_scale).to(input.dtype).unsqueeze(1)
    sin = (phase.sin() * attention_scale).to(input.dtype).unsqueeze(1)
    rotary = 2 * inv_freq.shape[0]
    x_rot = input[..., :rotary]
    x_pass = input[..., rotary:]
    if interleaved:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    else:
        half = rotary // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
    out = x_rot * cos + rotated * sin
    if x_pass.shape[-1] == 0:
        return out
    return torch.cat((out, x_pass), dim=-1)


rope_op: torch._ops.OpOverload = torch.ops.native.rope.default
