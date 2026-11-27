# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Custom ops for the native backend.

Registers ``native::rope``, the fused rotary position embedding op produced by
``FuseRoPEPass``, and ``native::gated_delta_rule``, the Gated DeltaNet recurrence
an exporter emits for linear-attention layers. Registration happens as an import
side effect, so this module is
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


lib.define(
    "gated_delta_rule(Tensor query, Tensor key, Tensor value, Tensor g, "
    "Tensor beta, Tensor state, bool qk_l2norm=False, int snapshots=1) -> "
    "(Tensor, Tensor)"
)


def _check_gated_delta_rule_shapes(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
    snapshots: int,
) -> None:
    torch._check(
        query.dim() == 4 and query.shape == key.shape,
        lambda: "gated_delta_rule query and key must have shape [B, T, H, K]",
    )
    torch._check(
        value.dim() == 4 and value.shape[:3] == query.shape[:3],
        lambda: "gated_delta_rule value must have shape [B, T, H, V]",
    )
    torch._check(
        g.shape == query.shape[:3] and beta.shape == query.shape[:3],
        lambda: "gated_delta_rule g and beta must have shape [B, T, H]",
    )
    b, _, h, k = query.shape
    torch._check(snapshots >= 1, lambda: "gated_delta_rule snapshots must be >= 1")
    if snapshots == 1:
        torch._check(
            state.shape == (b, h, value.shape[3], k),
            lambda: "gated_delta_rule state must have shape [B, H, V, K]",
        )
    else:
        torch._check(
            state.shape == (b, snapshots, h, value.shape[3], k),
            lambda: "gated_delta_rule state must have shape [B, S, H, V, K]",
        )


@torch.library.register_fake("native::gated_delta_rule", lib=lib)
def _gated_delta_rule_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
    qk_l2norm: bool = False,
    snapshots: int = 1,
) -> tuple[Tensor, Tensor]:
    _check_gated_delta_rule_shapes(query, key, value, g, beta, state, snapshots)
    return (
        query.new_empty(value.shape, dtype=query.dtype),
        state.new_empty(state.shape, dtype=_compute_dtype(query)),
    )


def _compute_dtype(query: Tensor) -> torch.dtype:
    return torch.promote_types(query.dtype, torch.float32)


def _l2norm(x: Tensor) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


@torch.library.impl("native::gated_delta_rule", "CompositeExplicitAutograd", lib=lib)
def _gated_delta_rule_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
    qk_l2norm: bool = False,
    snapshots: int = 1,
) -> tuple[Tensor, Tensor]:
    """The gated delta rule over T tokens in order, from `state`, as
    transformers' torch_recurrent_gated_delta_rule computes it.

    query and key [B, T, H, K], value [B, T, H, V], g (log decay) and beta
    [B, T, H]. The state S is stored value-major, [B, H, V, K]: element
    [b, h, j, i] is S[i][j] of transformers' [B, H, K, V] state. Per token,
    S = S * exp(g); S += k (beta (v - S^T k))^T; out = S^T q / sqrt(K). With
    `qk_l2norm`, query and key are first scaled by rsqrt(sum(x^2) + 1e-6).
    Computes in FP32, or FP64 for FP64 inputs, the l2norm included; returns
    the output in query's dtype and the final state in the compute dtype.

    With `snapshots` S > 1 the state is the S most recent states, newest
    first, [B, S, H, V, K]; the rule starts from slot 0 and returns the S
    most recent states after the call: slot i is the state after token
    T - 1 - i, or, for i >= T, the input's slot i - T. Speculative decoding
    rolls a sequence back to one of them.
    """
    _check_gated_delta_rule_shapes(query, key, value, g, beta, state, snapshots)
    dtype, compute = query.dtype, _compute_dtype(query)
    if query.shape[1] == 0:
        return query.new_empty(value.shape), state.to(compute).contiguous()
    q, k, v, beta, g = (
        x.transpose(1, 2).to(compute) for x in (query, key, value, beta, g)
    )
    if qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    q = q * (1 / q.shape[-1] ** 0.5)
    states = state.to(compute) if snapshots > 1 else state.to(compute)[:, None]
    s = states[:, 0].transpose(-1, -2)
    out, after = [], []
    for t in range(q.shape[2]):
        s = s * g[:, :, t].exp()[..., None, None]
        kv = (s * k[:, :, t, :, None]).sum(dim=-2)
        delta = (v[:, :, t] - kv) * beta[:, :, t, None]
        s = s + k[:, :, t, :, None] * delta[:, :, None, :]
        out.append((s * q[:, :, t, :, None]).sum(dim=-2))
        if t >= q.shape[2] - snapshots:
            after.append(s.transpose(-1, -2))
    newest = torch.stack(after[::-1], dim=1)
    final = torch.cat((newest, states[:, : snapshots - newest.shape[1]]), dim=1)
    return (
        torch.stack(out, dim=2).transpose(1, 2).contiguous().to(dtype),
        final.contiguous() if snapshots > 1 else final[:, 0].contiguous(),
    )


gated_delta_rule_op: torch._ops.OpOverload = torch.ops.native.gated_delta_rule.default
