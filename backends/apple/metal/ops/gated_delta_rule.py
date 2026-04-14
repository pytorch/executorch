# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
metal::gated_delta_rule custom op for linear attention recurrence.

Performs the gated delta rule recurrence over T time steps, mutating
the recurrent state in-place. The Metal fallback kernel is in
runtime/ops/op_gated_delta_rule.mm.
"""

import torch
from torch import Tensor


@torch.library.custom_op("metal::gated_delta_rule", mutates_args=("state",))
def gated_delta_rule(
    q: Tensor,  # [B, T, Hk, Dk]
    k: Tensor,  # [B, T, Hk, Dk]
    v: Tensor,  # [B, T, Hv, Dv]
    g: Tensor,  # [B, T, Hv] — decay gate (already exp'd)
    beta: Tensor,  # [B, T, Hv] — update gate
    state: Tensor,  # [B, Hv, Dv, Dk] — recurrent state (MUTATED)
) -> Tensor:
    """Reference implementation: sequential recurrence over T."""
    B, T_len, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    s = state.clone().float()
    ys = []

    for t in range(T_len):
        q_t = q[:, t].float()  # [B, Hk, Dk]
        k_t = k[:, t].float()  # [B, Hk, Dk]
        v_t = v[:, t].float()  # [B, Hv, Dv]
        g_t = g[:, t].float()  # [B, Hv]
        beta_t = beta[:, t].float()  # [B, Hv]

        # Decay
        s = s * g_t[:, :, None, None]

        # Project state by key
        kv_mem = (s * k_t[:, :, None, :]).sum(dim=-1)  # [B, Hv, Dv]

        # Delta rule update
        delta = (v_t - kv_mem) * beta_t[:, :, None]  # [B, Hv, Dv]
        s = s + k_t[:, :, None, :] * delta[:, :, :, None]  # [B, Hv, Dv, Dk]

        # Read from state
        y_t = (s * q_t[:, :, None, :]).sum(dim=-1)  # [B, Hv, Dv]
        ys.append(y_t)

    state.copy_(s.to(state.dtype))
    return torch.stack(ys, dim=1).to(q.dtype)


@torch.library.register_fake("metal::gated_delta_rule")
def gated_delta_rule_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
) -> Tensor:
    B, T = q.shape[:2]
    Hv, Dv = v.shape[-2:]
    return torch.empty(B, T, Hv, Dv, dtype=q.dtype, device=q.device)


# C shim mapping for AOTInductor code generation.
# The op mutates state in-place and returns one tensor (y). AOTInductor's
# auto_functionalized wrapper passes 6 input handles + 1 output pointer.
metal_gated_delta_rule_c_shim = {
    torch.ops.metal.gated_delta_rule.default: [
        "AOTITorchError aoti_torch_mps_gated_delta_rule("
        "AtenTensorHandle Q, AtenTensorHandle K, AtenTensorHandle V, "
        "AtenTensorHandle G, AtenTensorHandle Beta, AtenTensorHandle StateIn, "
        "AtenTensorHandle* retY)"
    ],
}
