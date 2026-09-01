# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Q4_K dequantization fallback for CUDA linear dispatch."""

import torch
import torch.nn.functional as F


_DEQUANT_N_THRESHOLD = 65536
_DEQUANT_N_CHUNK = 32768


def dequant_matmul(x, qdata, scale, scale_step, zero, zero_point_step, group_size):
    """Dequantize Q4_K weights to the activation dtype and call F.linear."""
    N, K_half = qdata.shape
    K = K_half * 2
    n_groups = K // group_size
    gs_half = group_size // 2
    n_super = K // 256
    groups_per_super = n_groups // n_super
    dtype = x.dtype

    def _unit_dq_mm(qd, sc, s_step, ze, z_step, rows):
        p = qd.to(torch.uint8).reshape(rows, n_groups, gs_half)
        low = (p & 0x0F).to(dtype)
        high = ((p >> 4) & 0x0F).to(dtype)
        data = torch.stack([low, high], dim=-1).reshape(rows, n_groups, group_size)
        scale_step_g = s_step.to(dtype).repeat_interleave(groups_per_super, dim=1)
        s = (sc.to(dtype) * scale_step_g).unsqueeze(-1)
        zero_point_step_g = z_step.to(dtype).repeat_interleave(groups_per_super, dim=1)
        z = (ze.to(dtype) * zero_point_step_g).unsqueeze(-1)
        w_deq = ((data - z) * s).reshape(rows, K)
        return F.linear(x, w_deq)

    if N <= _DEQUANT_N_THRESHOLD:
        return _unit_dq_mm(qdata, scale, scale_step, zero, zero_point_step, N)

    outs = []
    for i in range(0, N, _DEQUANT_N_CHUNK):
        j = min(i + _DEQUANT_N_CHUNK, N)
        outs.append(
            _unit_dq_mm(
                qdata[i:j],
                scale[i:j],
                scale_step[i:j],
                zero[i:j],
                zero_point_step[i:j],
                j - i,
            )
        )
    return torch.cat(outs, dim=-1)
