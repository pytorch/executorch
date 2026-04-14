# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
metal::gather_qmv custom op for MoE expert-indexed quantized matmul.

Performs y[i] = W[expert_idx[i]] @ x[i] with INT4 quantized expert weights.
The Metal fallback kernel is in runtime/ops/op_gather_qmv.mm.
"""

import torch
from torch import Tensor


@torch.library.custom_op("metal::gather_qmv", mutates_args=())
def gather_qmv(
    x: Tensor,  # [P, K] — activations (P = num token-expert pairs)
    w: Tensor,  # [E, N, K_packed] — packed INT4 expert weights
    scales: Tensor,  # [E, N, K/gs] — per-group scales
    biases: Tensor,  # [E, N, K/gs] — per-group biases
    expert_indices: Tensor,  # [P] — expert index per pair
    group_size: int,
) -> Tensor:
    """Reference implementation for tracing and CPU testing."""
    P, K = x.shape
    E, N, K_packed = w.shape

    y = torch.zeros(P, N, dtype=x.dtype, device=x.device)
    for i in range(P):
        eidx = expert_indices[i].item()
        w_e = w[eidx]  # [N, K_packed]
        s_e = scales[eidx]  # [N, K/gs]
        b_e = biases[eidx]  # [N, K/gs]

        # Dequantize: unpack INT4, apply affine dequant
        w_unpacked = _dequantize_int4_affine(w_e, s_e, b_e, K, group_size)
        y[i] = w_unpacked @ x[i]

    return y


def _dequantize_int4_affine(
    w_packed: Tensor, scales: Tensor, biases: Tensor, K: int, group_size: int
) -> Tensor:
    """Dequantize packed INT4 weights using MLX affine format."""
    N = w_packed.shape[0]
    w_bytes = w_packed.to(torch.int16)
    low = w_bytes & 0x0F
    high = (w_bytes >> 4) & 0x0F
    w_int = torch.stack([low, high], dim=-1).reshape(N, K).float()

    scales_expanded = scales.float().repeat_interleave(group_size, dim=-1)[:, :K]
    biases_expanded = biases.float().repeat_interleave(group_size, dim=-1)[:, :K]

    return (w_int * scales_expanded + biases_expanded).to(scales.dtype)


@torch.library.register_fake("metal::gather_qmv")
def gather_qmv_fake(
    x: Tensor,
    w: Tensor,
    scales: Tensor,
    biases: Tensor,
    expert_indices: Tensor,
    group_size: int,
) -> Tensor:
    P = x.shape[0]
    N = w.shape[1]
    return torch.empty(P, N, dtype=x.dtype, device=x.device)


# C shim mapping for AOTInductor code generation.
# Maps the torch op to the C function name that the generated wrapper calls.
metal_gather_qmv_c_shim = {
    torch.ops.metal.gather_qmv.default: [
        "AOTITorchError aoti_torch_mps_gather_qmv("
        "AtenTensorHandle X, AtenTensorHandle W, AtenTensorHandle S, "
        "AtenTensorHandle Z, AtenTensorHandle ExpertIndices, "
        "int64_t group_size, AtenTensorHandle* ret)"
    ],
}
