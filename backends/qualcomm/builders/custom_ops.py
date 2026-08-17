# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.library import impl, Library, register_fake


def _hadamard_matrix(dim: int, device, dtype) -> torch.Tensor:
    # Sylvester construction of the (unnormalized, ±1) Hadamard matrix.
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


if not hasattr(torch.ops, "qnn_custom") or not hasattr(
    torch.ops.qnn_custom, "hadamard_transform"
):
    hadamard_op_lib = Library("qnn_custom", "DEF")
    hadamard_op_lib.define("hadamard_transform(Tensor input, float scale) -> Tensor")

    @impl(hadamard_op_lib, "hadamard_transform", "CompositeExplicitAutograd")
    def hadamard_transform_impl(input: torch.Tensor, scale: float) -> torch.Tensor:
        # Normalized Walsh-Hadamard transform along the last dim, times scale.
        # Matches a linear/matmul whose weight is scipy.linalg.hadamard(dim) * s,
        # where the rewrite pass sets scale = s * sqrt(dim) (scale == 1 when the
        # weight is the orthonormal H / sqrt(dim)).
        dim = input.shape[-1]
        h = _hadamard_matrix(dim, input.device, input.dtype)
        return torch.matmul(input, h) * (scale / (dim**0.5))

    @register_fake("qnn_custom::hadamard_transform")
    def hadamard_transform_fake(input: torch.Tensor, scale: float) -> torch.Tensor:
        # Hadamard weight is square, so the transform preserves shape.
        return torch.empty_like(input)

else:
    hadamard_op_lib = Library("qnn_custom", "FRAGMENT")
