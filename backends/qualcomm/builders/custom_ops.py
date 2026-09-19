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


if (
    not hasattr(torch.ops, "qnn_custom")
    or not hasattr(torch.ops.qnn_custom, "hadamard_transform")
    or not hasattr(torch.ops.qnn_custom, "space_to_depth")
):
    qnn_custom_lib = Library("qnn_custom", "DEF")
    qnn_custom_lib.define("hadamard_transform(Tensor input, float scale) -> Tensor")
    qnn_custom_lib.define(
        "space_to_depth(Tensor input, int block_h, int block_w) -> Tensor"
    )

    # Qnn HadamardTransform
    @impl(qnn_custom_lib, "hadamard_transform", "CompositeExplicitAutograd")
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

    # Qnn SpaceToDepth
    @impl(qnn_custom_lib, "space_to_depth", "CompositeExplicitAutograd")
    def space_to_depth_impl(
        input: torch.Tensor, block_h: int, block_w: int
    ) -> torch.Tensor:
        # Generalizes torch.nn.functional.pixel_unshuffle to independent block
        # sizes along height/width, matching QNN SpaceToDepth's CRD mode: output
        # channel c*block_h*block_w + i*block_w + j holds input channel c's
        # (i, j)'th block element.
        n, c, h, w = input.shape
        out_h, out_w = h // block_h, w // block_w
        x = input.view(n, c, out_h, block_h, out_w, block_w)
        x = x.permute(0, 1, 3, 5, 2, 4)
        return x.reshape(n, c * block_h * block_w, out_h, out_w)

    @register_fake("qnn_custom::space_to_depth")
    def space_to_depth_fake(
        input: torch.Tensor, block_h: int, block_w: int
    ) -> torch.Tensor:
        n, c, h, w = input.shape
        return input.new_empty(n, c * block_h * block_w, h // block_h, w // block_w)

else:
    qnn_custom_lib = Library("qnn_custom", "FRAGMENT")
