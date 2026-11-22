# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CUDA NVFP4 gate/up epilogues for fused Muse Glimmer MLP weights."""

# Registers triton::nvfp4_gated_linear.
import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
import torch
import torch.nn as nn
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor


class CudaNVFP4MLP(nn.Module):
    """``MuseGlimmerMLP`` with the fused gate/up projection as one
    ``triton::nvfp4_gated_linear``, which picks the fused NVFP4 kernel or
    ``F.linear`` plus SiLU-gating per shape."""

    def __init__(self, mlp):
        super().__init__()
        self.norm = mlp.norm
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj
        self.train(mlp.training)

    def forward(self, x):
        return self.down_proj(
            torch.ops.triton.nvfp4_gated_linear.default(
                self.norm(x), self.gate_up_proj.weight
            )
        )


def install_nvfp4_gate_up_epilogues(model):
    count = 0
    for layer in model.layers:
        mlp = layer.mlp
        if not mlp.fuse_gate_up:
            continue
        weight = mlp.gate_up_proj.weight
        if (
            not isinstance(weight, CudaNVFP4Tensor)
            or weight.orig_dtype != torch.bfloat16
            or weight.shape[0] != 2 * mlp.intermediate_size
            or mlp.gate_up_proj.bias is not None
        ):
            continue
        layer.mlp = CudaNVFP4MLP(mlp)
        count += 1
    return count
