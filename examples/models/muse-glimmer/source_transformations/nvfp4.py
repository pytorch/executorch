# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CUDA NVFP4 gate/up epilogues for fused Muse Glimmer MLP weights."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    can_launch_triton,
)
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import _has_native_fp4
from executorch.backends.cuda.triton.kernels.nvfp4_output import (
    nvfp4_prefill_fp32_enabled,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true


class CudaNVFP4MLP(nn.Module):
    @classmethod
    def from_mlp(cls, mlp):
        result = cls()
        result.intermediate_size = mlp.intermediate_size
        result.fuse_gate_up = mlp.fuse_gate_up
        result.norm = mlp.norm
        result.gate_up_proj = mlp.gate_up_proj
        result.down_proj = mlp.down_proj
        result.train(mlp.training)
        return result

    def forward(self, x):
        h = self.norm(x)
        flat = h.reshape(-1, h.shape[-1]).contiguous()
        mode = os.environ.get("ET_CUDA_NVFP4_ACT", "auto")
        if mode not in ("auto", "w4a4", "w4a16"):
            raise ValueError("ET_CUDA_NVFP4_ACT must be auto, w4a4, or w4a16")
        if (
            mode == "w4a16"
            or statically_known_true(flat.shape[0] <= 4)
            or (
                h.dtype != torch.bfloat16
                and not (h.dtype == torch.float32 and nvfp4_prefill_fp32_enabled())
            )
            or (not torch.compiler.is_compiling() and not can_launch_triton(flat))
            or not _has_native_fp4()
        ):
            projected = self.gate_up_proj(h)
            gate = projected[..., : self.intermediate_size]
            up = projected[..., self.intermediate_size :]
            return self.down_proj(F.silu(gate) * up)
        activated = torch.ops.triton.nvfp4_gated_linear.default(
            h, self.gate_up_proj.weight
        )
        return self.down_proj(activated)


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
        layer.mlp = CudaNVFP4MLP.from_mlp(mlp)
        count += 1
    return count
