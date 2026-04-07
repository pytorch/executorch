#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SwitchLinear — per-expert linear layer using mlx::gather_mm / mlx::gather_qmm.

A self-contained expert linear primitive (like nn.Linear but with expert
selection via index-based gather). Mirrors mlx-lm's SwitchLinear /
QuantizedSwitchLinear but uses PyTorch nn.Module with torchao quantization
support.

Lifecycle:
  1. __init__: creates per-expert nn.Linear in nn.ModuleList
  2. (optional) quantize_model_(): quantizes all expert linears
  3. pack(): stacks weights into 3D buffers, deletes ModuleList
  4. forward(x, indices): unsqueeze → gather_mm/gather_qmm → squeeze

Usage:
    from executorch.backends.mlx.llm.switch import SwitchLinear, pack_all_switch_linears

    gate_proj = SwitchLinear(hidden, inter, num_experts)
    up_proj = SwitchLinear(hidden, inter, num_experts)
    down_proj = SwitchLinear(inter, hidden, num_experts)

    # After optional quantize_model_():
    pack_all_switch_linears(model)

    # In forward:
    for k in range(top_k):
        idx = expert_indices[:, k]
        gate = gate_proj(x, idx)
        up = up_proj(x, idx)
        h = F.silu(gate) * up
        down = down_proj(h, idx)
        output += routing_weights[:, k:k+1] * down
"""

import logging

import torch
import torch.nn as nn

# Import MLX custom ops to register mlx::gather_mm and mlx::gather_qmm
from executorch.backends.mlx import custom_ops as _mlx_custom_ops  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["SwitchLinear", "SwitchMLP", "pack_all_switch_linears"]


class SwitchLinear(nn.Module):
    """Per-expert linear layer using mlx::gather_mm / mlx::gather_qmm.

    Stores expert weights as nn.ModuleList of nn.Linear, so quantize_model_()
    naturally quantizes them. After quantization (or without it), call pack()
    to stack weights into 3D buffers for the MLX gather custom ops.

    Args:
        input_dims: Input feature dimension
        output_dims: Output feature dimension
        num_experts: Number of experts
        bias: Whether to use bias (default: False)
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self._packed = False
        self._is_quantized = False

        self.experts = nn.ModuleList(
            [nn.Linear(input_dims, output_dims, bias=bias) for _ in range(num_experts)]
        )

    def pack(self):
        """Stack per-expert weights into 3D buffers and delete the ModuleList.

        Must be called after quantization (if any) and before forward/export.

        - Quantized: extracts inner tensors (qdata, scale, zero_point),
          stacks into [E, out, in_packed] buffers. Weight layout matches
          mlx::gather_qmm's expectations (transpose=True handles transposition).
        - Unquantized: stacks weight.data into [E, out, in], then pretransposes
          to [E, in, out] so gather_mm receives the correct layout directly
          (no runtime transpose needed).
        """
        if self._packed:
            return

        w0 = self.experts[0].weight
        self._is_quantized = hasattr(w0, "qdata")

        if self._is_quantized:
            _, metadata = w0.__tensor_flatten__()
            self.group_size = metadata["block_size"][-1]

            self.register_buffer(
                "qdata",
                torch.stack([e.weight.qdata for e in self.experts]),
            )
            self.register_buffer(
                "scale",
                torch.stack([e.weight.scale for e in self.experts]),
            )
            self.register_buffer(
                "zero_point",
                torch.stack([e.weight.zero_point for e in self.experts]),
            )
        else:
            # Stack [E, out, in] then pretranspose to [E, in, out]
            stacked = torch.stack([e.weight.data for e in self.experts])
            self.register_buffer("weight", stacked.transpose(-1, -2).contiguous())

        del self.experts
        self._packed = True

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        sorted_indices: bool = False,
    ) -> torch.Tensor:
        """Forward without unsqueeze/squeeze — caller manages dimensions.

        Used by UnfusedMoEExperts which passes x as [N, 1, 1, D]
        and indices as [N, top_k] to handle all experts at once.
        """
        if not self._packed:
            raise RuntimeError("SwitchLinear.pack() must be called before forward_raw.")

        if self._is_quantized:
            return torch.ops.mlx.gather_qmm(
                x,
                self.qdata,
                self.scale,
                biases=self.zero_point,
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                sorted_indices=sorted_indices,
            )
        else:
            return torch.ops.mlx.gather_mm(
                x,
                self.weight,
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )


class SwitchMLP(nn.Module):
    """Gated MoE MLP using SwitchLinear for each projection.

    Bundles gate + up + down projections with gated activation and optional
    expert sorting into a single reusable component. Works with any gated
    activation (SwiGLU, GeGLU, ReGLU, etc.).

    When fuse_gate_up=True, gate and up projections share a single
    SwitchLinear with output dim 2*intermediate_size. This reduces
    gather_mm/gather_qmm calls from 3 to 2 per forward pass (one
    fused gate+up gather, one down gather). The output is split via
    a cheap tensor slice.

    Args:
        hidden_size: Model hidden dimension (input/output of MLP)
        intermediate_size: MLP intermediate dimension (per expert)
        num_experts: Number of experts
        activation: Gating activation function (default: F.silu for SwiGLU)
        bias: Whether expert linears use bias
        fuse_gate_up: Fuse gate and up projections into a single SwitchLinear
            (default: False). When True, uses one [E, 2*inter, D] weight
            instead of two [E, inter, D] weights, saving one gather call.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        activation=None,
        bias: bool = False,
        fuse_gate_up: bool = False,
    ):
        super().__init__()
        if activation is None:
            activation = nn.functional.silu
        self.activation = activation
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.fuse_gate_up = fuse_gate_up

        if fuse_gate_up:
            self.gate_up_proj = SwitchLinear(
                hidden_size, 2 * intermediate_size, num_experts, bias=bias
            )
        else:
            self.gate_proj = SwitchLinear(
                hidden_size, intermediate_size, num_experts, bias=bias
            )
            self.up_proj = SwitchLinear(
                hidden_size, intermediate_size, num_experts, bias=bias
            )
        self.down_proj = SwitchLinear(
            intermediate_size, hidden_size, num_experts, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        top_k: int,
        sort_experts: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the gated MoE MLP.

        Args:
            x: Input activations [N, D]
            expert_weights: Routing weights [N, top_k] (already softmaxed)
            expert_indices: Expert assignments [N, top_k]
            top_k: Number of experts per token
            sort_experts: Sort tokens by expert index for coalesced memory
                access during prefill. No effect on decode (single token).

        Returns:
            Output tensor [N, D]
        """
        N = x.shape[0]

        if sort_experts:
            flat_indices = expert_indices.flatten()
            order = flat_indices.argsort().to(torch.int32)
            inv_order = order.argsort().to(torch.int32)
            sorted_idx = flat_indices[order].to(torch.int32)
            x_sorted = x[(order // top_k).to(torch.int64)]
            x_input = x_sorted.unsqueeze(-2)
            idx = sorted_idx
        else:
            x_input = x.unsqueeze(-2).unsqueeze(-2)
            idx = expert_indices

        if self.fuse_gate_up:
            gate_up = self.gate_up_proj(x_input, idx, sorted_indices=sort_experts)
            gate = gate_up[..., : self.intermediate_size]
            up = gate_up[..., self.intermediate_size :]
        else:
            gate = self.gate_proj(x_input, idx, sorted_indices=sort_experts)
            up = self.up_proj(x_input, idx, sorted_indices=sort_experts)
        h = self.activation(gate) * up
        down = self.down_proj(h, idx, sorted_indices=sort_experts)

        if sort_experts:
            down = down.squeeze(-2)
            down = down[inv_order].reshape(N, top_k, -1)
        else:
            down = down.squeeze(-2)

        return (down * expert_weights.unsqueeze(-1)).sum(dim=-2)


def pack_all_switch_linears(model: nn.Module) -> int:
    """Call pack() on all SwitchLinear modules in the model.

    Args:
        model: The model to pack

    Returns:
        Number of SwitchLinear modules packed
    """
    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, SwitchLinear):
            module.pack()
            count += 1
    if count > 0:
        logger.info(f"Packed {count} SwitchLinear modules")
    return count
