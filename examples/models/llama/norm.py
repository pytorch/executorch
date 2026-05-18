# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            add_unit_offset (bool, optional): Whether to scale normalized output by
                `(1 + weight)` instead of `weight`. Default is False.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return output * (1.0 + self.weight.float()).type_as(x)
        return output * self.weight.type_as(x)


class ScalelessRMSNorm(torch.nn.RMSNorm):
    """RMSNorm with weight hardcoded to ones and not trainable.

    Equivalent to a scaleless RMSNorm (no learnable scaling) but implemented as a
    torch.nn.RMSNorm so the op composes/decomposes cleanly for backends like QNN
    instead of being expressed as a hand-rolled decomposition.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)
        self.dim = dim
        with torch.no_grad():
            self.weight.fill_(1.0)
        self.weight.requires_grad = False


class RMSNormCoreML(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        CoreML-friendly RMSNorm — uses `torch.linalg.vector_norm` so the op is
        preserved in the CoreML graph for numerical stability.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): Floor on the L2-norm denominator
                (`clamp_min(‖x‖₂, √(dim·eps))`). Prevents `0/0 = NaN` on
                zero-padded positions and matches standard RMSNorm's
                `rsqrt(mean(x²) + eps)` semantics on a zero input. Must be > 0.

        Attributes:
            eps (float): Floor coefficient consumed by `_norm`.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        assert eps > 0, (
            "RMSNormCoreML requires eps > 0; eps=0 collapses the denominator "
            "floor and produces NaN on zero-padded positions"
        )
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Floor the denominator to avoid 0 / 0 = NaN on zero-padded positions
        # (chunked prefill in StaticAttentionIOManager pads each chunk to
        # input_len with zeros). Use sqrt(dim * eps) so the floor matches
        # standard RMSNorm's eps semantics (`rsqrt(mean(x²) + eps)`) and is
        # large enough to survive fp16 (1e-6 alone underflows in fp16).
        floor_val = torch.sqrt(torch.tensor(self.dim * self.eps, dtype=x.dtype))
        norm_val = torch.clamp_min(
            torch.linalg.vector_norm(x, dim=-1, keepdim=True), floor_val
        )
        rms_norm_eps0 = (
            x
            * torch.sqrt(torch.tensor(self.dim, dtype=x.dtype))
            * torch.reciprocal(norm_val)
        )
        return rms_norm_eps0

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class RMSNormWithInputScale(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scaled = self.weight * x
        return F.rms_norm(scaled, (self.dim,), None, self.eps)


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def replace_rms_norm_for_coreml_(model: torch.nn.Module) -> torch.nn.Module:
    """In-place: walk `model` and swap every RMSNorm-family module for RMSNormCoreML.

    Mirrors the post-construction transform pattern used by torchao's
    `quantize_(model, config)`: instead of threading a `use_coreml_norm` flag
    through every norm construction site, build the model with the standard
    norms and then call this once before CoreML export. Trained scale weights
    are preserved.

    Swaps these classes (everything else is left alone):
      * `RMSNorm` (this module)
      * `ScalelessRMSNorm` (this module — no-op weight)
      * `torch.nn.RMSNorm` (used for affine q_norm/k_norm in StaticAttention)
    """
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, (RMSNorm, ScalelessRMSNorm, torch.nn.RMSNorm)):
            continue
        # All three carry the normalized dim either as `dim` or in `normalized_shape[-1]`.
        dim = getattr(mod, "dim", None) or mod.normalized_shape[-1]
        eps = getattr(mod, "eps", 1e-6) or 1e-6
        new = RMSNormCoreML(dim, eps=eps)
        # Preserve trained scale (no-op for ScalelessRMSNorm).
        if getattr(mod, "weight", None) is not None:
            new.weight = mod.weight
        # Locate parent module via the dotted name and rebind the attribute.
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, new)
    return model
