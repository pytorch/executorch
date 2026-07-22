# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Source transformation that replaces `MOEFeedForward`
modules with a `QuantizedMoEFFN` module wrapping the
`llama::quantized_moe_ffn` portable-runtime custom op.

The transform symmetrically INT4-quantizes each per-expert weight matrix
(group_size=32 by default), packs each expert with torchao's
`torchao::_pack_8bit_act_4bit_weight` op, and stacks the per-expert
opaque blobs into `[E, packed_bytes]` buffers consumed by the custom op.
"""

from __future__ import annotations

import logging

import torch

from executorch.examples.models.llama.llama_transformer import MOEFeedForward
from torch import nn
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    MappingType,
    quantize_affine,
)


logger: logging.Logger = logging.getLogger(__name__)


def _symmetric_quantize_per_group(
    w: torch.Tensor, group_size: int, n_bit: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group quantization using torchao primitives.

    Returns:
      qvals: int8 tensor of shape [N, K].
      scales: float32 tensor of shape [N * (K // group_size)].
    """
    qmin = -(1 << (n_bit - 1))
    qmax = (1 << (n_bit - 1)) - 1
    block_size = (1, group_size)
    scale, zero_point = choose_qparams_affine(
        w.float(),
        MappingType.SYMMETRIC,
        block_size,
        target_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
    )
    qvals = quantize_affine(
        w.float(),
        block_size,
        scale,
        zero_point,
        output_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
    )
    return qvals, scale.reshape(-1).to(torch.float32)


def _torchao_pack_int4_weight(
    w: torch.Tensor, group_size: int, target: str = "universal"
) -> torch.Tensor:
    """Symmetric INT4 group-quantize + torchao pack a 2D weight `[N, K]`."""
    qvals, scales = _symmetric_quantize_per_group(w, group_size, n_bit=4)
    return torch.ops.torchao._pack_8bit_act_4bit_weight(
        qvals,
        scales,
        None,
        group_size,
        None,
        target,
    )


def _torchao_pack_int8_weight(
    w: torch.Tensor, group_size: int, target: str = "universal"
) -> torch.Tensor:
    """Symmetric INT8 group-quantize + torchao pack a 2D weight `[N, K]`."""
    qvals, scales = _symmetric_quantize_per_group(w, group_size, n_bit=8)
    return torch.ops.torchao._pack_8bit_act_8bit_weight(
        qvals,
        scales,
        None,
        group_size,
        None,
        target,
    )


class QuantizedMoEFFN(nn.Module):
    """Drop-in replacement for `MOEFeedForward` that calls
    `torch.ops.llama.quantized_moe_ffn`.

    Buffers (registered, not parameters):
      gate_weight  [E, D]   fp32  — copied from `MOEFeedForward.gate.weight`
      expert_bias  [E] or [0]  fp32 — empty when `use_expert_bias=False`
      packed_w13   [E, packed_bytes_w13]  uint8 — fused w1+w3 torchao blobs
      packed_w2    [E, packed_bytes_w2]   uint8
    """

    def __init__(
        self,
        gate_weight: torch.Tensor,
        expert_bias: torch.Tensor | None,
        packed_w13: torch.Tensor,
        packed_w2: torch.Tensor,
        *,
        num_experts: int,
        num_activated_experts: int,
        hidden_dim: int,
        dim: int,
        group_size: int,
        weight_nbit: int,
        score_func: str,
        route_scale: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts
        self.group_size = group_size
        self.weight_nbit = weight_nbit
        self.score_func = score_func
        self.route_scale = float(route_scale)

        self.register_buffer(
            "gate_weight",
            gate_weight.dequantize().detach().clone().to(torch.float32),
        )
        # Always register an `expert_bias` buffer; size 0 means "not used".
        if expert_bias is None:
            expert_bias_buf = torch.empty(0, dtype=torch.float32)
        else:
            expert_bias_buf = expert_bias.to(torch.float32)
        self.register_buffer("expert_bias", expert_bias_buf)

        # torchao packed blobs are int8 tensors; reinterpret the bytes as
        # uint8 (no value conversion) for the op schema.
        self.register_buffer("packed_w13", packed_w13.view(torch.uint8))
        self.register_buffer("packed_w2", packed_w2.view(torch.uint8))
        self.shared_expert: nn.Module | None = None

    # The C++ kernel requires fp32 gate_weight / expert_bias.  The export
    # pipeline calls model.to(dtype) after quantisation which would downcast
    # these buffers.  Override _apply so they stay fp32 regardless.
    _FP32_BUFFER_NAMES = frozenset({"gate_weight", "expert_bias"})

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for name in self._FP32_BUFFER_NAMES:
            buf = getattr(self, name, None)
            if (
                buf is not None
                and buf.is_floating_point()
                and buf.dtype != torch.float32
            ):
                setattr(self, name, buf.to(torch.float32))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_flat = x.reshape(-1, self.dim).to(torch.float32)
        out = torch.ops.llama.quantized_moe_ffn(
            x_flat,
            self.gate_weight,
            self.expert_bias,
            self.packed_w13,
            self.packed_w2,
            self.num_activated_experts,
            self.num_experts,
            self.hidden_dim,
            self.dim,
            self.group_size,
            self.weight_nbit,
            self.score_func,
            self.route_scale,
        )
        if self.shared_expert is not None:
            out = out + self.shared_expert(x_flat)
        out = out.view(x.shape[:-1] + (self.dim,))
        return out.to(input_dtype) if input_dtype != torch.float32 else out


def _stack_per_expert_packed(
    per_expert_blobs: list[torch.Tensor],
) -> torch.Tensor:
    """Stack `[E]` per-expert packed 1D blobs into a `[E, packed_bytes]`
    tensor. All blobs are required to have the same length (they will,
    because every expert has the same `[N, K]` shape and we use the same
    quant config)."""
    if not per_expert_blobs:
        raise ValueError("per_expert_blobs must be non-empty")
    expected_size = per_expert_blobs[0].numel()
    for i, blob in enumerate(per_expert_blobs):
        if blob.numel() != expected_size:
            raise ValueError(
                f"per-expert packed blob {i} size {blob.numel()} != {expected_size}"
            )
    return torch.stack([blob.reshape(-1) for blob in per_expert_blobs], dim=0)


def _build_quantized_moe_ffn_from_eager(
    moe: MOEFeedForward,
    *,
    group_size: int,
    weight_nbit: int,
) -> QuantizedMoEFFN:
    """Construct a QuantizedMoEFFN from an existing eager MOEFeedForward."""
    cond = moe.cond_ffn
    e = cond.num_experts
    w1 = cond.w1  # [E, F, D]
    w3 = cond.w3  # [E, F, D]
    w2 = cond.w2  # [E, F, D]
    if not (w1.dim() == 3 and w3.dim() == 3 and w2.dim() == 3):
        raise ValueError(
            f"expert weights must be 3D [E, F, D], got "
            f"w1={tuple(w1.shape)}, w3={tuple(w3.shape)}, w2={tuple(w2.shape)}"
        )

    f_dim, d_dim = w1.shape[1], w1.shape[2]

    # torchao group-quantization requires the packed K dim to be a multiple of
    # group_size: K=D for w1/w3 and K=F for the transposed w2. Validate here so
    # non-default model dims / group sizes fail with a clear error rather than
    # an obscure torchao failure or a runtime-rejected packed buffer.
    if d_dim % group_size != 0:
        raise ValueError(
            f"w1/w3 K dim D={d_dim} not divisible by group_size={group_size}"
        )
    if f_dim % group_size != 0:
        raise ValueError(f"w2 K dim F={f_dim} not divisible by group_size={group_size}")

    # w2 is [E, F, D] for the einsum path; our op packs [N, K] so
    # transpose to [E, D, F] before packing.
    w2_packed_in = w2.transpose(-2, -1).contiguous()  # [E, D, F]

    pack_fn = (
        _torchao_pack_int4_weight if weight_nbit == 4 else _torchao_pack_int8_weight
    )

    packed_w13_list: list[torch.Tensor] = []
    packed_w2_list: list[torch.Tensor] = []
    for ei in range(e):
        # Fuse w1 and w3 into a single [2F, D] matrix before packing.
        w13 = torch.cat([w1[ei], w3[ei]], dim=0)  # [2F, D]
        packed_w13_list.append(pack_fn(w13, group_size))
        packed_w2_list.append(pack_fn(w2_packed_in[ei], group_size))

    packed_w13 = _stack_per_expert_packed(packed_w13_list)
    packed_w2 = _stack_per_expert_packed(packed_w2_list)

    expert_bias = (
        moe.expert_bias.detach().clone()
        if getattr(moe, "use_expert_bias", False) and hasattr(moe, "expert_bias")
        else None
    )

    replacement = QuantizedMoEFFN(
        gate_weight=moe.gate.weight.detach().clone(),
        expert_bias=expert_bias,
        packed_w13=packed_w13,
        packed_w2=packed_w2,
        num_experts=e,
        num_activated_experts=moe.num_activated_experts,
        hidden_dim=f_dim,
        dim=d_dim,
        group_size=group_size,
        weight_nbit=weight_nbit,
        score_func=moe.score_func,
        route_scale=moe.route_scale,
    )
    # The shared expert (when present) is intentionally left in its original
    # eager/fp32 form: it is a dense FFN, not a per-expert quantized MoE, so
    # it is not routed through `llama::quantized_moe_ffn`.
    if getattr(moe, "shared_expert", None) is not None:
        replacement.shared_expert = moe.shared_expert
    return replacement


def _replace_moe_recursive(
    module: nn.Module, group_size: int, weight_nbit: int
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, MOEFeedForward):
            replacement = _build_quantized_moe_ffn_from_eager(
                child,
                group_size=group_size,
                weight_nbit=weight_nbit,
            )
            setattr(module, name, replacement)
            logger.info(
                "Replaced MOEFeedForward at %s with QuantizedMoEFFN "
                "(E=%d, A=%d, D=%d, F=%d, score=%s, route_scale=%s)",
                name,
                replacement.num_experts,
                replacement.num_activated_experts,
                replacement.dim,
                replacement.hidden_dim,
                replacement.score_func,
                replacement.route_scale,
            )
        else:
            _replace_moe_recursive(child, group_size, weight_nbit)


def replace_moe_with_quantized_op(
    model: nn.Module,
    *,
    group_size: int = 32,
    weight_nbit: int = 4,
) -> nn.Module:
    """Walk `model` and swap every `MOEFeedForward` for a `QuantizedMoEFFN`.

    Args:
      model: the eager model graph (typically the result of
        `Llama4BackboneModel.get_eager_model()`).
      group_size: torchao quantization group size. 32 matches MobileMoE-0.3B.
      weight_nbit: 4 (production) or 8 (debug).

    Returns:
      The mutated model (same object).
    """
    # Imported here rather than at module scope so that merely importing this
    # module has no op-registration side effect: export flows that pull in this
    # file (it lives in the shared llama source_transformation library) but do
    # not use the quantized MoE op must not load the custom-ops AOT library.
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    if not hasattr(torch.ops.llama, "quantized_moe_ffn"):
        raise RuntimeError(
            "llama::quantized_moe_ffn is not registered. "
            "Ensure executorch.extension.llm.custom_ops.custom_ops is "
            "imported and the AOT library is loaded."
        )
    if weight_nbit not in (4, 8):
        raise ValueError(f"weight_nbit must be 4 or 8, got {weight_nbit}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    _replace_moe_recursive(model, group_size, weight_nbit)
    return model
