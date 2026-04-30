# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize weights to canonical form.

``quantize_weight`` quantizes a single tensor given a ``QuantConfig``,
dispatching to the appropriate algorithm based on ``config.method``:

  - ``"min_max"``: standard symmetric/asymmetric quantization via torchao's
    ``choose_qparams_affine`` + ``quantize_affine``. Runs on CPU or CUDA.
  - ``"hqq"``: Half-Quadratic Quantization — iteratively refines scales via
    a proximal solver for better accuracy. ``symmetric=False`` optimizes both
    scale and zero (requires CUDA). ``symmetric=True`` optimizes scale only
    (CPU or CUDA).

``quantize_model`` walks a model's parameters, applies a ``QuantRecipe``,
and returns two dicts: quantized weights as ``CanonicalQuantizedWeight``
and unquantized weights as plain tensors.

Both are model-agnostic — they work for any ``nn.Module`` and any weight
shape (2D linears, 3D fused-expert stacks, etc.).
"""

import torch
import torch.nn as nn

from .recipe import QuantConfig, QuantRecipe

from .serialize import CanonicalQuantizedWeight


# ---------------------------------------------------------------------------
# Per-weight quantization


def _quantize_min_max(
    weight: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard min/max quantization. Returns (int_data, scale, zero_point)."""
    from torchao.quantization.quant_primitives import (
        choose_qparams_affine,
        MappingType,
        quantize_affine,
    )

    if config.bits == 4:
        qmin, qmax = (-8, 7) if config.symmetric else (0, 15)
    elif config.bits == 8:
        qmin, qmax = -128, 127
    else:
        raise ValueError(f"Unsupported bits={config.bits}")

    mapping = MappingType.SYMMETRIC if config.symmetric else MappingType.ASYMMETRIC
    block_size = tuple([1] * (weight.ndim - 1) + [config.group_size])

    scale, zero_point = choose_qparams_affine(
        weight.float(),
        mapping,
        block_size,
        target_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
        scale_dtype=torch.bfloat16,
        zero_point_dtype=torch.bfloat16,
    )
    int_data = quantize_affine(
        weight.float(),
        block_size,
        scale,
        zero_point,
        output_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
    )
    return int_data, scale, zero_point


def _quantize_hqq_asymmetric(
    weight: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full HQQ (asymmetric, optimizes scale + zero). Requires CUDA.

    Returns (int_data, scale, zero_point) in canonical layout.
    """
    from torchao.quantization.quant_primitives import (
        _choose_qparams_and_quantize_affine_hqq,
    )

    device = weight.device
    if device.type != "cuda":
        device = torch.device("cuda")

    W_q, scale, zero, _shape = _choose_qparams_and_quantize_affine_hqq(
        weight,
        nbits=config.bits,
        group_size=config.group_size,
        axis=1,
        compute_dtype=torch.bfloat16,
        device=str(device),
        raw_output=True,
    )

    int_data = W_q.to(torch.int8)
    scale = scale.to(torch.bfloat16).reshape(*weight.shape[:-1], -1)
    zero = zero.to(torch.bfloat16).reshape(*weight.shape[:-1], -1)

    return int_data, scale, zero


def _quantize_hqq_symmetric(
    weight: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-only HQQ (symmetric, optimizes scale only). Runs on CPU or CUDA.

    Returns (int_data, scale, zero_point) where zero_point is all zeros.
    """
    from torchao.quantization.quant_primitives import (
        _choose_qparams_and_quantize_scale_only_hqq,
    )

    if config.bits == 4:
        qmin, qmax = -8, 7
    elif config.bits == 8:
        qmin, qmax = -128, 127
    else:
        raise ValueError(f"Unsupported bits={config.bits}")

    # scale_only_hqq requires 2D. For 3D+, flatten → quantize → reshape.
    orig_shape = weight.shape
    weight_2d = weight.reshape(-1, weight.shape[-1]) if weight.ndim > 2 else weight

    qdata, scale = _choose_qparams_and_quantize_scale_only_hqq(
        weight_2d,
        [1, config.group_size],
        qmin,
        qmax,
    )

    int_data = qdata.to(torch.int8).reshape(orig_shape)
    scale = scale.to(torch.bfloat16).reshape(*orig_shape[:-1], -1)
    zero_point = torch.zeros_like(scale)

    return int_data, scale, zero_point


def quantize_weight(
    weight: torch.Tensor,
    config: QuantConfig,
) -> CanonicalQuantizedWeight:
    """Quantize ``weight`` to canonical form.

    Dispatches to the algorithm specified by ``config.method``. The input is
    processed in float32 internally for numerical stability. Does NOT pad or
    pack for any backend.
    """
    if config.method == "min_max":
        int_data, scale, zero_point = _quantize_min_max(weight, config)
    elif config.method == "hqq":
        if config.symmetric:
            int_data, scale, zero_point = _quantize_hqq_symmetric(weight, config)
        else:
            int_data, scale, zero_point = _quantize_hqq_asymmetric(weight, config)
    else:
        raise ValueError(
            f"Unknown quantization method: {config.method!r}. "
            f"Supported: 'min_max', 'hqq'."
        )

    # Normalize 4-bit to unsigned [0, 15] for uniform storage and nibble
    # packing. Symmetric min_max produces [-8, 7]; shift to [0, 15].
    # HQQ already produces [0, 15] (asymmetric internally).
    if config.bits == 4 and config.symmetric:
        int_data = int_data + 8

    return CanonicalQuantizedWeight(
        qdata=int_data.to(torch.int8),
        scale=scale.to(torch.bfloat16),
        zero=zero_point.to(torch.bfloat16) if not config.symmetric else None,
        config=config,
    )


def dequantize_weight(
    cw: CanonicalQuantizedWeight,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a ``CanonicalQuantizedWeight`` back to a dense tensor.

    Inverse of ``quantize_weight``. Useful for embedding lookups (which
    need dense weights) or for inspecting quantized values.
    """
    gs = cw.config.group_size
    scale = cw.scale.float().repeat_interleave(gs, dim=-1)
    if cw.zero is not None:
        zero = cw.zero.float().repeat_interleave(gs, dim=-1)
        return ((cw.qdata.float() - zero) * scale).to(dtype)
    return (cw.qdata.float() * scale).to(dtype)


# ---------------------------------------------------------------------------
# Per-model quantization


def quantize_model(
    model: nn.Module,
    recipe: QuantRecipe,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, CanonicalQuantizedWeight], dict[str, torch.Tensor]]:
    """Walk model parameters + persistent buffers, apply recipe.

    For each parameter matched by a recipe rule: quantize to canonical.
    Parameters that match ``None`` (skip) rules and persistent buffers go
    into the unquantized dict (cast to ``dtype``). Non-persistent buffers
    (KV cache, RoPE tables, etc.) are excluded.

    Returns ``(quantized, unquantized)`` dicts keyed by FQN.
    """
    quantized: dict[str, CanonicalQuantizedWeight] = {}
    unquantized: dict[str, torch.Tensor] = {}
    persistent_keys = set(model.state_dict().keys())

    n_params = sum(1 for _ in model.named_parameters())
    for i, (fqn, param) in enumerate(model.named_parameters()):
        config = recipe.get_config(fqn)
        if config is None:
            unquantized[fqn] = param.data.to(dtype)
        else:
            quantized[fqn] = quantize_weight(param.data, config)
            print(f"  Quantized {i + 1}/{n_params}: {fqn}", end="\r")
    print()

    for fqn, buf in model.named_buffers():
        if fqn in persistent_keys and fqn not in quantized:
            unquantized[fqn] = buf.data

    return quantized, unquantized
