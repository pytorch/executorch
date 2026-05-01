# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize weights to torchao tensor subclasses.

``quantize_weight`` quantizes a single tensor given a ``QuantConfig``,
returning an ``Int4Tensor`` (4-bit) or ``IntxUnpackedToInt8Tensor`` (8-bit).

``quantize_model`` walks a model's parameters, applies a ``QuantRecipe``,
and returns a single state dict containing both quantized subclass tensors
and unquantized plain tensors.
"""

import torch
import torch.nn as nn

from .recipe import QuantConfig, QuantRecipe


# ---------------------------------------------------------------------------
# Per-weight quantization


def _quantize_min_max(
    weight: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard min/max 4-bit quantization. Returns (int_data, scale, zero_point)."""
    from torchao.quantization.quant_primitives import (
        choose_qparams_affine,
        MappingType,
        quantize_affine,
    )

    qmin, qmax = (-8, 7) if config.symmetric else (0, 15)

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
    """Full HQQ (asymmetric, optimizes scale + zero). Requires CUDA."""
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
    """Scale-only HQQ (symmetric 4-bit, optimizes scale only). Runs on CPU or CUDA."""
    from torchao.quantization.quant_primitives import (
        _choose_qparams_and_quantize_scale_only_hqq,
    )

    qmin, qmax = -8, 7

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


def _to_int4_tensor(
    int_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    config: QuantConfig,
) -> torch.Tensor:
    """Wrap quantized 4-bit data into an Int4Tensor."""
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    # Normalize 4-bit signed [-8, 7] to unsigned [0, 15] for storage.
    if config.symmetric:
        int_data = int_data + 8
        zero_point = torch.full_like(scale, 8.0)

    # Int4Tensor stores qdata as nibble-packed uint8 (N, K//2)
    q = int_data.to(torch.uint8)
    packed = q[..., ::2] | (q[..., 1::2] << 4)

    # Int4Tensor stores scale/zero as (K//gs, N) — transposed from our (N, K//gs)
    return Int4Tensor(
        qdata=packed,
        scale=scale.t().contiguous(),
        zero_point=zero_point.t().contiguous(),
        block_size=[1, config.group_size],
        shape=torch.Size(int_data.shape),
    )


def _to_intx_tensor(
    weight: torch.Tensor,
    config: QuantConfig,
) -> torch.Tensor:
    """Quantize 8-bit and wrap in IntxUnpackedToInt8Tensor.

    Quantizes in float32 for numerical precision, then constructs the
    subclass directly. We avoid ``from_hp`` because it quantizes in the
    input dtype (bf16), which loses precision for small-magnitude weights.
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quant_primitives import (
        choose_qparams_affine,
        MappingType,
        quantize_affine,
    )

    if config.method == "hqq":
        if not config.symmetric:
            raise ValueError(
                "8-bit HQQ only supports symmetric quantization "
                "(HQQ_SCALE_ONLY). Use method='min_max' for asymmetric 8-bit."
            )
        from torchao.quantization.quant_primitives import (
            _choose_qparams_and_quantize_scale_only_hqq,
        )

        w2d = weight.float().reshape(-1, weight.shape[-1])
        qdata, scale = _choose_qparams_and_quantize_scale_only_hqq(
            w2d, [1, config.group_size], -128, 127
        )
        qdata = qdata.to(torch.int8).reshape(weight.shape)
        scale = scale.to(torch.bfloat16).reshape(weight.shape[0], -1)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
    else:
        mapping = MappingType.SYMMETRIC if config.symmetric else MappingType.ASYMMETRIC
        block_size = (1, config.group_size)
        scale, zero_point = choose_qparams_affine(
            weight.float(),
            mapping,
            block_size,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            scale_dtype=torch.bfloat16,
            zero_point_dtype=torch.int8,
        )
        qdata = quantize_affine(
            weight.float(),
            block_size,
            scale,
            zero_point,
            output_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        N, n_groups = weight.shape[0], weight.shape[-1] // config.group_size
        scale = scale.reshape(N, n_groups)
        zero_point = zero_point.reshape(N, n_groups)

    return IntxUnpackedToInt8Tensor(
        qdata=qdata,
        scale=scale,
        zero_point=zero_point,
        target_dtype=torch.int8,
        block_size=(1, config.group_size),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )


def quantize_weight(weight: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Quantize ``weight`` to a torchao tensor subclass.

    Returns ``Int4Tensor`` for 4-bit or ``IntxUnpackedToInt8Tensor`` for 8-bit.
    """
    if config.bits == 8:
        return _to_intx_tensor(weight, config)

    if config.bits != 4:
        raise ValueError(f"Unsupported bits={config.bits}")

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

    return _to_int4_tensor(int_data, scale, zero_point, config)


def dequantize_weight(
    weight: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a torchao quantized tensor back to a dense tensor."""
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    if isinstance(weight, Int4Tensor):
        # Unpack nibbles
        p = weight.qdata.to(torch.uint8)
        low = (p & 0x0F).float()
        high = ((p >> 4) & 0x0F).float()
        qdata = torch.stack([low, high], dim=-1).reshape(weight.shape)
        # Scale is (K//gs, N), transpose to (N, K//gs) for broadcast
        gs = weight.block_size[-1]
        scale = weight.scale.t().float().repeat_interleave(gs, dim=-1)
        zero = weight.zero_point.t().float().repeat_interleave(gs, dim=-1)
        return ((qdata - zero) * scale).to(dtype)

    if isinstance(weight, IntxUnpackedToInt8Tensor):
        gs = weight.block_size[-1]
        scale = weight.scale.float().repeat_interleave(gs, dim=-1)
        zero = weight.zero_point.float().repeat_interleave(gs, dim=-1)
        return ((weight.qdata.float() - zero) * scale).to(dtype)

    raise TypeError(f"Cannot dequantize {type(weight).__name__}")


# ---------------------------------------------------------------------------
# Per-model quantization


def quantize_model(
    model: nn.Module,
    recipe: QuantRecipe,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Walk model parameters + persistent buffers, apply recipe.

    Returns a single state dict containing quantized tensor subclasses
    (``Int4Tensor``, ``IntxUnpackedToInt8Tensor``) and unquantized plain
    tensors. Non-persistent buffers (KV cache, RoPE tables) are excluded.
    """
    state: dict[str, torch.Tensor] = {}
    persistent_keys = set(model.state_dict().keys())

    n_params = sum(1 for _ in model.named_parameters())
    for i, (fqn, param) in enumerate(model.named_parameters()):
        config = recipe.get_config(fqn)
        if config is None:
            state[fqn] = param.data.to(dtype)
        else:
            state[fqn] = quantize_weight(param.data, config)
            print(f"  Quantized {i + 1}/{n_params}: {fqn}", end="\r")
    print()

    for fqn, buf in model.named_buffers():
        if fqn in persistent_keys and fqn not in state:
            state[fqn] = buf.data

    return state
