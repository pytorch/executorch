# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize weights to torchao tensor subclasses.

``quantize_weight`` quantizes a single tensor given a ``QuantConfig``,
returning an ``Int4Tensor`` (4-bit) or ``IntxUnpackedToInt8Tensor`` (5-, 6-,
or 8-bit).

``quantize_model`` walks a model's parameters, applies a ``QuantRecipe``,
and returns a single state dict containing both quantized subclass tensors
and unquantized plain tensors.

``quantize_stream`` and ``quantize_model`` differ by design: ``quantize_stream``
is the *serialization* producer (yields torchao-native ``Int4Tensor``, which
round-trips through torchao's safetensors allowlist and is what
``quantize_and_save`` writes to disk), while ``quantize_model`` is the
*in-memory model* producer -- the dual of :func:`load_checkpoint`, applying
``convert`` (default ``to_default``: ``Int4Tensor -> ExportableInt4Tensor``)
then :func:`maybe_cast` so the state dict is castable (e.g. fp16 for MLX) and
in the export-canonical form the backend packers consume.
"""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn

from .convert import Convert, maybe_cast, to_default
from .recipe import QuantConfig, QuantRecipe

# ---------------------------------------------------------------------------
# Per-weight quantization


def _quantize_affine_min_max(
    weight: torch.Tensor,
    group_size: int,
    qmin: int,
    qmax: int,
    symmetric: bool,
    zero_point_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine min/max quantization. Returns (int_data, scale, zero_point).

    ``int_data`` is int8 in [qmin, qmax]; ``scale``/``zero_point`` keep the
    ``choose_qparams_affine`` block-reduced shape.
    """
    from torchao.quantization.quant_primitives import (
        choose_qparams_affine,
        MappingType,
        quantize_affine,
    )

    mapping = MappingType.SYMMETRIC if symmetric else MappingType.ASYMMETRIC
    block_size = tuple([1] * (weight.ndim - 1) + [group_size])
    weight = weight.float()
    scale, zero_point = choose_qparams_affine(
        weight,
        mapping,
        block_size,
        target_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
        scale_dtype=torch.bfloat16,
        zero_point_dtype=zero_point_dtype,
    )
    int_data = quantize_affine(
        weight,
        block_size,
        scale,
        zero_point,
        output_dtype=torch.int8,
        quant_min=qmin,
        quant_max=qmax,
    )
    return int_data, scale, zero_point


def _quantize_scale_only_hqq(
    weight_2d: torch.Tensor,
    group_size: int,
    qmin: int,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale-only HQQ on a 2D weight. Returns (int_data int8, scale bf16)."""
    from torchao.quantization.quant_primitives import (
        _choose_qparams_and_quantize_scale_only_hqq,
    )

    qdata, scale = _choose_qparams_and_quantize_scale_only_hqq(
        weight_2d, [1, group_size], qmin, qmax
    )
    return qdata.to(torch.int8), scale.to(torch.bfloat16)


def _quantize_min_max(
    weight: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard min/max 4-bit quantization. Returns (int_data, scale, zero_point)."""
    qmin, qmax = (-8, 7) if config.symmetric else (0, 15)
    return _quantize_affine_min_max(
        weight,
        config.group_size,
        qmin,
        qmax,
        config.symmetric,
        zero_point_dtype=torch.bfloat16,
    )


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
    orig_shape = weight.shape
    weight_2d = weight.reshape(-1, weight.shape[-1]) if weight.ndim > 2 else weight
    int_data, scale = _quantize_scale_only_hqq(weight_2d, config.group_size, -8, 7)
    int_data = int_data.reshape(orig_shape)
    scale = scale.reshape(*orig_shape[:-1], -1)
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
    """Quantize to 5-, 6-, or 8-bit and wrap in IntxUnpackedToInt8Tensor.

    Quantizes in float32 for numerical precision, then constructs the
    subclass directly. We avoid ``from_hp`` because it quantizes in the
    input dtype (bf16), which loses precision for small-magnitude weights.

    Sub-byte data (e.g. 6-bit) is still stored unpacked in an int8 container;
    ``target_dtype`` records the true bit width for the export/runtime path.
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor

    qmin = -(1 << (config.bits - 1))
    qmax = (1 << (config.bits - 1)) - 1
    target_dtype = getattr(torch, f"int{config.bits}")

    if config.method == "hqq":
        if not config.symmetric:
            raise ValueError(
                "intx HQQ only supports symmetric quantization "
                "(HQQ_SCALE_ONLY). Use method='min_max' for asymmetric intx."
            )
        w2d = weight.float().reshape(-1, weight.shape[-1])
        qdata, scale = _quantize_scale_only_hqq(w2d, config.group_size, qmin, qmax)
        qdata = qdata.reshape(weight.shape)
        scale = scale.reshape(weight.shape[0], -1)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
    else:
        qdata, scale, zero_point = _quantize_affine_min_max(
            weight,
            config.group_size,
            qmin,
            qmax,
            config.symmetric,
            zero_point_dtype=torch.int8,
        )
        N, n_groups = weight.shape[0], weight.shape[-1] // config.group_size
        scale = scale.reshape(N, n_groups)
        zero_point = zero_point.reshape(N, n_groups)

    return IntxUnpackedToInt8Tensor(
        qdata=qdata,
        scale=scale,
        zero_point=zero_point,
        target_dtype=target_dtype,
        block_size=(1, config.group_size),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )


def quantize_weight(weight: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Quantize ``weight`` to a torchao tensor subclass.

    Returns ``Int4Tensor`` for 4-bit or ``IntxUnpackedToInt8Tensor`` for 5-,
    6-, and 8-bit.
    """
    if config.bits in (5, 6, 8):
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
    from executorch.extension.llm.export.int4 import ExportableInt4Tensor
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    if isinstance(weight, Int4Tensor):
        return ExportableInt4Tensor.from_int4_tensor(weight).dequantize(dtype)

    if isinstance(weight, ExportableInt4Tensor):
        return weight.dequantize(dtype)

    if isinstance(weight, IntxUnpackedToInt8Tensor):
        return weight.to(dtype).dequantize()

    # ExportableGGUFTensor (native GGUF Q4_K/Q6_K) carries its own gguf-package
    # dequant. The tied CUDA token embedding keeps the raw GGUF tensor and is
    # dequantized to bf16 here for the gather. Imported lazily to avoid a hard
    # extension/llm dependency.
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor

    if isinstance(weight, ExportableGGUFTensor):
        return weight.dequantize(dtype)

    # CudaDp4aPlanarInt6Tensor (GGUF Q6_K on CUDA) carries its own dequant
    # (symmetric, ql/qh split bit-planes). Imported lazily to avoid a hard
    # backends/cuda dependency.
    from executorch.backends.cuda.dp4a_planar_int6_tensor import (
        CudaDp4aPlanarInt6Tensor,
    )

    if isinstance(weight, CudaDp4aPlanarInt6Tensor):
        return weight.dequantize(dtype)

    raise TypeError(f"Cannot dequantize {type(weight).__name__}")


# ---------------------------------------------------------------------------
# Per-weight streaming quantization


def quantize_stream(
    pairs: Iterable[tuple[str, torch.Tensor]],
    recipe: QuantRecipe,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Lazy, per-weight dual of :func:`quantize_model`.

    For each ``(fqn, tensor)`` pair, apply the recipe -- quantize to a torchao
    subclass when ``recipe.get_config(fqn)`` matches, else keep the tensor -- then
    :func:`maybe_cast` the result to ``dtype`` (a no-op for non-float, already
    matching, or non-castable values, so a routed int buffer is left intact and a
    quantized subclass only has its dequantized output dtype re-stamped).
    Model-free -- the caller owns iteration order and decides what to feed in
    (typically a module's parameters).

    This is the *serialization* form: int4 weights stay as torchao-native
    ``Int4Tensor`` (a no-op under ``maybe_cast``, so int4 is always bf16 here),
    which round-trips through torchao's safetensors allowlist and is what
    ``quantize_and_save`` writes to disk. The ``dtype`` re-stamp for int4 happens
    later at the convert boundary (:func:`load_checkpoint` or
    :func:`quantize_model`, which wrap it as an ``ExportableInt4Tensor`` first).
    """
    for fqn, tensor in pairs:
        config = recipe.get_config(fqn)
        value = tensor if config is None else quantize_weight(tensor, config)
        yield fqn, maybe_cast(value, dtype)


# ---------------------------------------------------------------------------
# Per-model quantization


def quantize_model(
    model: nn.Module,
    recipe: QuantRecipe,
    *,
    convert: Convert = to_default,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Walk model parameters + persistent buffers, apply recipe.

    Returns a single state dict for *in-memory* model use (the dual of
    :func:`load_checkpoint`). Parameters run through :func:`quantize_stream`,
    then each value is ``convert``-ed (default ``to_default``:
    ``Int4Tensor -> ExportableInt4Tensor``) and :func:`maybe_cast` to ``dtype``
    -- convert-then-cast, so a raw int4 weight is wrapped as an
    ``ExportableInt4Tensor`` (which ``maybe_cast`` *can* re-stamp, e.g. to fp16
    for MLX) and comes out in the export-canonical form the backend packers
    consume. Contrast with :func:`quantize_stream`, whose output stays
    torchao-native for serialization.

    Non-persistent buffers (KV cache, RoPE tables) are excluded; persistent
    buffers are passed through unchanged (never converted or cast -- they may be
    non-float).
    """
    state: dict[str, torch.Tensor] = {}
    persistent_keys = set(model.state_dict().keys())

    n_params = sum(1 for _ in model.named_parameters())
    params = ((fqn, param.data) for fqn, param in model.named_parameters())
    for i, (fqn, qt) in enumerate(quantize_stream(params, recipe, dtype)):
        state[fqn] = maybe_cast(convert(fqn, qt), dtype)
        if verbose:
            print(f"  Quantized {i + 1}/{n_params}: {fqn}", end="\r")
    if verbose:
        print()

    for fqn, buf in model.named_buffers():
        if fqn in persistent_keys and fqn not in state:
            state[fqn] = buf.data

    return state
