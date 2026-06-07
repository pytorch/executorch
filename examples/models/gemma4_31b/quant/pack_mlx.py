# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX packer: convert quantized weights to MLX-compatible format.

``Int4Tensor`` weights are wrapped as ``ExportableInt4Tensor`` so they export to
``dequantize_int4_tensor -> linear/embedding`` (matched by MLX's Int4 handlers).
``IntxUnpackedToInt8Tensor`` (e.g. int8 / Q6_K) already exports to
``dequantize_affine -> linear`` and is assigned directly, regrouped to an
MLX-compatible group size when needed.

The backend-agnostic ``pack_model`` dispatcher lives in ``pack.py``.
"""

import json

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401

_MLX_SUPPORTED_GROUP_SIZES = (128, 64, 32, 16)


# ---------------------------------------------------------------------------
# Embedding group_size regrouping


def _mlx_group_size(gs: int, K: int) -> int:
    """Find an MLX-compatible group_size for the given weight group_size.

    If ``gs`` is already in {32, 64, 128}, return it.  Otherwise find the
    largest supported group_size that divides ``gs`` so per-axis scales can
    be repeated to fill finer groups.
    """
    if gs in _MLX_SUPPORTED_GROUP_SIZES:
        return gs
    for candidate in _MLX_SUPPORTED_GROUP_SIZES:
        if gs % candidate == 0 and K % candidate == 0:
            return candidate
    raise ValueError(
        f"MLX requires group_size in {set(_MLX_SUPPORTED_GROUP_SIZES)} "
        f"(or a multiple thereof), got {gs}"
    )


def _regroup_intx(w: torch.Tensor, new_gs: int) -> torch.Tensor:
    """Regroup an ``IntxUnpackedToInt8Tensor`` to a finer group_size."""
    from torchao.quantization import IntxUnpackedToInt8Tensor

    old_gs = w.block_size[-1]
    if old_gs % new_gs != 0:
        raise ValueError(
            f"new group_size {new_gs} must evenly divide old group_size {old_gs}"
        )
    repeat_factor = old_gs // new_gs
    N = w.qdata.shape[0]
    n_groups = w.qdata.shape[-1] // new_gs

    scale = w.scale.repeat_interleave(repeat_factor, dim=-1).reshape(N, n_groups)
    zero_point = w.zero_point.repeat_interleave(repeat_factor, dim=-1).reshape(
        N, n_groups
    )

    return IntxUnpackedToInt8Tensor(
        qdata=w.qdata,
        scale=scale,
        zero_point=zero_point,
        target_dtype=w.target_dtype,
        block_size=(1, new_gs),
        dtype=w.dtype,
        activation_quantization=w.activation_quantization,
    )


# ---------------------------------------------------------------------------
# Per-module packer


def pack_for_mlx(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Pack a quantized weight for MLX.

    ``Int4Tensor`` is converted to ``IntxUnpackedToInt8Tensor`` so the
    default dispatch produces the ``dequantize_affine → linear`` pattern
    MLX expects.  Regroups to a compatible group_size when needed (e.g.
    per-axis group_size=5376 → group_size=128) since MLX's
    ``parse_dequant_node`` only accepts group_size in {16, 32, 64, 128}.
    Group sizes ≥ 32 use the fused ``QuantizedMatmulNode``; group_size=16
    (e.g. GGUF Q6_K) falls back to ``DequantizeNode`` + matmul at export.
    """
    from executorch.extension.llm.export.int4 import ExportableInt4Tensor
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        # Int4 group is MLX-native (32); wrap so it exports to
        # dequantize_int4_tensor -> linear/embedding.
        w = ExportableInt4Tensor.from_int4_tensor(w)
    elif isinstance(w, IntxUnpackedToInt8Tensor):
        gs = w.block_size[-1]
        K = w.qdata.shape[-1]
        target_gs = _mlx_group_size(gs, K)
        if target_gs != gs:
            w = _regroup_intx(w, target_gs)
    module.weight = nn.Parameter(w, requires_grad=False)


DEFAULT_MLX_PACKERS: dict[type, ModulePackerFn] = {
    nn.Linear: pack_for_mlx,
    nn.Embedding: pack_for_mlx,
}


# ---------------------------------------------------------------------------
# Load + pack (I/O wrapper)


def load_and_pack_for_mlx(
    path: str,
    model: nn.Module,
    packers: dict[type, ModulePackerFn] | None = None,
) -> None:
    """Load a quantized safetensors file and pack for MLX.

    Streams one weight at a time via torchao's safetensors support.
    """
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    from .pack import pack_one

    _packers = packers or DEFAULT_MLX_PACKERS
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        all_keys = list(f.keys())
        tensor_names = json.loads(metadata.get("tensor_names", "[]"))

        for name in tensor_names:
            parts = name.rsplit(".", 1)
            module_fqn = parts[0] if len(parts) > 1 else ""
            weight_name = parts[-1]
            prefix = (
                f"{module_fqn}._{weight_name}_" if module_fqn else f"_{weight_name}_"
            )
            partial = {}
            for key in all_keys:
                if key.startswith(prefix) or key == name:
                    partial[key] = f.get_tensor(key)
            result, _ = unflatten_tensor_state_dict(partial, metadata)
            for fqn, value in result.items():
                pack_one(model, fqn, value, _packers)

    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in checkpoint "
                f"(model/checkpoint version mismatch?)"
            )
