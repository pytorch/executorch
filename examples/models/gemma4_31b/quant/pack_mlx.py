# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX packer: convert quantized weights to MLX-compatible format.

``Int4Tensor`` weights are wrapped as ``ExportableInt4Tensor`` so they export to
``dequantize_int4_tensor -> linear/embedding`` (matched by MLX's Int4 handlers).
``IntxUnpackedToInt8Tensor`` (e.g. int8 / Q6_K) already exports to
``dequantize_affine -> linear`` and is assigned directly. Coarse/per-axis group
sizes are regrouped to an MLX-legal size (16/32/64/128) inside the MLX pattern
handlers at export time (``regroup_affine_scales``), so no pack-time regroup is
needed here.

The backend-agnostic ``pack_model`` dispatcher lives in ``pack.py``.
"""

import json

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401


# ---------------------------------------------------------------------------
# Per-module packer


def pack_for_mlx(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Pack a quantized weight for MLX.

    ``Int4Tensor`` is wrapped as ``ExportableInt4Tensor`` (exports to
    ``dequantize_int4_tensor → linear/embedding``). ``IntxUnpackedToInt8Tensor``
    is assigned directly; coarse/per-axis group sizes are regrouped to an
    MLX-legal size in the MLX pattern handlers at export time.
    """
    from executorch.extension.llm.export.int4 import ExportableInt4Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        # Int4 group is MLX-native (32); wrap so it exports to
        # dequantize_int4_tensor -> linear/embedding.
        w = ExportableInt4Tensor.from_int4_tensor(w)
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
