# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load a GGUF or MLX-affine DFlash draft checkpoint.

Token embeddings and the output head are shared with the target model at
export time.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# GGUF pattern → model FQN pattern. ``{}`` is the layer index.
_KEY_MAP = {
    "fc.weight": "fc.weight",
    "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook.weight",
    "selector_successor.weight": "candidate_selector.successor_codebook.weight",
    "blk.{}.attn_conv_base": "layers.{}.attention_conv.base_kernel",
    "blk.{}.attn_conv_proj.weight": "layers.{}.attention_conv.kernel_projection.weight",
    "blk.{}.ffn_conv_base": "layers.{}.mlp_conv.base_kernel",
    "blk.{}.ffn_conv_proj.weight": "layers.{}.mlp_conv.kernel_projection.weight",
    "enc.output_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
    "blk.{}.attn_q.weight": "layers.{}.self_attn.q_proj.weight",
    "blk.{}.attn_k.weight": "layers.{}.self_attn.k_proj.weight",
    "blk.{}.attn_v.weight": "layers.{}.self_attn.v_proj.weight",
    "blk.{}.attn_output.weight": "layers.{}.self_attn.o_proj.weight",
    "blk.{}.attn_norm.weight": "layers.{}.input_layernorm.weight",
    "blk.{}.ffn_norm.weight": "layers.{}.post_attention_layernorm.weight",
    "blk.{}.attn_q_norm.weight": "layers.{}.self_attn.q_norm.weight",
    "blk.{}.attn_k_norm.weight": "layers.{}.self_attn.k_norm.weight",
    "blk.{}.ffn_gate.weight": "layers.{}.mlp.gate_proj.weight",
    "blk.{}.ffn_up.weight": "layers.{}.mlp.up_proj.weight",
    "blk.{}.ffn_down.weight": "layers.{}.mlp.down_proj.weight",
}

_IGNORED_KEYS = {
    "rope_freqs.weight",
}


def dflash_gguf_to_model_key(gguf_key: str) -> Optional[str]:
    """Map a DFlash GGUF tensor name to a model FQN, or ``None`` to skip."""
    if gguf_key in _IGNORED_KEYS:
        return None

    for gguf_pat, model_pat in _KEY_MAP.items():
        if "{}" not in gguf_pat:
            if gguf_key == gguf_pat:
                return model_pat
            continue
        prefix, suffix = gguf_pat.split("{}")
        if gguf_key.startswith(prefix) and gguf_key.endswith(suffix):
            layer_str = gguf_key[len(prefix) : len(gguf_key) - len(suffix)]
            if layer_str.isdigit():
                return model_pat.replace("{}", layer_str)

    return None


def _validate_no_meta(model: nn.Module) -> None:
    """Ensure all parameters have been loaded."""
    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in DFlash GGUF file "
                f"(model/checkpoint version mismatch?)"
            )
    for p in model.parameters():
        p.requires_grad_(False)


def load_dflash_gguf(
    gguf_path: str,
    backend: str = "mlx",
    max_seq_len: int = 131072,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load a DFlash draft GGUF, remap keys, and return (model, config).

    Streams weights through ``load_checkpoint`` with a DFlash key remap. The
    caller applies backend-specific packing after loading.

    Returns ``(model, config)`` where config is a ``DFlashConfig``.
    """
    from executorch.examples.models.muse_glimmer.model.dflash_model import (
        DFlashConfig,
        DFlashDraftModel,
    )
    from executorch.extension.llm.export.load import load_checkpoint
    from executorch.extension.llm.export.quant import to_default

    # Read metadata from GGUF to build config
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    metadata = {
        key: field.contents()
        for key, field in reader.fields.items()
        if key.startswith("dflash.") or key == "tokenizer.ggml.mask_token_id"
    }
    config = DFlashConfig.from_gguf_metadata(metadata)
    config.max_seq_len = max_seq_len
    if config.selector_top_k:
        unknown = [
            tensor.name
            for tensor in reader.tensors
            if tensor.name not in _IGNORED_KEYS
            and dflash_gguf_to_model_key(tensor.name) is None
        ]
        if unknown:
            raise ValueError(f"Unrecognized DFlash2 GGUF tensors: {unknown}")
    del reader

    print(
        f"DFlash config: {config.n_layers} layers, dim={config.dim}, "
        f"block_size={config.block_size}, target_layers={config.target_layers}"
    )

    print("Building draft model on meta device...")
    with torch.device("meta"):
        model = DFlashDraftModel(config, max_context_length=max_seq_len)

    expected_shapes = {name: value.shape for name, value in model.named_parameters()}

    def convert_weight(name, value):
        if config.selector_top_k and value.shape != expected_shapes[name]:
            raise ValueError(
                f"DFlash2 tensor {name} has shape {value.shape}, expected {expected_shapes[name]}"
            )
        return to_default(name, value)

    print(f"Loading DFlash draft from {gguf_path}...")
    load_checkpoint(
        gguf_path,
        model,
        key_map=dflash_gguf_to_model_key,
        convert=convert_weight,
        dtype=activation_dtype,
    )
    _validate_no_meta(model)
    model.eval()

    print(f"Loaded DFlash draft: {config.n_layers} layers, dim={config.dim}")
    return model, config


def load_dflash_mlx(
    mlx_dir: str,
    backend: str = "mlx",
    max_seq_len: int = 131072,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load an MLX affine-quantized DFlash draft (e.g. dflash-3l_wind2048-mlx-q5).

    The MLX draft's tensor names already match the ``DFlashDraftModel`` FQNs
    (``fc``, ``hidden_norm``, ``norm``, per-layer ``self_attn`` / ``mlp`` /
    layernorms / ``q_norm`` / ``k_norm``), so no key remap is needed:
    ``iter_checkpoint`` auto-detects the MLX layout, ``_iter_mlx`` rebuilds each
    ``(weight, scales, biases)`` triplet as a torchao intx tensor, and the draft
    is already quantized. ``DFlashConfig`` is built
    from the checkpoint's HF-style ``config.json``. Returns ``(model, config)``.
    """
    import json
    import os

    from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
        _iter_mlx,
    )
    from executorch.examples.models.muse_glimmer.model.dflash_model import (
        DFlashConfig,
        DFlashDraftModel,
    )
    from executorch.extension.llm.export.load import load_checkpoint
    from executorch.extension.llm.export.quant import to_default

    with open(os.path.join(mlx_dir, "config.json"), "r") as f:
        c = json.load(f)
    layer_types = c.get("layer_types", [])
    config = DFlashConfig(
        dim=c["hidden_size"],
        n_layers=c["num_hidden_layers"],
        n_heads=c["num_attention_heads"],
        n_kv_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        ffn_dim=c["intermediate_size"],
        vocab_size=c["vocab_size"],
        block_size=c["block_size"],
        target_layers=c["target_layer_ids"],
        rope_theta=c["rope_theta"],
        norm_eps=c["rms_norm_eps"],
        mask_token_id=c["mask_token_id"],
        max_seq_len=max_seq_len,
        sliding_window=c.get("sliding_window"),
        sliding_window_pattern=(
            [lt == "sliding_attention" for lt in layer_types] if layer_types else None
        ),
    )

    print("Building DFlash draft model on meta device...")
    with torch.device("meta"):
        model = DFlashDraftModel(config)

    print(f"Loading MLX DFlash draft from {mlx_dir}...")
    load_checkpoint(
        mlx_dir, model, raw_iter=_iter_mlx, convert=to_default, dtype=activation_dtype
    )
    _validate_no_meta(model)
    model.eval()

    print(f"Loaded DFlash draft: {config.n_layers} layers, dim={config.dim}")
    return model, config
