# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load a GGUF file into a Gemma 4 31B model.

Streams tensors one at a time via ``iter_gguf_tensors`` for low peak
memory, remaps GGUF names to model FQNs, handles tied embed/lm_head,
and packs for the target backend.

Community GGUFs typically pack the text decoder only (no vision tower
weights). For multimodal inference, pass a directory of HF bf16
safetensors via ``vision_safetensors_dir``; the loader will populate
``vision_tower.*`` / ``embed_vision.*`` from that checkpoint after the
GGUF stream completes.

Usage:
    # Text-only:
    model, config = load_gguf_model("model.gguf", backend="cuda")

    # Multimodal (text from GGUF, vision from HF bf16):
    model, config = load_gguf_model(
        "model.gguf", backend="cuda",
        vision_safetensors_dir="/path/to/gemma-4-31B",
    )
"""

import os
from typing import Optional

import torch

# GGUF pattern → model FQN pattern. ``{}`` is the layer index.
_KEY_MAP = {
    "token_embd.weight": "embed_tokens.weight",
    "output_norm.weight": "norm.weight",
    # Per-layer attention
    "blk.{}.attn_q.weight": "layers.{}.self_attn.q_proj.weight",
    "blk.{}.attn_k.weight": "layers.{}.self_attn.k_proj.weight",
    "blk.{}.attn_v.weight": "layers.{}.self_attn.v_proj.weight",
    "blk.{}.attn_output.weight": "layers.{}.self_attn.o_proj.weight",
    "blk.{}.attn_q_norm.weight": "layers.{}.self_attn.q_norm.weight",
    "blk.{}.attn_k_norm.weight": "layers.{}.self_attn.k_norm.weight",
    # Per-layer norms
    "blk.{}.attn_norm.weight": "layers.{}.input_layernorm.weight",
    "blk.{}.post_attention_norm.weight": "layers.{}.post_attention_layernorm.weight",
    "blk.{}.ffn_norm.weight": "layers.{}.pre_feedforward_layernorm.weight",
    "blk.{}.post_ffw_norm.weight": "layers.{}.post_feedforward_layernorm.weight",
    # Per-layer MLP
    "blk.{}.ffn_gate.weight": "layers.{}.mlp.gate_proj.weight",
    "blk.{}.ffn_up.weight": "layers.{}.mlp.up_proj.weight",
    "blk.{}.ffn_down.weight": "layers.{}.mlp.down_proj.weight",
    # Per-layer scalar
    "blk.{}.layer_output_scale.weight": "layers.{}.layer_scalar",
}

_IGNORED_KEYS = {"rope_freqs.weight"}


def gguf_to_model_key(gguf_key: str) -> Optional[str]:
    """Map a GGUF tensor name to a model FQN, or ``None`` to skip."""
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


def _resolve_tied_lm_head(model, embed_quant, packers):
    """Handle tied embed/lm_head after streaming all tensors."""
    from executorch.examples.models.gemma4_31b.quant import pack_one

    lm_head = getattr(model.lm_head, "weight", None)
    if lm_head is None or lm_head.device.type != "meta":
        return
    if embed_quant is not None:
        pack_one(model, "lm_head.weight", embed_quant, packers)
    else:
        pack_one(
            model,
            "lm_head.weight",
            model.embed_tokens.weight.data.clone(),
            packers,
        )


def _validate_no_meta(model):
    """Ensure all parameters have been loaded."""
    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in GGUF file "
                f"(model/checkpoint version mismatch?)"
            )
    for p in model.parameters():
        p.requires_grad_(False)


def _load_vision_from_hf(
    model,
    config,
    vision_safetensors_dir: str,
) -> None:
    """Populate ``vision_tower.*`` / ``embed_vision.*`` from an HF bf16 checkpoint.

    Streams the safetensors shards on CPU, filters to vision keys via the
    shared HF→model remap, then assigns the result with
    ``load_state_dict(..., assign=True)`` so that meta-device vision params
    become real bf16 tensors. Non-vision keys in the shard are ignored.
    """
    from executorch.examples.models.gemma4_31b.model import (
        _hf_to_model_key,
        _VISION_PREFIXES,
    )
    from safetensors import safe_open

    index_path = os.path.join(
        vision_safetensors_dir, "model.safetensors.index.json"
    )
    if os.path.exists(index_path):
        import json as _json

        with open(index_path, "r") as f:
            index = _json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    elif os.path.exists(os.path.join(vision_safetensors_dir, "model.safetensors")):
        shard_files = ["model.safetensors"]
    else:
        raise FileNotFoundError(
            f"No safetensors checkpoint in {vision_safetensors_dir}"
        )

    vision_state: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        shard_path = os.path.join(vision_safetensors_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for ckpt_key in f.keys():
                norm = ckpt_key
                if norm.startswith("model.language_model."):
                    continue  # text decoder lives in GGUF
                if not norm.startswith(_VISION_PREFIXES):
                    continue
                model_key = _hf_to_model_key(ckpt_key)
                if model_key is None:
                    continue
                vision_state[model_key] = f.get_tensor(ckpt_key)

    if not vision_state:
        raise ValueError(
            f"No vision_tower/embed_vision keys found in {vision_safetensors_dir}."
        )

    missing, unexpected = model.load_state_dict(
        vision_state, strict=False, assign=True
    )
    print(
        f"  Vision: loaded {len(vision_state)} tensors from {vision_safetensors_dir}"
    )
    if unexpected:
        print(f"  WARNING: unexpected vision keys: {sorted(unexpected)[:5]}")


def load_gguf_model(
    gguf_path: str,
    max_seq_len: int = 4096,
    backend: str = "cuda",
    vision_safetensors_dir: Optional[str] = None,
) -> tuple:
    """Load a GGUF file, remap keys, and pack for the target backend.

    Streams tensors one at a time for low peak memory.

    GGUF ties ``embed_tokens`` and ``lm_head`` into a single Q4_K tensor.
    We untie them: the embedding is dequantized to bf16 (``nn.Embedding``
    needs gather, which ``Int4TilePackedTo4dTensor`` does not support),
    while ``lm_head`` keeps the original Q4_K quantization (``nn.Linear``
    matmul via tinygemm).

    If ``vision_safetensors_dir`` is provided, vision_tower / embed_vision
    weights are loaded from that HF bf16 checkpoint after the GGUF stream
    completes. Without it, vision params stay on meta and the model is
    text-only.

    Returns ``(model, config)``.
    """
    from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig
    from executorch.examples.models.gemma4_31b.quant import dequantize_weight, pack_one
    from executorch.examples.models.gemma4_31b.quant.gguf import iter_gguf_tensors
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.quant import DEFAULT_CUDA_PACKERS

        packers = DEFAULT_CUDA_PACKERS
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'cuda'.")

    config = Gemma4_31BConfig(max_seq_len=max_seq_len)
    if vision_safetensors_dir is not None:
        # Bring in vision_config (and any vision-relevant text overrides) from
        # the HF config.json so that vision_tower / embed_vision are
        # instantiated with matching dims.
        hf_config = Gemma4_31BConfig.from_hf_config(
            os.path.join(vision_safetensors_dir, "config.json")
        )
        hf_config.max_seq_len = max_seq_len
        config = hf_config

    print("Building model on meta device...")
    with torch.device("meta"):
        model = Gemma4_31B(config)

    embed_quant = None
    n_processed = 0

    print(f"Streaming GGUF from {gguf_path}...")
    for gguf_name, result in iter_gguf_tensors(gguf_path):
        model_key = gguf_to_model_key(gguf_name)
        if model_key is None:
            continue

        if type(result) is torch.Tensor and result.dtype == torch.float32:
            result = result.to(torch.bfloat16)

        if model_key == "embed_tokens.weight" and isinstance(result, Int4Tensor):
            embed_quant = result
            result = dequantize_weight(result, torch.bfloat16)

        pack_one(model, model_key, result, packers)

        n_processed += 1
        if n_processed % 100 == 0:
            print(f"  Processed {n_processed} tensors...")

    _resolve_tied_lm_head(model, embed_quant, packers)
    del embed_quant

    if vision_safetensors_dir is not None:
        print(f"Loading vision_tower + embed_vision from {vision_safetensors_dir}...")
        _load_vision_from_hf(model, config, vision_safetensors_dir)
        _validate_no_meta(model)
    else:
        # Text-only: skip meta-device validation so the (still-meta) vision
        # submodules don't trip the check. The caller is responsible for not
        # invoking vision_tower / embed_vision in this mode.
        for p in model.parameters():
            p.requires_grad_(False)

    model.eval()

    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
    return model, config
