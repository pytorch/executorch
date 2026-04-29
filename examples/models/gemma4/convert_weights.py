"""HuggingFace Gemma4 checkpoint -> Gemma4TextModel state dict.

NO HuggingFace dependency: reads safetensors via the `safetensors` package
(a thin wrapper around the file format, not part of `transformers`).
"""

import json
import os
import re
from typing import Dict

import torch


# HF key (with `model.` stripped) -> Gemma4TextModel parameter key.
_HF_TO_ET = {
    "embed_tokens.weight": "embed_tokens.weight",
    "norm.weight": "norm.weight",
    "layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
    "layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
    "layers.{}.pre_feedforward_layernorm.weight": "layers.{}.pre_feedforward_layernorm.weight",
    "layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_feedforward_layernorm.weight",
    "layers.{}.self_attn.q_proj.weight": "layers.{}.self_attn.q_proj.weight",
    "layers.{}.self_attn.k_proj.weight": "layers.{}.self_attn.k_proj.weight",
    "layers.{}.self_attn.v_proj.weight": "layers.{}.self_attn.v_proj.weight",
    "layers.{}.self_attn.o_proj.weight": "layers.{}.self_attn.o_proj.weight",
    "layers.{}.self_attn.q_norm.weight": "layers.{}.self_attn.q_norm.weight",
    "layers.{}.self_attn.k_norm.weight": "layers.{}.self_attn.k_norm.weight",
    "layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.gate_proj.weight",
    "layers.{}.mlp.up_proj.weight": "layers.{}.mlp.up_proj.weight",
    "layers.{}.mlp.down_proj.weight": "layers.{}.mlp.down_proj.weight",
    # Learned per-layer residual scalar (Gemma4 specific). Default-initializing
    # this to 1.0 silently breaks the model — outputs collapse to a single token.
    "layers.{}.layer_scalar": "layers.{}.layer_scalar",
    # lm_head: HF tied embeddings checkpoints typically omit this; the model
    # ties it from embed_tokens. We map it through if present.
    "lm_head.weight": "lm_head.weight",
}


def _build_mapping_regexes():
    out = []
    for hf_pat, et_pat in _HF_TO_ET.items():
        if "{}" not in hf_pat:
            out.append((re.compile("^" + re.escape(hf_pat) + "$"), et_pat))
        else:
            regex = "^" + re.escape(hf_pat).replace(r"\{\}", r"(\d+)") + "$"
            out.append((re.compile(regex), et_pat))
    return out


_MAPPING = _build_mapping_regexes()


def _map_key(hf_key: str) -> str | None:
    """Map a (model.-stripped) HF key to a Gemma4TextModel key, or None to skip."""
    for pat, et_pat in _MAPPING:
        m = pat.match(hf_key)
        if m:
            if "{}" in et_pat:
                out = et_pat
                for g in m.groups():
                    out = out.replace("{}", g, 1)
                return out
            return et_pat
    return None


def gemma4_hf_to_executorch(
    state_dict: Dict[str, torch.Tensor],
    config,
) -> Dict[str, torch.Tensor]:
    """Convert HF state dict to Gemma4TextModel state dict in-memory.

    Key rules:
      - Strip leading "model." prefix.
      - Drop multimodal keys (vision tower, audio tower, multimodal projector).
      - For full_attention layers, the HF checkpoint has NO v_proj weight
        (attention_k_eq_v=True); skip silently.
    """
    out: Dict[str, torch.Tensor] = {}
    for raw_key, tensor in state_dict.items():
        key = raw_key
        # Multimodal Gemma4 checkpoints nest text weights under language_model.
        if key.startswith("language_model."):
            key = key[len("language_model.") :]
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
        # Skip vision / audio / multimodal projector weights.
        if key.startswith(("vision_tower.", "audio_tower.", "multi_modal_projector.")):
            continue
        if key.startswith(("model.vision_tower.", "model.audio_tower.", "model.multi_modal_projector.")):
            continue
        # Strip the `model.` prefix (text path).
        if key.startswith("model."):
            key = key[len("model.") :]

        mapped = _map_key(key)
        if mapped is None:
            # Unknown key — likely a buffer (rotary inv_freq) we recompute.
            continue
        out[mapped] = tensor
    return out


def load_checkpoint(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load a HuggingFace safetensors checkpoint (sharded or single)."""
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state_dict: Dict[str, torch.Tensor] = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        for shard in shards:
            state_dict.update(load_file(os.path.join(model_dir, shard)))
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"No safetensors found in {model_dir}")
        state_dict.update(load_file(single))
    return state_dict


def load_and_remap_checkpoint(model_dir: str, config) -> Dict[str, torch.Tensor]:
    raw = load_checkpoint(model_dir)
    return gemma4_hf_to_executorch(raw, config)
