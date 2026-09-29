# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a Meta-format Llama checkpoint (``consolidated.00.pth`` and
``params.json``) to a HuggingFace model directory for ``export_hf_llm``."""

import argparse
import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def _config(params: dict) -> LlamaConfig:
    dim = params["dim"]
    hidden = int(2 * (4 * dim) / 3)
    hidden = int(params.get("ffn_dim_multiplier", 1.0) * hidden)
    multiple = params["multiple_of"]
    hidden = multiple * ((hidden + multiple - 1) // multiple)
    rope_scaling = None
    if params.get("use_scaled_rope", False):
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    return LlamaConfig(
        vocab_size=params["vocab_size"],
        hidden_size=dim,
        intermediate_size=hidden,
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params["n_kv_heads"],
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        rope_scaling=rope_scaling,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
    )


def _permute(w: torch.Tensor, heads: int) -> torch.Tensor:
    # Meta checkpoints pair channels (2i, 2i+1); HF rotate_half pairs (i, i+D/2).
    rows, cols = w.shape
    return (
        w.view(heads, rows // heads // 2, 2, cols).transpose(1, 2).reshape(rows, cols)
    )


def _meta_to_hf(sd: dict, config: LlamaConfig) -> dict:
    out = {
        "model.embed_tokens.weight": sd["tok_embeddings.weight"],
        "model.norm.weight": sd["norm.weight"],
        "lm_head.weight": sd.get("output.weight", sd["tok_embeddings.weight"]).clone(),
    }
    for i in range(config.num_hidden_layers):
        p, q = f"layers.{i}.", f"model.layers.{i}."
        out[q + "self_attn.q_proj.weight"] = _permute(
            sd[p + "attention.wq.weight"], config.num_attention_heads
        )
        out[q + "self_attn.k_proj.weight"] = _permute(
            sd[p + "attention.wk.weight"], config.num_key_value_heads
        )
        out[q + "self_attn.v_proj.weight"] = sd[p + "attention.wv.weight"]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attention.wo.weight"]
        out[q + "mlp.gate_proj.weight"] = sd[p + "feed_forward.w1.weight"]
        out[q + "mlp.down_proj.weight"] = sd[p + "feed_forward.w2.weight"]
        out[q + "mlp.up_proj.weight"] = sd[p + "feed_forward.w3.weight"]
        out[q + "input_layernorm.weight"] = sd[p + "attention_norm.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="HuggingFace model directory.")
    args = parser.parse_args()

    with open(args.params) as f:
        config = _config(json.load(f))
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model.load_state_dict(_meta_to_hf(sd, config), strict=True, assign=True)
    model.save_pretrained(args.output)
    print(f"saved {args.output}")
