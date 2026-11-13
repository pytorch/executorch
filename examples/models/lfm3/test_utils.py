# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections.abc import Sequence
from importlib.resources import files

import torch
from executorch.examples.models.lfm3.convert_weights import lfm_3_to_meta
from executorch.examples.models.llama.llama_transformer import (
    construct_transformer,
    Transformer,
)
from executorch.examples.models.llama.model_args import ModelArgs

LAYER_TYPES: list[str] = [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "sliding_attention",
    "conv",
]
DIM: int = 64
HIDDEN: int = 128
HEADS: int = 4
KV_HEADS: int = 2
HEAD_DIM: int = 16
VOCAB: int = 97
CONV_L_CACHE: int = 7
WINDOW: int = 8
FULL_THETA: float = 1e6
FULL_PARTIAL: float = 0.5
LOCAL_THETA: float = 1e4


def shared_meta_config() -> dict:
    with files("executorch.examples.models.lfm3").joinpath(
        "config/1_7b_config.json"
    ).open() as config_file:
        return json.load(config_file)


def tiny_meta_config() -> dict:
    config = shared_meta_config()
    config.update(
        dim=DIM,
        hidden_dim=HIDDEN,
        n_heads=HEADS,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        n_layers=len(LAYER_TYPES),
        vocab_size=VOCAB,
        sliding_window=WINDOW,
        layer_types=LAYER_TYPES.copy(),
    )
    return config


def tiny_hf_config() -> dict:
    """HF config before the converter's zero-tail to prefix mapping."""
    return {
        "hidden_size": DIM,
        "num_hidden_layers": len(LAYER_TYPES),
        "intermediate_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "norm_eps": 1e-5,
        "conv_L_cache": CONV_L_CACHE,
        "sliding_window": WINDOW,
        "layer_types": LAYER_TYPES.copy(),
        "rope_parameters": {
            "full_attention": {
                "rope_theta": FULL_THETA,
                "partial_rotary_factor": FULL_PARTIAL,
            },
            "sliding_attention": {"rope_theta": LOCAL_THETA},
        },
    }


def random_hf_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights with LFM3's HF parameter names and shapes."""
    generator = torch.Generator().manual_seed(seed)

    def weight(*shape: int, scale: float = 0.1) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * scale

    def norm(dim: int) -> torch.Tensor:
        return 1.0 + weight(dim)

    state_dict = {
        "model.embed_tokens.weight": weight(VOCAB, DIM, scale=1.0),
        "model.embedding_norm.weight": norm(DIM),
    }
    for layer, layer_type in enumerate(LAYER_TYPES):
        prefix = f"model.layers.{layer}."
        for name in (
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ):
            state_dict[prefix + name + ".weight"] = norm(DIM)
        state_dict[prefix + "feed_forward.w1.weight"] = weight(HIDDEN, DIM)
        state_dict[prefix + "feed_forward.w3.weight"] = weight(HIDDEN, DIM)
        state_dict[prefix + "feed_forward.w2.weight"] = weight(DIM, HIDDEN)
        if layer_type == "conv":
            state_dict[prefix + "conv.in_proj.weight"] = weight(3 * DIM, DIM)
            state_dict[prefix + "conv.out_proj.weight"] = weight(DIM, DIM)
            state_dict[prefix + "conv.conv.weight"] = weight(
                DIM, 1, CONV_L_CACHE, scale=0.3
            )
        else:
            prefix += "self_attn."
            state_dict[prefix + "q_proj.weight"] = weight(HEADS * HEAD_DIM, DIM)
            state_dict[prefix + "k_proj.weight"] = weight(KV_HEADS * HEAD_DIM, DIM)
            state_dict[prefix + "v_proj.weight"] = weight(KV_HEADS * HEAD_DIM, DIM)
            state_dict[prefix + "o_proj.weight"] = weight(DIM, HEADS * HEAD_DIM)
            state_dict[prefix + "gate_proj.weight"] = weight(HEADS, DIM)
            state_dict[prefix + "q_norm.weight"] = norm(HEAD_DIM)
            state_dict[prefix + "k_norm.weight"] = norm(HEAD_DIM)
    return state_dict


def generic_model(max_context_len: int = 64, max_seq_len: int = WINDOW) -> Transformer:
    """Generic Transformer using the same Meta config and converter as QNN."""
    config = tiny_meta_config()
    config.update(
        max_context_len=max_context_len,
        max_seq_len=max_seq_len,
        use_kv_cache=True,
        enable_dynamic_shape=True,
        generate_full_logits=True,
    )
    model = construct_transformer(ModelArgs(**config)).eval()
    state_dict = lfm_3_to_meta(random_hf_state_dict(), tiny_hf_config())
    assert set(dict(model.named_parameters())) == set(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= set(dict(model.named_buffers())), missing
    return model


def run(model: Transformer, tokens: Sequence[int], chunk: int) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for pos in range(0, len(tokens), chunk):
            piece = tokens[pos : pos + chunk]
            logits = model(torch.tensor([piece]), {"input_pos": torch.tensor([pos])})
            logits = logits[0] if isinstance(logits, tuple) else logits
            outputs.append(logits[0])
    return torch.cat(outputs)
