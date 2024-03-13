# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, Mapping


def convert_model_state_dict(
    state_dict: Dict[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a model state dictionary to fairseq2.

    :param state_dict:
        The original model state dictionary.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted model state dictionary that is compatible with fairseq2.
    """
    new_state_dict = {}

    def get_new_key(old_key: str) -> str:
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                return new_key

        return old_key

    # Convert module keys from fairseq to fairseq2.
    for old_key in state_dict.keys():
        new_key = get_new_key(old_key)

        new_state_dict[new_key] = state_dict[old_key]

    return new_state_dict


def convert_to_llama_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a fairseq2 LLaMA checkpoint to the reference format."""
    # state_dict = checkpoint["model"]

    key_map = {
        # fmt: off
        r"decoder.layers.([0-9]+).self_attn.q_proj.": r"layers.\1.attention.wq.",
        r"decoder.layers.([0-9]+).self_attn.k_proj.": r"layers.\1.attention.wk.",
        r"decoder.layers.([0-9]+).self_attn.v_proj.": r"layers.\1.attention.wv.",
        r"decoder.layers.([0-9]+).self_attn.output_proj.": r"layers.\1.attention.wo.",
        r"decoder.layers.([0-9]+).self_attn_layer_norm.": r"layers.\1.attention_norm.",
        r"decoder.layers.([0-9]+).ffn.gate_proj.": r"layers.\1.feed_forward.w1.",
        r"decoder.layers.([0-9]+).ffn.output_proj.": r"layers.\1.feed_forward.w2.",
        r"decoder.layers.([0-9]+).ffn.inner_proj.": r"layers.\1.feed_forward.w3.",
        r"decoder.layers.([0-9]+).ffn_layer_norm.": r"layers.\1.ffn_norm.",
        r"decoder.layer_norm.": r"norm.",
        r"decoder_frontend.embed.": r"tok_embeddings.",
        r"final_proj.": r"output.",
        # fmt: on
    }

    return convert_model_state_dict(checkpoint, key_map)
