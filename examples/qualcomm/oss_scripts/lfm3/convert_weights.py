# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import re

import torch
from safetensors.torch import load_file

_KEY_MAP = {
    r"model\.embed_tokens\.weight": "tok_embeddings.weight",
    r"model\.embedding_norm\.weight": "norm.weight",
    r"model\.layers\.(\d+)\.operator_norm\.weight": r"layers.\1.attention_norm.weight",
    r"model\.layers\.(\d+)\.post_operator_norm\.weight": r"layers.\1.post_attention_norm.weight",
    r"model\.layers\.(\d+)\.ffn_norm\.weight": r"layers.\1.ffn_norm.weight",
    r"model\.layers\.(\d+)\.post_ffn_norm\.weight": r"layers.\1.post_ffn_norm.weight",
    r"model\.layers\.(\d+)\.feed_forward\.(w[123])\.weight": r"layers.\1.feed_forward.\2.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": r"layers.\1.attention.wq.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": r"layers.\1.attention.wk.weight",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": r"layers.\1.attention.wv.weight",
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": r"layers.\1.attention.wo.weight",
    r"model\.layers\.(\d+)\.self_attn\.gate_proj\.weight": r"layers.\1.attention.gate.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight": r"layers.\1.attention.q_norm_fn.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight": r"layers.\1.attention.k_norm_fn.weight",
    r"model\.layers\.(\d+)\.conv\.in_proj\.weight": r"layers.\1.attention.in_proj.weight",
    r"model\.layers\.(\d+)\.conv\.out_proj\.weight": r"layers.\1.attention.out_proj.weight",
    r"model\.layers\.(\d+)\.conv\.conv\.weight": r"layers.\1.attention.conv_weight",
}


# Where the text decoder lives in the multimodal LFM3 checkpoints.
_DECODER_PREFIXES = ("model.language_model.", "lfm.model.")


def lfm3_hf_to_static(state_dict):
    converted = {}
    multimodal = any(k.startswith(_DECODER_PREFIXES) for k in state_dict)
    for key, value in state_dict.items():
        for prefix in _DECODER_PREFIXES:
            if key.startswith(prefix):
                key = "model." + key[len(prefix) :]
                break
        else:
            if multimodal:
                # Vision/audio towers, projectors and audio heads are not exported.
                continue
        for pattern, target in _KEY_MAP.items():
            if re.fullmatch(pattern, key):
                new_key = re.sub(pattern, target, key)
                break
        else:
            raise KeyError(f"Unexpected LFM3 weight: {key}")
        if new_key.endswith("conv_weight"):
            # [dim, 1, kernel] -> [kernel, dim]
            value = value.squeeze(1).transpose(0, 1).contiguous()
        converted[new_key] = value.float()
    # Tied embeddings.
    converted["output.weight"] = converted["tok_embeddings.weight"]
    return converted


def convert_weights(input_dir: str, output_file: str) -> None:
    sd = load_file(f"{input_dir}/model.safetensors")
    torch.save(lfm3_hf_to_static(sd), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LFM3 HF weights.")
    parser.add_argument("input_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)
