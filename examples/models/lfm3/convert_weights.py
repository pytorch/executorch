# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch

from executorch.examples.models.checkpoint import get_mapped_key
from safetensors.torch import load_file

_LFM_3_TO_META = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.embedding_norm.weight": "norm.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.gate_proj.weight": "layers.{}.attention.og.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.layers.{}.operator_norm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_operator_norm.weight": "layers.{}.post_attention_norm.weight",
}

_CONFIG = os.path.join(os.path.dirname(__file__), "config", "1_7b_config.json")
# Where the text decoder lives in the multimodal LFM3 checkpoints.
_DECODER_PREFIXES = ("model.language_model.", "lfm.model.")


def _zero_tail_to_prefix(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Head-dim permutation mapping LFM3's zero-tail partial RoPE onto the
    prefix layout of `hf_apply_rotary_emb`.

    LFM3 rotates dim pairs (i, i + head_dim/2) for i < rotary_dim/2 and leaves the
    rest. The prefix layout rotates (j, j + rotary_dim/2) within the first
    rotary_dim dims. Ordering [0:r/2, h/2:h/2+r/2, r/2:h/2, h/2+r/2:h] lines the two
    up; q and k get the same permutation, so attention scores are unchanged.
    LFM3's frequencies use head_dim as the exponent denominator, so in the prefix
    layout the full-attention rope_theta becomes theta ** partial_rotary_factor
    (see config/1_7b_config.json).
    """
    half, rhalf = head_dim // 2, rotary_dim // 2
    return torch.cat(
        [
            torch.arange(0, rhalf),
            torch.arange(half, half + rhalf),
            torch.arange(rhalf, half),
            torch.arange(half + rhalf, head_dim),
        ]
    )


def lfm_3_to_meta(  # noqa: C901
    state_dict: Dict[str, torch.Tensor], cfg: Optional[Dict] = None
) -> Dict[str, torch.Tensor]:
    if cfg is None:
        with open(_CONFIG) as f:
            cfg = json.load(f)
    head_dim = cfg["head_dim"]
    full_rope = cfg["rope_parameters"]["full_attention"]
    perm = _zero_tail_to_prefix(
        head_dim, int(head_dim * full_rope.get("partial_rotary_factor", 1.0))
    )

    multimodal = any(k.startswith(_DECODER_PREFIXES) for k in state_dict)
    converted = {}
    for key, value in state_dict.items():
        prefix = next((p for p in _DECODER_PREFIXES if key.startswith(p)), None)
        if prefix is not None:
            key = "model." + key[len(prefix) :]
        elif multimodal:
            continue  # vision/audio towers and heads are not part of the decoder
        try:
            new_key = get_mapped_key(key, _LFM_3_TO_META)
        except Exception:
            # Remaining LFM2-style names (feed_forward.*, conv.*) map 1:1.
            if not key.startswith(("model.layers.",)) or not any(
                part in key
                for part in (
                    ".feed_forward.",
                    ".conv.",
                    ".ffn_norm.",
                    ".post_ffn_norm.",
                )
            ):
                raise KeyError(f"Unexpected LFM3 weight: {key}")
            new_key = key.removeprefix("model.")

        layer = int(new_key.split(".")[1]) if new_key.startswith("layers.") else None
        if layer is not None and cfg["layer_types"][layer] == "full_attention":
            if new_key.endswith(("attention.wq.weight", "attention.wk.weight")):
                value = value.view(-1, head_dim, value.shape[-1])[:, perm].reshape(
                    value.shape
                )
            elif new_key.endswith(("q_norm_fn.weight", "k_norm_fn.weight")):
                value = value[perm]

        if new_key.endswith(".conv.in_proj.weight"):
            for name, split_value in zip(
                ["B_proj", "C_proj", "x_proj"], torch.chunk(value, 3, dim=0)
            ):
                converted[new_key.replace("in_proj", name)] = split_value
        else:
            converted[new_key] = value

    converted["output.weight"] = converted["tok_embeddings.weight"]
    return converted


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint index key: {key}")
        result[key] = value
    return result


def _checkpoint_shard_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors", filename
    ):
        raise ValueError(f"Invalid checkpoint shard filename: {filename!r}")
    path = root / filename
    resolved = path.resolve()
    if resolved.parent == root:
        return path
    # HF snapshots link their shards to content-addressed blobs in the same repo.
    blob_root = root.parent.parent / "blobs"
    if (
        root.parent.name == "snapshots"
        and root.parent.parent.name.startswith("models--")
        and re.fullmatch(r"[0-9a-f]{40}", root.name)
        and resolved.parent == blob_root
        and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name)
    ):
        return path
    raise ValueError(f"Checkpoint shard escapes input directory: {filename!r}")


def load_checkpoint(input_dir: str) -> Dict:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if not os.path.lexists(index_path):
        return load_file(os.path.join(input_dir, "model.safetensors"))

    with open(index_path) as f:
        index = json.load(f, object_pairs_hook=_unique_json_object)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index must contain a non-empty weight_map")

    shards = {}
    root = Path(input_dir).resolve()
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Checkpoint tensor names must be non-empty strings")
        _checkpoint_shard_path(root, filename)
        shards.setdefault(filename, set()).add(key)

    state_dict = {}
    for filename, expected_keys in sorted(shards.items()):
        shard = load_file(str(root / filename))
        if shard.keys() != expected_keys:
            missing = sorted(expected_keys - shard.keys())
            unexpected = sorted(shard.keys() - expected_keys)
            raise ValueError(
                f"Checkpoint shard {filename!r} disagrees with weight_map: "
                f"missing={missing}, unexpected={unexpected}"
            )
        state_dict.update(shard)
    return state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    sd = lfm_3_to_meta(load_checkpoint(input_dir))
    torch.save(sd, output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert LFM3 weights to Meta format.")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
