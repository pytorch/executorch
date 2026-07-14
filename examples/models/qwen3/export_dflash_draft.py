# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exports the DFlash draft model to a .pte program. 

This script loads the pretrained DFlash draft checkpoint, copies the shared embedding and output projection weights from target model, applies same 4-bit quantization used by target, and exports the draft model for MLX inference. 
The exported model is used alongside the target model during speculative decoding. 
"""

import argparse
from pathlib import Path

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import (
    DFlashDraftModel,
    load_dflash_config,
)
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.export import Dim
from transformers import AutoModelForCausalLM


def load_draft_model(draft_id: str, target_state_dict: dict) -> DFlashDraftModel:
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    config = load_dflash_config(path)
    model = DFlashDraftModel(config)

    draft_weights = {}
    for f in path.glob("*.safetensors"):
        draft_weights.update(load_file(str(f)))

    missing, unexpected = model.load_state_dict(draft_weights, strict=False)
    assert not unexpected, f"Unexpected draft checkpoint keys: {unexpected}"
    still_missing = [
        k for k in missing if not k.startswith(("embed_tokens.", "lm_head."))
    ]
    assert not still_missing, f"Missing draft checkpoint keys: {still_missing}"

    model.embed_tokens.weight.data.copy_(target_state_dict["model.embed_tokens.weight"])
    lm_head_key = (
        "lm_head.weight"
        if "lm_head.weight" in target_state_dict
        else "model.embed_tokens.weight"
    )
    model.lm_head.weight.data.copy_(target_state_dict[lm_head_key])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--output", default="qwen3_4b_dflash_draft.pte")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--max-ctx-len", type=int, default=4096)
    args = parser.parse_args()

    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto")
    model = load_draft_model(args.draft_model, target.state_dict())
    model.eval()
    del target

    # Quantize the draft model to match the target model.
    # Keeping both models at the same precision reduces memory usage and helps keep their predictions consistent, which is important for achieving a high draft acceptance rate.
    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=32,
        qembedding_config="4w",
        qembedding_group_size=32,
        tie_word_embeddings=False,
    )

    block_size, ctx_len = args.block_size, args.ctx_len
    hidden_size = model.fc.in_features
    tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
    target_hidden = torch.randn(1, ctx_len, hidden_size)
    position_ids = torch.arange(ctx_len + block_size).unsqueeze(0)

    ctx_dim = Dim("ctx_len", min=1, max=args.max_ctx_len)
    dynamic_shapes = {
        "tokens": None,
        "target_hidden": {1: ctx_dim},
        "position_ids": {1: ctx_dim + block_size},
    }

    import torch.fx.experimental._config as fx_config

    with fx_config.patch(backed_size_oblivious=True):
        exported = torch.export.export(
            model, (tokens, target_hidden, position_ids), dynamic_shapes=dynamic_shapes
        )

    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.exir import to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(exported, partitioner=[MLXPartitioner()])
    et_program = edge.to_executorch()

    with open(args.output, "wb") as f:
        f.write(et_program.buffer)
    print(f"Saved draft model to: {args.output}")
    print(
        f"Dynamic ctx_len supported: 1 to {args.max_ctx_len}, block_size fixed at {block_size}."
    )


if __name__ == "__main__":
    main()
