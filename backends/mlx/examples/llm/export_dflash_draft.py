# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the DFlash draft model to a .pte program. 

Loads the pretrained Qwen3 DFlash checkpoint, copies embedding and output projection weights from the target model, applies matching 4-bit quantization, and exports the draft for MLX inference. 
"""

import argparse
from pathlib import Path

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import (
    DFlashDraftModel,
    load_dflash_config,
)
from huggingface_hub import snapshot_download
from torch.export import Dim
from transformers import AutoModelForCausalLM


def load_draft_model(draft_id: str, target_state_dict: dict) -> DFlashDraftModel:
    from executorch.backends.mlx.examples.llm.dflash_qwen3_adapter import (
        load_qwen3_dflash_draft_weights,
    )

    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    config = load_dflash_config(path)
    model = DFlashDraftModel(config)
    load_qwen3_dflash_draft_weights(model, path, target_state_dict)
    return model


def main():
    # Register the MLX attention implementation so DFlashQwen3Attention can use the same attention dispatch as the target model.
    from executorch.backends.mlx.llm.hf_attention import register_mlx_attention

    register_mlx_attention()

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--output", default="qwen3_4b_dflash_draft.pte")
    parser.add_argument("--block-size", type=int, default=16)
    # --ctx-len provides the example shape used during tracing, the dimension remains dynamic.
    # --max-ctx-len sets the upper bound supported at inference time.
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--max-ctx-len", type=int, default=4096)
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Export with a persistent draft KV cache instead of reprojecting the full context each round.",
    )
    args = parser.parse_args()

    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto")
    model = load_draft_model(args.draft_model, target.state_dict())
    model.eval()
    del target

    # Match the target model's 4-bit quantization to reduce memory usage and keep draft predictions aligned with the target for better acceptance.
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

    # Require at least 2 proposal tokens because block_len=1 is not supported by the export shape handling, the runtime skips the draft for that case.
    block_dim = Dim("block_len", min=2, max=block_size)

    import torch.fx.experimental._config as fx_config

    if args.cached:
        from executorch.backends.mlx.examples.llm.dflash_draft_cache import (
            DFlashDraftKVCache,
        )

        class DFlashCachedDraftModel(torch.nn.Module):
            """Wraps the draft model with a persistent KV cache.

            The cache is stored as a module subcomponent so torch.export can preserve its mutable state instead of requiring the cache as a forward argument.
            """

            def __init__(self, draft_model, max_seq_len):
                super().__init__()
                self.draft = draft_model
                self.cache = DFlashDraftKVCache(
                    num_layers=draft_model.config.num_hidden_layers,
                    num_heads=draft_model.config.num_key_value_heads,
                    head_dim=draft_model.config.head_dim,
                    max_seq_len=max_seq_len,
                )

            def forward(self, tokens, new_ctx, cache_position):
                return self.draft(
                    tokens, new_ctx, cache=self.cache, cache_position=cache_position
                )

        cached_model = DFlashCachedDraftModel(model, args.max_ctx_len).eval()

        # Trace using a mid-generation example with existing cached context and a small batch of newly confirmed tokens.
        prev_ctx_len_example = 32
        new_ctx = torch.randn(1, ctx_len, hidden_size)
        cache_position = torch.arange(
            prev_ctx_len_example, prev_ctx_len_example + ctx_len, dtype=torch.long
        )

        new_ctx_dim = Dim("new_ctx_len", min=1, max=args.max_ctx_len)
        dynamic_shapes = {
            "tokens": {1: block_dim},
            "new_ctx": {1: new_ctx_dim},
            "cache_position": {0: new_ctx_dim},
        }

        with fx_config.patch(backed_size_oblivious=True):
            exported = torch.export.export(
                cached_model,
                (tokens, new_ctx, cache_position),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
    else:
        target_hidden = torch.randn(1, ctx_len, hidden_size)
        ctx_dim = Dim("ctx_len", min=1, max=args.max_ctx_len)
        dynamic_shapes = {
            "tokens": {1: block_dim},
            "target_hidden": {1: ctx_dim},
        }

        with fx_config.patch(backed_size_oblivious=True):
            exported = torch.export.export(
                model, (tokens, target_hidden), dynamic_shapes=dynamic_shapes
            )

    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.exir import to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(exported, partitioner=[MLXPartitioner()])
    et_program = edge.to_executorch()

    with open(args.output, "wb") as f:
        f.write(et_program.buffer)
    print(f"Saved draft model to: {args.output}")
    if args.cached:
        print(
            f"Cached draft: dynamic new_ctx_len 1 to {args.max_ctx_len}, "
            f"dynamic block_len 2 to {block_size}."
        )
    else:
        print(
            f"Dynamic ctx_len supported: 1 to {args.max_ctx_len}, dynamic block_len supported: 2 to {block_size}."
        )


if __name__ == "__main__":
    main()
