# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the DFlash target and draft as two methods of one .pte program.

Weights the draft copies from the target (embedding, output projection) are
stored once. The target's hidden-state taps come from the draft checkpoint's
target_layer_ids.
"""

import argparse
from pathlib import Path

import torch

from executorch.backends.mlx.examples.llm.dflash.adapters import get_adapter
from executorch.backends.mlx.examples.llm.dflash.model import (
    DFlashDraftModel,
    load_dflash_config,
)
from huggingface_hub import snapshot_download
from torch.export import Dim
from transformers import AutoModelForCausalLM


def load_draft_model(draft_id: str, target_state_dict: dict) -> DFlashDraftModel:
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    config = load_dflash_config(path)
    model = DFlashDraftModel(config)
    get_adapter(config.model_type).load_draft_weights(model, path, target_state_dict)
    return model


def main():
    from executorch.backends.mlx.llm.hf_attention import register_mlx_attention

    register_mlx_attention()

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--output", default="qwen3_4b_dflash.pte")
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Largest block the draft is traced for, i.e. 1 + the most tokens it "
        "can speculate. Defaults to the draft checkpoint's trained block_size; "
        "a smaller value trades speculation depth for a cheaper export.",
    )
    parser.add_argument(
        "--max-ctx-len",
        type=int,
        default=4096,
        help="Context length / cache capacity for both target and draft.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--target-custom-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the target with mlx::custom_sdpa, matching the draft's attention dispatch.",
    )
    parser.add_argument(
        "--target-custom-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the target with the MLX KV cache.",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=8,
        help="Example ctx len for tracing draft (dynamic).",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Max tokens per forward step. Bounds the traced seq_len dimension "
        "and, for sliding-window models, sizes the ring buffer as "
        "window + chunk - 1 via max_write_len.",
    )

    from executorch.backends.mlx.llm.quantization import add_quantization_args

    add_quantization_args(parser)
    parser.set_defaults(
        qlinear="4w",
        qembedding="4w",
        qlinear_group_size=32,
        qembedding_group_size=32,
    )
    args = parser.parse_args()

    max_ctx_len = args.max_ctx_len
    prefill_chunk_size = args.prefill_chunk_size

    qlinear = args.qlinear
    qembedding = args.qembedding
    qlinear_group_size = args.qlinear_group_size
    qembedding_group_size = args.qembedding_group_size

    draft_path = Path(
        snapshot_download(args.draft_model, allow_patterns=["*.safetensors", "*.json"])
    )
    draft_config = load_dflash_config(draft_path)
    tap_layers = list(draft_config.target_layer_ids)
    print(f"Draft checkpoint requires target tap layers: {tap_layers}")

    # The block size the draft was trained for. Tracing a longer block would
    # feed the draft more mask positions than it ever saw in training.
    if args.block_size is None:
        block_size = draft_config.block_size
    elif not 2 <= args.block_size <= draft_config.block_size:
        raise ValueError(
            f"--block-size {args.block_size} must be in "
            f"[2, {draft_config.block_size}], the block size "
            f"{args.draft_model} was trained for"
        )
    else:
        block_size = args.block_size
    print(
        f"Draft block size: {block_size} "
        f"(checkpoint trained for {draft_config.block_size}); "
        f"speculates up to {block_size - 1} tokens per round"
    )

    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype="auto")
    model = load_draft_model(args.draft_model, target.state_dict())
    model.eval()
    del target

    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
        tie_word_embeddings=False,
    )

    ctx_len = args.ctx_len
    hidden_size = model.fc.in_features
    tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
    block_dim = Dim("block_len", min=2, max=block_size)

    import torch.fx.experimental._config as fx_config

    from executorch.backends.mlx.examples.llm.dflash.cache import DFlashDraftKVCache

    class DFlashCachedDraftModel(torch.nn.Module):
        def __init__(self, draft_model, max_ctx_len):
            super().__init__()
            self.draft = draft_model
            self.cache = DFlashDraftKVCache(
                num_layers=draft_model.config.num_hidden_layers,
                num_heads=draft_model.config.num_key_value_heads,
                head_dim=draft_model.config.head_dim,
                max_ctx_len=max_ctx_len,
            )

        def forward(self, tokens, new_ctx, cache_position):
            return self.draft(
                tokens, new_ctx, cache=self.cache, cache_position=cache_position
            )

    cached_model = DFlashCachedDraftModel(model, max_ctx_len).eval()

    prev_ctx_len_example = 32
    new_ctx = torch.randn(1, ctx_len, hidden_size)
    cache_position = torch.arange(
        prev_ctx_len_example, prev_ctx_len_example + ctx_len, dtype=torch.long
    )

    new_ctx_dim = Dim("new_ctx_len", min=1, max=max_ctx_len)
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

    from executorch.backends.mlx.examples.llm.export_llm_hf import (
        build_hf_exported_program,
    )

    print(
        f"Exporting target {args.target_model} with taps {tap_layers} "
        f"and quant {qlinear}/{qembedding} g={qlinear_group_size}/{qembedding_group_size} "
        f"max_ctx_len {max_ctx_len} prefill_chunk_size {prefill_chunk_size}..."
    )
    target_exported, prefill_chunk_size = build_hf_exported_program(
        model_id=args.target_model,
        revision=None,
        max_ctx_len=max_ctx_len,
        dtype=args.dtype,
        qlinear=qlinear,
        qembedding=qembedding,
        use_custom_sdpa=args.target_custom_sdpa,
        use_custom_kv_cache=args.target_custom_kv_cache,
        qlinear_group_size=qlinear_group_size,
        qembedding_group_size=qembedding_group_size,
        no_tie_word_embeddings=args.no_tie_word_embeddings,
        tap_layers=tap_layers,
        prefill_chunk_size=prefill_chunk_size,
    )

    import executorch.exir as exir
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    constant_methods = {
        "get_max_ctx_len": max_ctx_len,
        "get_prefill_chunk_size": prefill_chunk_size,
        "get_max_block_len": block_size,
        "get_mask_token_id": draft_config.mask_token_id,
    }
    edge = exir.to_edge_transform_and_lower(
        {"target": target_exported, "draft": exported},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )

    with open(args.output, "wb") as f:
        f.write(et_program.buffer)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Saved DFlash program to: {args.output} ({size_mb:.1f} MB)")
    print(f"  methods: {sorted(edge.methods)}")
    print(
        f"  target: max_ctx_len {max_ctx_len}, prefill_chunk_size {prefill_chunk_size}, tap layers {tap_layers}"
    )
    print(
        f"  draft (cached): dynamic new_ctx_len 1 to {max_ctx_len}, "
        f"dynamic block_len 2 to {block_size} (drafts up to {block_size - 1} tokens)."
    )


if __name__ == "__main__":
    main()
