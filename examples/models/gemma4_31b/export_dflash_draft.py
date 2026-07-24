# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exports the Gemma4-31B DFlash draft model to a .pte program.

Mirrors examples/models/qwen3/export_dflash_draft.py's flow (DFlashDraftModel
and load_dflash_config are fully generic -- no Gemma4-specific changes
needed there), but pulls the shared embed_tokens/lm_head weights from the
*prequantized* target checkpoint instead of loading the full bf16 target
model via AutoModelForCausalLM. For a 31B model, loading the full bf16
target (~62GB resident) just to copy two weight tensors before discarding
it would resurrect the exact memory problem the prequantized-checkpoint
export path was chosen to avoid.

embed_tokens and lm_head are stored as separately-quantized tensors in the
prequantized checkpoint (confirmed via direct safetensors key inspection:
embed_tokens._weight_qdata/_scale/_zero_point and lm_head._weight_qdata/...
both present) -- config.json's tie_word_embeddings: true is stale/does not
reflect this; do not rely on it. Each is unflattened back into its torchao
tensor subclass via unflatten_tensor_state_dict (same utility
quant/pack_mlx.py's load_and_pack_for_mlx uses) and dequantized to bf16
before being copied into the draft model's own embed_tokens/lm_head.
"""

import argparse
import json

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import (
    DFlashDraftModel,
    load_dflash_config,
)
from executorch.examples.models.gemma4_31b.quant.quantize import dequantize_weight
from huggingface_hub import snapshot_download
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file
from torch.export import Dim


def _load_dequantized_tensor(
    safetensors_path: str, logical_name: str, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Load one named weight from a torchao-quantized safetensors checkpoint,
    reconstructing the tensor subclass then dequantizing to a dense tensor.

    logical_name is a dotted module path, e.g. "embed_tokens.weight" or
    "lm_head.weight" -- matches the tensor_names entries in the checkpoint's
    safetensors metadata, not the raw on-disk key prefixes.
    """
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        all_keys = list(f.keys())
        tensor_names = json.loads(metadata.get("tensor_names", "[]"))
        if logical_name not in tensor_names:
            raise KeyError(
                f"{logical_name!r} not found in checkpoint tensor_names "
                f"(have: {[n for n in tensor_names if 'embed' in n or 'lm_head' in n]})"
            )
        parts = logical_name.rsplit(".", 1)
        module_fqn, weight_name = parts[0], parts[-1]
        prefix = f"{module_fqn}._{weight_name}_"
        partial = {k: f.get_tensor(k) for k in all_keys if k.startswith(prefix)}
        if not partial:
            raise KeyError(
                f"No keys found with prefix {prefix!r} for {logical_name!r}"
            )
        result, _ = unflatten_tensor_state_dict(partial, metadata)

    reconstructed = result[logical_name]
    return dequantize_weight(reconstructed, dtype=dtype)


def load_draft_model_from_prequantized_target(
    draft_id: str, target_prequantized_dir: str, max_ctx_len: int = 4096
) -> DFlashDraftModel:
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    config = load_dflash_config(path)
    model = DFlashDraftModel(config, max_ctx_len=max_ctx_len)

    draft_weights = {}
    for f in path.glob("*.safetensors"):
        draft_weights.update(load_file(str(f)))

    missing, unexpected = model.load_state_dict(draft_weights, strict=False)
    assert not unexpected, f"Unexpected draft checkpoint keys: {unexpected}"
    still_missing = [
        k for k in missing if not k.startswith(("embed_tokens.", "lm_head."))
    ]
    assert not still_missing, f"Missing draft checkpoint keys: {still_missing}"

    target_safetensors = f"{target_prequantized_dir}/model.safetensors"
    print(f"Dequantizing embed_tokens.weight from {target_safetensors}...")
    embed_weight = _load_dequantized_tensor(target_safetensors, "embed_tokens.weight")
    model.embed_tokens.weight.data.copy_(embed_weight)

    print(f"Dequantizing lm_head.weight from {target_safetensors}...")
    lm_head_weight = _load_dequantized_tensor(target_safetensors, "lm_head.weight")
    model.lm_head.weight.data.copy_(lm_head_weight)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-prequantized",
        default="./gemma-4-31B-it-HQQ-INT4",
        help="Directory with the prequantized target checkpoint "
        "(model.safetensors + config.json) -- used only to source "
        "embed_tokens/lm_head weights, not to build the full target model.",
    )
    parser.add_argument("--draft-model", default="z-lab/gemma-4-31B-it-DFlash")
    parser.add_argument("--output", default="gemma4_31b_dflash_draft.pte")
    parser.add_argument("--block-size", type=int, default=16)
    # --ctx-len only seeds the example shape used for tracing (ctx_len is a
    # dynamic dim below via Dim("ctx_len", ...)); it is not a runtime cap.
    # --max-ctx-len is the actual bound on context length at inference time.
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--max-ctx-len", type=int, default=4096)
    args = parser.parse_args()

    model = load_draft_model_from_prequantized_target(
        args.draft_model, args.target_prequantized, max_ctx_len=args.max_ctx_len
    )
    model.eval()

    # Quantize the draft model to match the target model's precision --
    # keeping both at the same precision reduces memory and helps keep
    # their predictions consistent, important for a high acceptance rate.
    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=32,
        qembedding_config="4w",
        qembedding_group_size=32,
        tie_word_embeddings=False,
    )

    block_size, new_ctx_len = args.block_size, args.ctx_len
    hidden_size = model.fc.in_features
    tokens = torch.randint(0, 1000, (1, block_size), dtype=torch.long)
    # Example trace input: a single "new chunk" of hidden states, NOT the
    # full accumulated context -- the model now caches everything older
    # internally (see dflash_draft_model.py's DFlashAttention.ctx_cache).
    new_target_hidden = torch.randn(1, new_ctx_len, hidden_size)
    ctx_start_pos = torch.tensor([0], dtype=torch.long)

    new_len_dim = Dim("new_ctx_len", min=1, max=args.max_ctx_len)
    dynamic_shapes = {
        "tokens": None,
        "new_target_hidden": {1: new_len_dim},
        "ctx_start_pos": None,
    }

    import torch.fx.experimental._config as fx_config

    with fx_config.patch(backed_size_oblivious=True):
        exported = torch.export.export(
            model,
            (tokens, new_target_hidden, ctx_start_pos),
            dynamic_shapes=dynamic_shapes,
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
