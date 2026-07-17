"""Quantize Qwen 3.5 MoE and save as a self-contained safetensors bundle.

Runs quantization once and saves the result so export.py can skip
re-quantizing via --prequantized. The output directory contains everything
needed to load the model — no reference to the original HF checkpoint required.

Two backends are supported, producing bundles for the matching export backend:
  --backend cuda (default): packed-INT4 experts for the fused_moe Triton kernel
    plus tile_packed_to_4d dense layers. Loaded by export.py --backend cuda.
  --backend mlx: applies the MLX source transforms (SwitchMLP experts, MLX KV
    cache, etc.), quantizes via torchao, and packs SwitchLinear weights. Loaded
    by export.py --backend mlx. Note: --hqq has no effect here; the MLX 4w/8w
    configs always use torchao's hqq_scale_only qparams internally.

Output:
  output_dir/
    model.safetensors       # quantized weights (with reconstruction metadata in header)
    config.json             # model architecture config
    tokenizer.json          # tokenizer (for runtime)
    tokenizer_config.json
    merges.txt
    vocab.json

Usage:
  python quantize_and_save.py --model-dir /path/to/Qwen3.5-MoE-A3B --qlinear 4w
  python quantize_and_save.py --model-dir /path/to/model --qlinear 4w --hqq
  python quantize_and_save.py --model-dir /path/to/model --backend mlx --qlinear 4w
"""

import argparse
import json
import os
import shutil

import torch

from executorch.examples.models.qwen3_5_moe.export import _quantize
from executorch.examples.models.qwen3_5_moe.model import Qwen35MoE
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Safetensors roundtrip for quantized models
#
# Tensor subclasses (Int4Tensor, Int4TilePackedTo4dTensor,
# IntxUnpackedToInt8Tensor) can't be stored directly in safetensors, so we use
# torchao's flatten/unflatten helpers. They deconstruct each subclass into its
# plain inner tensors plus JSON reconstruction metadata in the safetensors
# header. This matches the bundle format used by gemma4's quantize_and_save.
# ---------------------------------------------------------------------------


def save_quantized_tensors(items, safetensors_path):
    """Flatten an iterable of ``(key, tensor)`` and write a safetensors bundle.

    Tensor subclasses are flattened via torchao's ``flatten_tensor_state_dict``.
    ``nn.Parameter`` values are unwrapped to their tensor data (bare subclasses
    are left as-is). Duplicate keys and meta / None tensors are skipped.
    """
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )

    state_dict = {}
    for key, val in items:
        if key in state_dict:
            continue
        if val is None or val.device.type == "meta":
            continue
        if isinstance(val, torch.nn.Parameter):
            val = val.data
        state_dict[key] = val

    tensors_data, metadata = flatten_tensor_state_dict(state_dict)
    save_file(tensors_data, safetensors_path, metadata=metadata)
    return len(state_dict)


def save_quantized_model(model, safetensors_path):
    """Save a quantized model to safetensors with subclass reconstruction metadata.

    Iterates named_parameters and named_buffers directly (no state_dict copy)
    to avoid doubling peak memory.
    """
    items = list(model.named_parameters()) + list(model.named_buffers())
    return save_quantized_tensors(items, safetensors_path)


def load_quantized_state_dict(safetensors_path):
    """Load a quantized state dict from safetensors, reconstructing tensor subclasses.

    Returns a state dict with plain tensors and reconstructed tensor subclasses
    ready for model.load_state_dict(state_dict, strict=False, assign=True).
    """
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        flat_tensors = {key: f.get_tensor(key) for key in f.keys()}

    state_dict, _ = unflatten_tensor_state_dict(flat_tensors, metadata)
    return state_dict


# ---------------------------------------------------------------------------
# Streaming MLX quantization
#
# Quantizes one decoder layer at a time so peak memory stays at ~one bf16 layer
# instead of the whole model. Each layer's weights are read lazily from the
# checkpoint shards, run through the exact same MLX pipeline used for a
# full-model export (source transforms -> torchao quant -> pack), then the bf16
# is released and only the (much smaller) quantized tensors are kept. The bundle
# is byte-compatible with the full-model path and loads via
# export.load_prequantized_model_mlx.
# ---------------------------------------------------------------------------


def _open_checkpoint_shards(model_dir):
    """Return (weight_map, get_handle) for lazy per-tensor checkpoint access.

    weight_map maps checkpoint key -> shard filename. get_handle(shard) returns
    a cached safetensors handle (mmap; get_tensor copies only the requested
    tensor, so only one tensor is materialized at a time).
    """
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"No safetensors checkpoint in {model_dir}")
        with safe_open(single, framework="pt", device="cpu") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    handles = {}

    def get_handle(shard):
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(model_dir, shard), framework="pt", device="cpu"
            )
        return handles[shard]

    return weight_map, get_handle


def _load_remapped_subset(weight_map, get_handle, config, predicate):
    """Load + remap the checkpoint keys whose normalized name matches predicate.

    Returns a state dict in export-model key space (qkv / gate_up / experts
    fused), the same structure Qwen35MoE.from_hf_checkpoint produces.
    """
    from executorch.examples.models.qwen3_5_moe.model import (
        _fuse_projection_weights,
        _process_checkpoint_key,
    )

    sd = {}
    expert_weights = {}
    for ckpt_key, shard in weight_map.items():
        norm_key = ckpt_key.replace("model.language_model.", "model.", 1)
        if not predicate(norm_key):
            continue
        tensor = get_handle(shard).get_tensor(ckpt_key)
        _process_checkpoint_key(ckpt_key, tensor, sd, expert_weights)

    # Stack per-expert weights (alternative checkpoint format) into [E, N, K].
    if expert_weights:
        for layer_idx in range(config.num_hidden_layers):
            gate = [
                expert_weights.get((layer_idx, "gate", e))
                for e in range(config.num_experts)
            ]
            up = [
                expert_weights.get((layer_idx, "up", e))
                for e in range(config.num_experts)
            ]
            down = [
                expert_weights.get((layer_idx, "down", e))
                for e in range(config.num_experts)
            ]
            if gate[0] is not None:
                sd[f"layers.{layer_idx}.mlp.experts.w1_weight"] = torch.cat(
                    [torch.stack(gate), torch.stack(up)], dim=1
                )
            if down[0] is not None:
                sd[f"layers.{layer_idx}.mlp.experts.w2_weight"] = torch.stack(
                    down, dim=0
                )

    _fuse_projection_weights(sd, config)
    return sd


def _materialize_meta_buffers(module):
    """Replace meta buffers with CPU zeros (mirrors the full export path)."""
    for fqn, buf in list(module.named_buffers()):
        if buf.device.type == "meta":
            parts = fqn.rsplit(".", 1)
            parent = module.get_submodule(parts[0]) if len(parts) > 1 else module
            parent.register_buffer(
                parts[-1], torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
            )


def _mlx_quant_recipe(config, group_size):
    """gemma4-style recipe for Qwen 3.5 MoE (applied to source-transformed model).

    int4 min/max for all linear weights, int8 per-axis for the embedding, and
    everything else (RMSNorm weights, conv1d, biases, A_log, _rms_weight, ...)
    left unquantized. Matched with re.fullmatch. Uses min/max (single-pass) —
    HQQ's iterative scale search is far too slow across per-expert linears.
    """
    from executorch.examples.models.gemma4_31b.quant import (
        QuantConfig,
        QuantRecipe,
        QuantRule,
    )

    int4 = QuantConfig(bits=4, group_size=group_size, symmetric=False, method="min_max")
    int8 = QuantConfig(
        bits=8, group_size=config.hidden_size, symmetric=True, method="min_max"
    )

    return QuantRecipe(
        rules=[
            QuantRule(r".*embed_tokens\.weight", int8),
            QuantRule(r".*(ln_1|ln_2|q_norm|k_norm|norm)\.weight", None),
            QuantRule(r".*conv1d\.weight", None),
            QuantRule(r".*\.weight", int4),
        ]
    )


def stream_quantize_and_save_mlx(model_dir, config, args, safetensors_path):
    """Quantize + save an MLX bundle one decoder layer at a time (low memory).

    Uses gemma4's ``quantize_model`` to produce ``Int4Tensor`` /
    ``IntxUnpackedToInt8Tensor`` subclasses. Packing is deferred to load
    (``pack_all_switch_linears``). Peak memory is ~one bf16 decoder layer.
    """
    import torch.nn as nn

    from executorch.examples.models.gemma4_31b.quant import quantize_model
    from executorch.examples.models.qwen3_5_moe.mlx_source_transformations import (
        mlx_source_transformations,
    )
    from executorch.examples.models.qwen3_5_moe.model import Block, GemmaRMSNorm

    recipe = _mlx_quant_recipe(config, args.qlinear_group_size)
    weight_map, get_handle = _open_checkpoint_shards(model_dir)
    accum = []  # (key, tensor)

    # --- Decoder layers, one at a time ---
    for i in range(config.num_hidden_layers):
        layer_sd = _load_remapped_subset(
            weight_map,
            get_handle,
            config,
            lambda n, i=i: n.startswith(f"model.layers.{i}."),
        )
        prefix = f"layers.{i}."
        block_sd = {
            k[len(prefix) :]: v for k, v in layer_sd.items() if k.startswith(prefix)
        }
        del layer_sd

        with torch.device("meta"):
            block = Block(config, layer_idx=i)
        block.load_state_dict(block_sd, strict=False, assign=True)
        del block_sd
        block = block.to(torch.bfloat16)
        _materialize_meta_buffers(block)
        block.eval()

        # Source-transform (experts -> per-expert nn.Linear), then quantize the
        # nn.Linears via the recipe. Packing happens on load.
        mlx_source_transformations(
            block,
            model_dtype=torch.bfloat16,
            config=config,
            sort_experts=True,
            fuse_gate_up=False,
        )
        for name, val in quantize_model(block, recipe).items():
            accum.append((prefix + name, val))
        del block
        print(
            f"  Streamed layer {i + 1}/{config.num_hidden_layers}", end="\r", flush=True
        )
    print()

    # --- Top-level: embed_tokens (int8), norm (RMS), lm_head (int4) ---
    top_sd = _load_remapped_subset(
        weight_map,
        get_handle,
        config,
        lambda n: n in ("model.embed_tokens.weight", "model.norm.weight")
        or n == "lm_head.weight",
    )
    embed_w = top_sd["embed_tokens.weight"].to(torch.bfloat16)
    norm_w = top_sd["norm.weight"].to(torch.bfloat16)
    lm_w = top_sd.get("lm_head.weight")
    lm_w = embed_w.clone() if lm_w is None else lm_w.to(torch.bfloat16)
    del top_sd

    class _Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    top = _Top().to(torch.bfloat16)
    with torch.no_grad():
        top.embed_tokens.weight.copy_(embed_w)
        top.norm.weight.copy_(norm_w)
        top.lm_head.weight.copy_(lm_w)
    # Matches _swap_rms_norm: precompute (1 + weight) used by F.rms_norm.
    top.norm._rms_weight = nn.Parameter(1.0 + top.norm.weight.data)

    for name, val in quantize_model(top, recipe).items():
        accum.append((name, val))

    print("Writing quantized weights...")
    return save_quantized_tensors(accum, safetensors_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.5 MoE and save as safetensors"
    )
    parser.add_argument(
        "--model-dir", required=True, help="HuggingFace model directory"
    )
    parser.add_argument(
        "--output",
        default="qwen35_moe_quantized",
        help="Output directory (default: qwen35_moe_quantized/)",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "mlx"],
        help="Target backend for the bundle: cuda (default) or mlx.",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096, help="KV cache length")
    parser.add_argument(
        "--qlinear",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantize linear layers.",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=None,
        help="Group size for linear quantization "
        "(default: 64 for mlx per README recommendation, 32 for cuda).",
    )
    parser.add_argument(
        "--qembedding", default=None, choices=["8w"], help="Quantize embedding layers."
    )
    parser.add_argument(
        "--qembedding-group-size",
        type=int,
        default=None,
        help="Group size for embedding quantization (mlx backend).",
    )
    parser.add_argument(
        "--hqq",
        action="store_true",
        help="Use HQQ scale-only optimization for expert quantization "
        "(cuda backend only; ignored for mlx).",
    )
    args = parser.parse_args()

    if not args.qlinear and not args.qembedding:
        parser.error("At least one of --qlinear or --qembedding is required.")

    # Resolve the linear group size: the README recommends 64 for MLX, while
    # CUDA uses the 32 default. An explicit --qlinear-group-size overrides this.
    if args.qlinear_group_size is None:
        args.qlinear_group_size = 64 if args.backend == "mlx" else 32
        print(
            f"Using default --qlinear-group-size {args.qlinear_group_size} "
            f"for --backend {args.backend}."
        )

    if args.backend == "mlx" and args.hqq:
        print(
            "Note: --hqq is ignored for --backend mlx "
            "(MLX quant configs use hqq_scale_only qparams internally)."
        )

    os.makedirs(args.output, exist_ok=True)
    safetensors_path = os.path.join(args.output, "model.safetensors")

    if args.backend == "mlx":
        # Stream layer-by-layer so peak memory is ~one bf16 layer, not the whole
        # model. Reads config only; weights are pulled lazily from the shards.
        from executorch.examples.models.qwen3_5_moe.model import Qwen35MoEConfig

        config = Qwen35MoEConfig.from_hf_config(
            os.path.join(args.model_dir, "config.json")
        )
        config.max_seq_len = args.max_seq_len
        print(
            f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
            f"{config.num_experts} experts top-{config.num_experts_per_tok}"
        )
        print("Streaming + quantizing one layer at a time...")
        n_tensors = stream_quantize_and_save_mlx(
            args.model_dir, config, args, safetensors_path
        )
    else:
        # CUDA: load the full model, then quantize.
        print("Loading model...")
        model, config = Qwen35MoE.from_hf_checkpoint(
            args.model_dir, max_seq_len=args.max_seq_len
        )
        model.eval()
        print(
            f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
            f"{config.num_experts} experts top-{config.num_experts_per_tok}"
        )
        _quantize(model, config, args)
        print("Saving quantized weights...")
        n_tensors = save_quantized_model(model, safetensors_path)

    # Copy config and tokenizer from source
    for filename in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    ]:
        src = os.path.join(args.model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {n_tensors} tensors ({size_mb:.1f} MB) to {args.output}/")
    if args.backend == "mlx":
        print(
            f"Done. Use with: python export.py --backend mlx "
            f"--prequantized {args.output}"
        )
    else:
        print(f"Done. Use with: python export.py --prequantized {args.output}")


if __name__ == "__main__":
    main()
